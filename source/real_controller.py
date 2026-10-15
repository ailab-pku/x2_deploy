from __future__ import annotations

from functools import partial
from threading import Event, Lock, Thread
import time

import numpy as np
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, SmSystemState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .math_utils import quat_conjugate, quat_multiply, quat_to_matrix
from .state import RobotState
from .x2_ultra import X2UltraConfig


def torso_orientation(pelvis_quat: np.ndarray, waist_pos: np.ndarray) -> np.ndarray:
    """X2 chain: pelvis -> waist yaw (Z) -> pitch (Y) -> torso roll (X)."""
    quat = pelvis_quat
    for angle, axis in zip(waist_pos, (3, 2, 1)):
        rotation = np.zeros(4, dtype=np.float32)
        rotation[0], rotation[axis] = np.cos(angle / 2), np.sin(angle / 2)
        quat = quat_multiply(quat, rotation)
    return quat


class AimDKController(Node):
    """AimDK v1.1 ROS 2 HAL adapter for the original X2 Ultra (head yaw only)."""

    def __init__(
        self, robot: X2UltraConfig, *, dry_run: bool = False,
        imu_quat: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
        state_timeout: float = 0.1,
    ):
        super().__init__("x2_beyond_mimic")
        self.robot, self.dry_run, self.state_timeout = robot, dry_run, state_timeout
        self.kp, self.kd = robot.kp, robot.kd
        self.active = np.asarray([name != "head_pitch_joint" for name in robot.joint_names])
        self.groups = {"leg": list(range(12)), "waist": list(range(12, 15)),
                       "arm": list(range(15, 29)), "head": list(range(29, 31))}
        self.waist_ids = [robot.joint_names.index(f"waist_{axis}_joint") for axis in ("yaw", "pitch", "roll")]
        self.imu_quat = np.asarray(imu_quat, dtype=np.float32)
        norm = np.linalg.norm(self.imu_quat)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError("IMU mounting quaternion must be finite and nonzero (w x y z)")
        self.imu_quat /= norm
        self.imu_rotation = quat_to_matrix(self.imu_quat)
        self.heading = np.asarray([1, 0, 0, 0], dtype=np.float32)
        self.lock, self.stopping = Lock(), Event()
        self.command_lock = Lock()
        self.fault: Exception | None = None
        self.q = np.zeros(len(robot.joint_names), dtype=np.float32)
        self.dq = np.zeros_like(self.q)
        self.pelvis_quat, self.omega = self.heading.copy(), np.zeros(3, dtype=np.float32)
        self.received = {key: 0.0 for key in (*self.groups, "imu", "mode")}
        self.mode, self.mode_status = "", 0
        self.target_time = 0.0
        self.discovery_started = time.monotonic()
        self.hold_on_exit = False
        self.thread: Thread | None = None
        self.publishers_by_group = {}
        self.sequence = 0
        self.subscriptions_by_group = [
            self.create_subscription(
                JointStateArray, f"/aima/hal/joint/{group}/state",
                partial(self._on_joints, group), qos_profile_sensor_data,
            ) for group in self.groups
        ]
        self.imu_subscription = self.create_subscription(
            Imu, "/aima/hal/imu/torso/state", self._on_imu, qos_profile_sensor_data,
        )
        self.mode_subscription = self.create_subscription(
            SmSystemState, "/aima/sm/system_state", self._on_mode, qos_profile_sensor_data,
        )

    def _on_joints(self, group: str, msg: JointStateArray) -> None:
        try:
            # AIMDK message revisions differ: newer arrays may expose a HAL
            # state, while the X2 v1.1 message on the robot has only header/joints.
            state = getattr(msg, "state", None)
            state_value = getattr(state, "value", None)
            if state_value in (1, 2, 3, 4):
                raise RuntimeError(f"HAL {group} fault state: {state_value}")
            ids = self.groups[group]
            # Named feedback may arrive in another order. Empty names use SDK order.
            joints = {joint.name or self.robot.joint_names[i]: joint for i, joint in zip(ids, msg.joints)}
            active_ids = [i for i in ids if self.active[i]]
            samples = [joints[self.robot.joint_names[i]] for i in active_ids]
            values = np.asarray([(joint.position, joint.velocity) for joint in samples], dtype=np.float32)
            if not np.isfinite(values).all() or any(getattr(joint, "error_code", 0) for joint in samples):
                raise RuntimeError(f"Invalid or faulty HAL {group} joint feedback")
            with self.lock:
                self.q[active_ids], self.dq[active_ids] = values.T
                self.received[group] = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.fault = exc

    def _on_imu(self, msg: Imu) -> None:
        q, w = msg.orientation, msg.angular_velocity
        quat = np.asarray([q.w, q.x, q.y, q.z], dtype=np.float32)
        omega = np.asarray([w.x, w.y, w.z], dtype=np.float32)
        norm = np.linalg.norm(quat)
        with self.lock:
            if not np.isfinite(norm) or norm < 1e-6 or not np.isfinite(omega).all():
                self.fault = RuntimeError("Invalid pelvis IMU quaternion or angular velocity")
                return
            # q_BS maps sensor vectors into pelvis axes: q_WB = q_WS * inverse(q_BS).
            self.pelvis_quat = quat_multiply(quat / norm, quat_conjugate(self.imu_quat))
            self.omega = self.imu_rotation @ omega
            self.received["imu"] = time.monotonic()

    def _on_mode(self, msg: SmSystemState) -> None:
        with self.lock:
            self.mode, self.mode_status = msg.cur_state, msg.cur_status.value
            self.received["mode"] = time.monotonic()

    def wait_for_state(self, timeout: float = 10.0) -> None:
        # On the original X2, stopping native MC does not change the SM label.
        # Control ownership is checked against the command publishers in start().
        required = (*self.groups, "imu")
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                missing = [key for key in required if not self.received[key]]
                fault = self.fault
            if fault:
                raise RuntimeError(f"HAL feedback failed: {fault}") from fault
            if not missing:
                self.read_state()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Missing HAL feedback: {', '.join(missing)}")
            time.sleep(0.01)

    def _check_command_ownership(self) -> None:
        allowed = 1 if self.publishers_by_group else 0
        conflicts = {}
        for group in self.groups:
            topic = f"/aima/hal/joint/{group}/command"
            owners = self.get_publishers_info_by_topic(topic)
            foreign = any(owner.node_name != self.get_name()
                          or owner.node_namespace != self.get_namespace() for owner in owners)
            if len(owners) > allowed or (allowed and foreign):
                conflicts[group] = [f"{owner.node_namespace.rstrip('/')}/{owner.node_name}" for owner in owners]
        if conflicts:
            raise RuntimeError(
                f"HAL command publisher conflict: {conflicts}. "
                "On PC1, run 'aima em stop-app mc' before live deployment; "
                "wait for all native command publishers to disappear. "
                f"Current system mode: {self.mode or 'unknown'} / {self.mode_status}"
            )

    def read_state(self) -> RobotState:
        now = time.monotonic()
        with self.lock:
            if self.fault:
                raise RuntimeError(f"Real control stopped: {self.fault}") from self.fault
            stale = [key for key in (*self.groups, "imu") if now - self.received[key] > self.state_timeout]
            if stale:
                raise TimeoutError(f"Stale HAL feedback: {', '.join(stale)}")
            q, dq, omega = self.q.copy(), self.dq.copy(), self.omega.copy()
            pelvis_quat = quat_multiply(self.heading, self.pelvis_quat)
        if not self.dry_run and self.publishers_by_group:
            self._check_command_ownership()
        return RobotState(q, dq, omega, torso_orientation(pelvis_quat, q[self.waist_ids]))

    def align_heading(self, reference_root_quat: np.ndarray) -> None:
        with self.lock:
            reference = quat_to_matrix(reference_root_quat)
            current = quat_to_matrix(self.pelvis_quat)
            yaw = np.arctan2(reference[1, 0], reference[0, 0]) - np.arctan2(current[1, 0], current[0, 0])
            self.heading = np.asarray([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], dtype=np.float32)

    def set_target(self, targets: np.ndarray) -> None:
        """Publish one position/PD command per policy tick; the robot computes torque."""
        if self.dry_run:
            return
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != self.q.shape or not np.isfinite(targets).all():
            raise RuntimeError("Policy produced invalid joint targets")
        with self.command_lock:
            self.read_state()
            if self.stopping.is_set():
                raise RuntimeError("Real control has stopped")
            self._publish(targets)
            with self.lock:
                self.target_time = time.monotonic()

    def check_control_ownership(self) -> None:
        """Read-only preflight; do not create any command publisher."""
        # Allow DDS discovery before deciding whether native MC is absent.
        while time.monotonic() - self.discovery_started < 2.0:
            self.read_state()
            time.sleep(0.01)
        self.read_state()
        self._check_command_ownership()

    def start(self) -> None:
        if self.dry_run:
            return
        if self.publishers_by_group:
            raise RuntimeError("Real control has already started")
        self.check_control_ownership()
        self.publishers_by_group = {
            group: self.create_publisher(
                JointCommandArray, f"/aima/hal/joint/{group}/command", qos_profile_sensor_data,
            ) for group in self.groups
        }
        self.target_time = time.monotonic()
        self.thread = Thread(target=self._watchdog_loop, name="aimdk_watchdog", daemon=True)
        self.thread.start()

    def _publish(self, targets: np.ndarray, *, damping: bool = False) -> None:
        self.sequence = (self.sequence + 1) % (2 ** 32)
        for group, ids in self.groups.items():
            msg = JointCommandArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.sequence = self.sequence
            for i in ids:
                joint = JointCommand()
                joint.name = self.robot.joint_names[i]
                joint.position = float(targets[i]) if self.active[i] and not damping else 0.0
                joint.velocity = joint.effort = 0.0
                joint.stiffness = float(self.kp[i]) if self.active[i] and not damping else 0.0
                joint.damping = float(self.kd[i]) if self.active[i] else 0.0
                msg.joints.append(joint)
            self.publishers_by_group[group].publish(msg)

    def _watchdog_loop(self) -> None:
        """Monitor liveness only; normal commands come directly from set_target()."""
        try:
            while not self.stopping.wait(self.robot.policy_dt):
                self.read_state()
                with self.lock:
                    updated = self.target_time
                if time.monotonic() - updated > self.state_timeout:
                    raise TimeoutError("Policy target expired")
        except Exception as exc:
            with self.lock:
                self.fault = exc
            self.stopping.set()
        finally:
            # Normal completion / operator stop retains position stiffness.
            # Capture the measured pose once, never the last policy target.
            # Faults or stale joint measurements fall back to damping.
            try:
                with self.command_lock:
                    with self.lock:
                        fresh = all(time.monotonic() - self.received[group] <= self.state_timeout
                                    for group in self.groups)
                        hold = self.hold_on_exit and self.fault is None and fresh
                        targets = self.q.copy() if hold else np.zeros_like(self.q)
                    for _ in range(round(0.2 / self.robot.policy_dt)):
                        # Never send exit commands over another controller.
                        self._check_command_ownership()
                        self._publish(targets, damping=not hold)
                        time.sleep(self.robot.policy_dt)
                    self.get_logger().info(
                        "Exit: HAL retains measured joint positions with kp/kd (no balance control)"
                        if hold else "Exit: damping command sent"
                    )
            except Exception as exc:
                self.get_logger().error(f"Could not publish exit command: {exc}")

    def stop(self, *, hold_position: bool = False) -> None:
        with self.lock:
            if not self.stopping.is_set():
                self.hold_on_exit = hold_position
                self.stopping.set()
        if self.thread is not None:
            self.thread.join()
