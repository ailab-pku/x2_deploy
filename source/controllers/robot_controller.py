from __future__ import annotations

from functools import partial
from threading import Event, Lock
import time

import numpy as np
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray, SmSystemState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .. import RobotState, quat_multiply
from assets.x2_ultra import X2UltraConfig


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
        self, robot: X2UltraConfig, *,
        state_timeout: float = 0.1,
    ):
        super().__init__("x2_beyond_mimic")
        self.robot, self.state_timeout = robot, state_timeout
        self.kp, self.kd = robot.kp, robot.kd
        self.active = np.asarray([name != "head_pitch_joint" for name in robot.joint_names])
        self.groups = {"leg": list(range(12)), "waist": list(range(12, 15)),
                       "arm": list(range(15, 29)), "head": list(range(29, 31))}
        self.waist_ids = [robot.joint_names.index(f"waist_{axis}_joint") for axis in ("yaw", "pitch", "roll")]
        self.heading = np.asarray([1, 0, 0, 0], dtype=np.float32)
        self.lock, self.stopping = Lock(), Event()
        self.fault: Exception | None = None
        self.q = np.zeros(len(robot.joint_names), dtype=np.float32)
        self.dq = np.zeros_like(self.q)
        self.pelvis_quat, self.omega = self.heading.copy(), np.zeros(3, dtype=np.float32)
        self.received = {key: 0.0 for key in (*self.groups, "imu", "mode")}
        self.mode, self.mode_status = "", 0
        self.discovery_started = time.perf_counter()
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
        ids = self.groups[group]
        # Named feedback may arrive in another order. Empty names use SDK order.
        joints = {joint.name or self.robot.joint_names[i]: joint for i, joint in zip(ids, msg.joints)}
        active_ids = [i for i in ids if self.active[i]]
        samples = [joints[self.robot.joint_names[i]] for i in active_ids]
        values = np.asarray([(joint.position, joint.velocity) for joint in samples], dtype=np.float32)
        with self.lock:
            self.q[active_ids], self.dq[active_ids] = values.T
            self.received[group] = time.perf_counter()


    def _on_imu(self, msg: Imu) -> None:
        q, w = msg.orientation, msg.angular_velocity
        quat = np.asarray([q.w, q.x, q.y, q.z], dtype=np.float32)
        omega = np.asarray([w.x, w.y, w.z], dtype=np.float32)
        norm = np.linalg.norm(quat)
        with self.lock:
            self.pelvis_quat = quat / norm
            self.omega = omega
            self.received["imu"] = time.perf_counter()

    def _on_mode(self, msg: SmSystemState) -> None:
        with self.lock:
            self.mode, self.mode_status = msg.cur_state, msg.cur_status.value
            self.received["mode"] = time.perf_counter()


    def read_state(self) -> RobotState:
        q, dq, omega = self.q.copy(), self.dq.copy(), self.omega.copy()
        pelvis_quat = quat_multiply(self.heading, self.pelvis_quat)
        return RobotState(q, dq, omega, torso_orientation(pelvis_quat, q[self.waist_ids]))

    def set_target(self, targets: np.ndarray) -> None:
        """Publish one position/PD command per policy tick; the robot computes torque."""
        targets = np.asarray(targets, dtype=np.float32)
        self._publish(targets)

    def start(self) -> None:
        self.publishers_by_group = {
            group: self.create_publisher(
                JointCommandArray, f"/aima/hal/joint/{group}/command", qos_profile_sensor_data,
            ) for group in self.groups
        }

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
