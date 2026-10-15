from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from .x2_ultra import X2UltraConfig


@dataclass(frozen=True)
class RobotState:
    """Backend-neutral policy state; quaternions use wxyz."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat_w: np.ndarray
    root_ang_vel_b: np.ndarray
    anchor_pos_w: np.ndarray
    anchor_quat_w: np.ndarray


class RobotController(ABC):
    """Interface shared by simulation and a future robot backend."""

    @abstractmethod
    def reset(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        pass

    @abstractmethod
    def read_state(self) -> RobotState:
        pass

    @abstractmethod
    def step(self, joint_targets: np.ndarray) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class MujocoController(RobotController):
    def __init__(self, robot: X2UltraConfig, scene_path: str | Path | None = None):
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_path(str(scene_path or robot.scene_path))
        self.model.opt.timestep = robot.sim_dt
        self.data = mujoco.MjData(self.model)
        self._closed = False
        self.paused = False

        self.joint_ids = np.asarray([
            self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.joint_names
        ])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.asarray([
            self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
            for name in robot.joint_names
        ])
        self.root_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.root_qpos_id = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, robot.root_body_name)
        self.anchor_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, robot.anchor_body_name)
        self.gyro_sensor_id = self._required_id(mujoco.mjtObj.mjOBJ_SENSOR, "body-angular-velocity")
        self.gyro_address = int(self.model.sensor_adr[self.gyro_sensor_id])
        self.viewer = None

        if self.model.nu != len(robot.joint_names):
            raise ValueError(f"MJCF has {self.model.nu} actuators, expected {len(robot.joint_names)}")

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo object not found: {name}")
        return object_id

    def reset(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        root = self.root_qpos_id
        self.data.qpos[root : root + 3] = root_pos
        self.data.qpos[root + 3 : root + 7] = root_quat
        self.data.qpos[self.qpos_ids] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def read_state(self) -> RobotState:
        return RobotState(
            joint_pos=self.data.qpos[self.qpos_ids].astype(np.float32, copy=True),
            joint_vel=self.data.qvel[self.qvel_ids].astype(np.float32, copy=True),
            root_quat_w=self.data.xquat[self.root_body_id].astype(np.float32, copy=True),
            root_ang_vel_b=self.data.sensordata[
                self.gyro_address : self.gyro_address + 3
            ].astype(np.float32, copy=True),
            anchor_pos_w=self.data.xpos[self.anchor_body_id].astype(np.float32, copy=True),
            anchor_quat_w=self.data.xquat[self.anchor_body_id].astype(np.float32, copy=True),
        )

    def _physics_step(self, joint_targets: np.ndarray) -> None:
        position_error = joint_targets - self.data.qpos[self.qpos_ids]
        torque = self.robot.kp * position_error - self.robot.kd * self.data.qvel[self.qvel_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(
            torque, -self.robot.effort_limit, self.robot.effort_limit
        )
        mujoco.mj_step(self.model, self.data)

    def step(self, joint_targets: np.ndarray) -> None:
        for _ in range(self.robot.decimation):
            self._physics_step(joint_targets)

    def settle(self, joint_targets: np.ndarray) -> None:
        for _ in range(int(self.robot.initial_settle_seconds / self.robot.sim_dt)):
            self._physics_step(joint_targets)

    def launch_viewer(self) -> None:
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )
        self.viewer.cam.elevation = -20

    def _key_callback(self, keycode: int) -> None:
        if keycode == ord(" "):
            self.paused = not self.paused
            print(f"[MuJoCo] {'paused' if self.paused else 'resumed'}")

    def sync(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    @property
    def is_running(self) -> bool:
        return not self._closed and (self.viewer is None or self.viewer.is_running())

    def wait_while_paused(self) -> None:
        while self.paused and self.is_running:
            self.sync()
            time.sleep(0.01)

    def close(self) -> None:
        self._closed = True
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
