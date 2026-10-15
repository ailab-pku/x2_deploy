from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .math_utils import quat_to_matrix
from .x2_ultra import X2UltraConfig


@dataclass(frozen=True)
class RobotState:
    """Policy inputs; joint arrays follow action order, quaternions use wxyz."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_ang_vel_b: np.ndarray
    anchor_quat_w: np.ndarray


class MujocoController:
    def __init__(self, robot: X2UltraConfig, scene_path: str | Path | None = None):
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_path(str(scene_path or robot.scene_path))
        self.model.opt.timestep = robot.sim_dt
        self.decimation = robot.decimation
        self.kp, self.kd, self.effort_limit = robot.kp, robot.kd, robot.effort_limit

        joint_ids = np.asarray([
            self.model.joint(name).id for name in robot.joint_names
        ])
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[joint_ids]
        self.actuator_ids = np.asarray([
            self.model.actuator(f"motor_{name}").id
            for name in robot.joint_names
        ])
        root_joint_id = self.model.joint("floating_base_joint").id
        self.root_qpos_id = int(self.model.jnt_qposadr[root_joint_id])
        self.root_qvel_id = int(self.model.jnt_dofadr[root_joint_id])
        self.anchor_body_id = self.model.body(robot.anchor_body_name).id
        self.gyro_address = int(self.model.sensor("body-angular-velocity").adr[0])
        self.data = mujoco.MjData(self.model)

    def reset(
        self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray,
        *, joint_vel: np.ndarray, root_lin_vel_w: np.ndarray,
        root_ang_vel_w: np.ndarray,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        root = self.root_qpos_id
        self.data.qpos[root : root + 3] = root_pos
        self.data.qpos[root + 3 : root + 7] = root_quat
        self.data.qpos[self.qpos_ids] = joint_pos
        self.data.qvel[self.qvel_ids] = joint_vel
        root_vel = self.root_qvel_id
        self.data.qvel[root_vel : root_vel + 3] = root_lin_vel_w
        # MuJoCo free-joint linear velocity is world-frame, angular is local.
        self.data.qvel[root_vel + 3 : root_vel + 6] = quat_to_matrix(root_quat).T @ root_ang_vel_w
        mujoco.mj_forward(self.model, self.data)

    def read_state(self) -> RobotState:
        return RobotState(
            joint_pos=self.data.qpos[self.qpos_ids].astype(np.float32, copy=True),
            joint_vel=self.data.qvel[self.qvel_ids].astype(np.float32, copy=True),
            root_ang_vel_b=self.data.sensordata[
                self.gyro_address : self.gyro_address + 3
            ].astype(np.float32, copy=True),
            anchor_quat_w=self.data.xquat[self.anchor_body_id].astype(np.float32, copy=True),
        )

    def step(self, joint_targets: np.ndarray) -> None:
        for _ in range(self.decimation):
            torque = (
                self.kp * (joint_targets - self.data.qpos[self.qpos_ids])
                - self.kd * self.data.qvel[self.qvel_ids]
            )
            self.data.ctrl[self.actuator_ids] = np.clip(torque, -self.effort_limit, self.effort_limit)
            mujoco.mj_step(self.model, self.data)
        # Refresh poses and sensors after integration for the next observation.
        mujoco.mj_forward(self.model, self.data)
