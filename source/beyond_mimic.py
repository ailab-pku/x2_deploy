from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .state import RobotState
from .math_utils import quat_to_matrix, relative_orientation
from .x2_ultra import X2UltraConfig


OBSERVATION_DIM = 164


@dataclass(frozen=True)
class MotionReference:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


class BeyondMimicPolicy:
    """ONNX runner matching the project's state-estimation-free policy layout."""

    def __init__(
        self, model_path: str | Path, robot: X2UltraConfig,
        *, session_options: ort.SessionOptions | None = None,
    ):
        self.robot = robot
        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        self.body_names = self.session.get_modelmeta().custom_metadata_map["body_names"].split(",")
        self.last_action = np.zeros(len(robot.joint_names), dtype=np.float32)

        action_index = {name: i for i, name in enumerate(robot.joint_names)}
        self.obs_from_action = np.asarray(
            [action_index[name] for name in robot.observation_joint_names], dtype=np.int64
        )
        self.default_obs_order = robot.default_joint_pos[self.obs_from_action]
        self.anchor_index = self.body_names.index(robot.anchor_body_name)
        self.root_index = self.body_names.index(robot.root_body_name)
        self.reset()

    def _run(self, observation: np.ndarray, time_step: int) -> list[np.ndarray]:
        return self.session.run(
            ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
             "body_lin_vel_w", "body_ang_vel_w"],
            {
                "obs": observation.reshape(1, -1).astype(np.float32, copy=False),
                "time_step": np.asarray([[time_step]], dtype=np.float32),
            },
        )

    def reset(self) -> MotionReference:
        self.last_action.fill(0.0)
        self.frame = 0
        _, *reference = self._run(np.zeros(OBSERVATION_DIM, dtype=np.float32), 0)
        self.reference = MotionReference(*(value[0] for value in reference))
        return self.reference

    def observation(self, state: RobotState) -> np.ndarray:
        ref = self.reference
        relative_quat = relative_orientation(
            state.anchor_quat_w, ref.body_quat_w[self.anchor_index]
        )
        # Isaac Lab selects the first two matrix columns, then flattens row-major.
        anchor_orientation_6d = quat_to_matrix(relative_quat)[:, :2].reshape(-1)
        joint_pos_obs = state.joint_pos[self.obs_from_action] - self.default_obs_order
        joint_vel_obs = state.joint_vel[self.obs_from_action]
        return np.concatenate(
            (
                ref.joint_pos,
                ref.joint_vel,
                anchor_orientation_6d,
                state.root_ang_vel_b,
                joint_pos_obs,
                joint_vel_obs,
                self.last_action,
            )
        ).astype(np.float32, copy=False)

    def infer(self, state: RobotState) -> np.ndarray:
        obs = self.observation(state)
        # The actor depends only on obs. Asking for frame+1 in the same call caches
        # the next reference and avoids running the large ONNX graph twice per tick.
        action, *reference = self._run(obs, self.frame + 1)
        self.last_action = action[0].astype(np.float32, copy=True)
        self.reference = MotionReference(*(value[0] for value in reference))
        self.frame += 1
        return self.robot.action_offset + self.robot.action_scale * self.last_action

