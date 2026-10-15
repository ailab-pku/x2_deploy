from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .controller import RobotState
from .math_utils import quat_to_matrix, relative_orientation
from .x2_ultra import X2UltraConfig


OBSERVATION_TERMS = (
    "command", "motion_anchor_ori_b", "base_ang_vel",
    "joint_pos", "joint_vel", "actions",
)
OBSERVATION_DIM = 164


@dataclass(frozen=True)
class MotionReference:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray


class BeyondMimicPolicy:
    """ONNX runner matching the project's state-estimation-free policy layout."""

    def __init__(self, model_path: str | Path, robot: X2UltraConfig):
        self.robot = robot
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._validate_model()
        self.last_action = np.zeros(len(robot.joint_names), dtype=np.float32)
        self.frame = 0
        self.reference = self._query_reference(0)

        action_index = {name: i for i, name in enumerate(robot.joint_names)}
        self.obs_from_action = np.asarray(
            [action_index[name] for name in robot.observation_joint_names], dtype=np.int64
        )
        self.default_obs_order = robot.default_joint_pos[self.obs_from_action]
        self.anchor_index = self.body_names.index(robot.anchor_body_name)
        self.root_index = self.body_names.index(robot.root_body_name)

    @staticmethod
    def _metadata(session: ort.InferenceSession) -> dict[str, str]:
        return dict(session.get_modelmeta().custom_metadata_map)

    @staticmethod
    def _split(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def _validate_model(self) -> None:
        inputs = {item.name: item.shape for item in self.session.get_inputs()}
        outputs = {item.name: item.shape for item in self.session.get_outputs()}
        if inputs.get("obs") != [1, OBSERVATION_DIM] or "time_step" not in inputs:
            raise ValueError(f"Expected ONNX inputs obs[1,{OBSERVATION_DIM}] and time_step; got {inputs}")
        required_outputs = {
            "actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"
        }
        missing = required_outputs.difference(outputs)
        if missing:
            raise ValueError(f"ONNX is not a motion-export policy; missing outputs: {sorted(missing)}")

        meta = self._metadata(self.session)
        terms = self._split(meta.get("observation_names", ""))
        if tuple(terms) != OBSERVATION_TERMS:
            raise ValueError(f"Observation terms mismatch: expected {OBSERVATION_TERMS}, got {terms}")
        model_joints = self._split(meta.get("joint_names", ""))
        if tuple(model_joints) != self.robot.observation_joint_names:
            raise ValueError("ONNX Isaac Lab joint order does not match this X2 Ultra configuration")
        self.body_names = self._split(meta.get("body_names", ""))
        if self.robot.anchor_body_name not in self.body_names or self.robot.root_body_name not in self.body_names:
            raise ValueError("ONNX body metadata does not contain the configured root/anchor bodies")

        scale = np.asarray([float(x) for x in self._split(meta.get("action_scale", ""))])
        if scale.shape != self.robot.action_scale.shape or not np.allclose(scale, self.robot.action_scale, atol=5e-4):
            raise ValueError("ONNX action_scale metadata does not match the deployment configuration")

    def _run(self, observation: np.ndarray, time_step: int) -> list[np.ndarray]:
        return self.session.run(
            ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"],
            {
                "obs": observation.reshape(1, -1).astype(np.float32, copy=False),
                "time_step": np.asarray([[time_step]], dtype=np.float32),
            },
        )

    def _query_reference(self, frame: int) -> MotionReference:
        dummy = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        _, joint_pos, joint_vel, body_pos, body_quat = self._run(dummy, frame)
        return MotionReference(joint_pos[0], joint_vel[0], body_pos[0], body_quat[0])

    def reset(self) -> MotionReference:
        self.last_action.fill(0.0)
        self.frame = 0
        self.reference = self._query_reference(0)
        return self.reference

    def initial_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.reference
        return (
            ref.body_pos_w[self.root_index].copy(),
            ref.body_quat_w[self.root_index].copy(),
            ref.joint_pos.copy(),
        )

    def observation(self, state: RobotState) -> np.ndarray:
        ref = self.reference
        relative_quat = relative_orientation(
            state.anchor_quat_w, ref.body_quat_w[self.anchor_index]
        )
        # Isaac Lab selects the first two matrix columns, then flattens row-major.
        anchor_orientation_6d = quat_to_matrix(relative_quat)[:, :2].reshape(-1)
        joint_pos_obs = state.joint_pos[self.obs_from_action] - self.default_obs_order
        joint_vel_obs = state.joint_vel[self.obs_from_action]
        obs = np.concatenate(
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
        if obs.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Built observation has shape {obs.shape}, expected ({OBSERVATION_DIM},)")
        return obs

    def infer(self, state: RobotState) -> np.ndarray:
        obs = self.observation(state)
        # The actor depends only on obs. Asking for frame+1 in the same call caches
        # the next reference and avoids running the large ONNX graph twice per tick.
        action, joint_pos, joint_vel, body_pos, body_quat = self._run(obs, self.frame + 1)
        self.last_action = action[0].astype(np.float32, copy=True)
        self.reference = MotionReference(joint_pos[0], joint_vel[0], body_pos[0], body_quat[0])
        self.frame += 1
        return self.robot.default_joint_pos + self.robot.action_scale * self.last_action

