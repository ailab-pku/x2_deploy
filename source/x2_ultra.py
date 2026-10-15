from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

# Motion command, policy action, and MuJoCo joint order.
ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint", "head_yaw_joint", "head_pitch_joint",
)

# Isaac Lab articulation order. joint_pos and joint_vel policy terms use this order.
OBSERVATION_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint",
    "left_knee_joint", "right_knee_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint",
    "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
)

DEFAULT_BY_NAME = {
    "left_hip_pitch_joint": -0.312, "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669, "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": -0.3, "right_elbow_joint": -0.3,
}

KP_BY_NAME = {
    **{name: 120.0 for name in ("left_hip_pitch_joint", "right_hip_pitch_joint")},
    **{name: 100.0 for name in (
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint")},
    **{name: 150.0 for name in ("left_knee_joint", "right_knee_joint")},
    **{name: 40.0 for name in (
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint")},
    "waist_yaw_joint": 40.18, "waist_pitch_joint": 200.0, "waist_roll_joint": 200.0,
    **{name: 50.0 for name in ACTION_JOINT_NAMES if "shoulder" in name or "elbow" in name},
    **{name: 20.0 for name in ACTION_JOINT_NAMES if "wrist" in name},
    "head_yaw_joint": 20.0, "head_pitch_joint": 20.0,
}

KD_BY_NAME = {
    **{name: 5.0 for name in (
        "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint")},
    **{name: 4.0 for name in (
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint")},
    **{name: 2.0 for name in ACTION_JOINT_NAMES},
    **{name: 3.0 for name in ACTION_JOINT_NAMES if "shoulder" in name or "elbow" in name},
    "left_hip_pitch_joint": 5.0, "right_hip_pitch_joint": 5.0,
    "left_hip_roll_joint": 4.0, "right_hip_roll_joint": 4.0,
    "left_hip_yaw_joint": 4.0, "right_hip_yaw_joint": 4.0,
    "left_knee_joint": 5.0, "right_knee_joint": 5.0,
    "waist_yaw_joint": 2.56,
}

EFFORT_BY_NAME = {
    **{name: 120.0 for name in ACTION_JOINT_NAMES[:12]},
    "left_ankle_pitch_joint": 36.0, "right_ankle_pitch_joint": 36.0,
    "left_ankle_roll_joint": 24.0, "right_ankle_roll_joint": 24.0,
    "waist_yaw_joint": 120.0, "waist_pitch_joint": 48.0, "waist_roll_joint": 48.0,
    **{name: 36.0 for name in ACTION_JOINT_NAMES if "shoulder_pitch" in name or "shoulder_roll" in name},
    **{name: 24.0 for name in ACTION_JOINT_NAMES if "shoulder_yaw" in name or "elbow" in name or "wrist_yaw" in name},
    **{name: 4.8 for name in ACTION_JOINT_NAMES if "wrist_pitch" in name or "wrist_roll" in name},
    "head_yaw_joint": 2.6, "head_pitch_joint": 0.6,
}

# Exact action scale recorded in the exported checkpoint, in ACTION_JOINT_NAMES order.
ACTION_SCALE = np.asarray([
    0.250, 0.300, 0.300, 0.200, 0.225, 0.150,
    0.250, 0.300, 0.300, 0.200, 0.225, 0.150,
    0.747, 0.060, 0.060, 0.180, 0.180, 0.120, 0.120,
    0.300, 0.060, 0.060, 0.180, 0.180, 0.120, 0.120,
    0.300, 0.060, 0.060, 0.000, 0.000,
], dtype=np.float32)


def _array(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values.get(name, 0.0) for name in ACTION_JOINT_NAMES], dtype=np.float32)


@dataclass(frozen=True)
class X2UltraConfig:
    joint_names: tuple[str, ...] = ACTION_JOINT_NAMES
    observation_joint_names: tuple[str, ...] = OBSERVATION_JOINT_NAMES
    anchor_body_name: str = "torso_link"
    root_body_name: str = "pelvis"
    scene_path: Path = PROJECT_ROOT / "assets" / "x2_ultra" / "scene.xml"
    default_policy_path: Path = PROJECT_ROOT / "assets" / "prepare_policy.onnx"
    sim_dt: float = 0.005
    policy_dt: float = 0.02
    initial_settle_seconds: float = 0.5

    @property
    def default_joint_pos(self) -> np.ndarray:
        return _array(DEFAULT_BY_NAME)

    @property
    def kp(self) -> np.ndarray:
        return _array(KP_BY_NAME)

    @property
    def kd(self) -> np.ndarray:
        return _array(KD_BY_NAME)

    @property
    def effort_limit(self) -> np.ndarray:
        return _array(EFFORT_BY_NAME)

    @property
    def action_scale(self) -> np.ndarray:
        return ACTION_SCALE.copy()

    @property
    def decimation(self) -> int:
        ratio = self.policy_dt / self.sim_dt
        if not ratio.is_integer():
            raise ValueError("policy_dt must be an integer multiple of sim_dt")
        return int(ratio)


X2_ULTRA = X2UltraConfig()

