from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Policy inputs; joint arrays follow action order, quaternions use wxyz."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_ang_vel_b: np.ndarray
    anchor_quat_w: np.ndarray
