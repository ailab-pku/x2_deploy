from __future__ import annotations

import numpy as np


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = np.asarray(q, dtype=np.float32).copy()
    result[1:] *= -1.0
    return result


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float32)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def relative_orientation(current_w: np.ndarray, target_w: np.ndarray) -> np.ndarray:
    """Target orientation expressed in the current body frame."""
    return quat_multiply(quat_conjugate(current_w), target_w)


def projected_gravity(q_w: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q_w).T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)

