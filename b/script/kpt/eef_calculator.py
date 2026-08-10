"""EEF TCP position calculation matching RoboTwin _trans_endpose."""

from __future__ import annotations

import numpy as np
import transforms3d.quaternions as t3d_quat

from .config import DELTA_MATRIX, GLOBAL_TRANS_MATRIX, GRIPPER_BIAS


def compute_tcp_position(
    ee_joint_pos: np.ndarray,
    ee_joint_quat: np.ndarray,
    gripper_bias: float = GRIPPER_BIAS,
    global_trans_matrix: np.ndarray = GLOBAL_TRANS_MATRIX,
    delta_matrix: np.ndarray = DELTA_MATRIX,
) -> np.ndarray:
    """Compute EEF TCP world position from EE joint pose."""
    ee_joint_pos = np.asarray(ee_joint_pos, dtype=np.float64)
    ee_joint_quat = np.asarray(ee_joint_quat, dtype=np.float64)

    rot_ee = t3d_quat.quat2mat(ee_joint_quat)
    rot_tcp = rot_ee @ global_trans_matrix @ delta_matrix
    tcp_offset = rot_tcp @ np.array([gripper_bias, 0.0, 0.0], dtype=np.float64)
    tcp_pos = ee_joint_pos + tcp_offset
    return tcp_pos.astype(np.float32)
