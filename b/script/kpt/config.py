"""Configuration constants for SAPIEN keypoint extraction."""

from pathlib import Path

import numpy as np

ROBOTWIN_ROOT = Path("/home/luogang/share/zwy/Projects/RoboTwin")
URDF_PATH = ROBOTWIN_ROOT / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"

DATASET_DIR = Path("/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three")
OUTPUT_DIR = Path("/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim")

ROBOT_ROOT_POS = np.array([0.0, -0.65, 0.0], dtype=np.float64)
ROBOT_ROOT_QUAT = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float64)  # [w, x, y, z]

K = 14

LEFT_ARM_JOINT_NAMES = [
    "fl_joint1", "fl_joint2", "fl_joint3",
    "fl_joint4", "fl_joint5", "fl_joint6",
]
LEFT_ARM_LINK_NAMES = [
    "fl_link1", "fl_link2", "fl_link3",
    "fl_link4", "fl_link5", "fl_link6",
]
LEFT_EE_JOINT_NAME = "fl_joint6"

RIGHT_ARM_JOINT_NAMES = [
    "fr_joint1", "fr_joint2", "fr_joint3",
    "fr_joint4", "fr_joint5", "fr_joint6",
]
RIGHT_ARM_LINK_NAMES = [
    "fr_link1", "fr_link2", "fr_link3",
    "fr_link4", "fr_link5", "fr_link6",
]
RIGHT_EE_JOINT_NAME = "fr_joint6"

ALL_LINK_NAMES = LEFT_ARM_LINK_NAMES + RIGHT_ARM_LINK_NAMES

KEYPOINT_NAMES = (
    LEFT_ARM_LINK_NAMES + ["fl_eef_tcp"]
    + RIGHT_ARM_LINK_NAMES + ["fr_eef_tcp"]
)

GRIPPER_BIAS = 0.12
GLOBAL_TRANS_MATRIX = np.array(
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
)
DELTA_MATRIX = np.eye(3, dtype=np.float64)

GRIPPER_SCALE = [-0.01, 0.045]
LEFT_GRIPPER_JOINT_NAMES = ["fl_joint7", "fl_joint8"]
RIGHT_GRIPPER_JOINT_NAMES = ["fr_joint7", "fr_joint8"]

LEFT_ARM_STATE_SLICE = slice(0, 6)
LEFT_GRIPPER_STATE_IDX = 6
RIGHT_ARM_STATE_SLICE = slice(7, 13)
RIGHT_GRIPPER_STATE_IDX = 13

VOXEL_RANGE_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VOXEL_RANGE_MAX = np.array([1.6, 1.6, 1.0], dtype=np.float32)
VOXEL_CENTER = (VOXEL_RANGE_MIN + VOXEL_RANGE_MAX) / 2

# ===== Quaternion output config (Appendix 2) =====
QUAT_DIM = 4
QUAT_CONVENTION = "wxyz"  # [w, x, y, z] Hamilton convention (SAPIEN / transforms3d)
QUAT_FILENAME = "keypoint_quats.npy"
