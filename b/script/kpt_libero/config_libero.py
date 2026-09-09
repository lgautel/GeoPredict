"""Configuration constants for LIBERO-plus RLDS → LeRobot v3.0 conversion."""
from __future__ import annotations

from pathlib import Path
import numpy as np

# ─── Paths ───
RLDS_ROOT = Path("/home/luogang/DATA/libero_plus_rlds/libero_mix/1.0.0")
OUTPUT_ROOT = Path("/home/luogang/DATA/libero_plus_lrb3")
CACHE_DIR = Path("/tmp/libero_plus_cache")
PANDA_XML_PATH = Path("/tmp/panda_robosuite_full.xml")

# ─── Robot model ───
K = 8  # 7 arm links + 1 EEF
KEYPOINT_BODY_NAMES = [
    "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
    "robot0_link5", "robot0_link6", "robot0_link7", "gripper0_eef",
]
KEYPOINT_NAMES = KEYPOINT_BODY_NAMES  # display names = body names
ROBOT_NQPOS = 9   # 7 arm + 2 gripper
SCENE_NQ = 16     # full model (robot + cube)

# ─── Quaternion ───
QUAT_DIM = 4
QUAT_CONVENTION = "wxyz"

# ─── Normalization: auto-offset (GeoPredict) ───
VOXEL_RANGE_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VOXEL_RANGE_MAX = np.array([1.6, 1.6, 1.0], dtype=np.float32)
VOXEL_CENTER = (VOXEL_RANGE_MIN + VOXEL_RANGE_MAX) / 2.0

# ─── Normalization: R_pad (InternVLA) ───
BBOX_MARGIN = 0.15  # 15% safety margin

# ─── Dataset ───
FPS = 20
CHUNKS_SIZE = 1000
TOTAL_SHARDS = 1024

# ─── Video ───
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CRF = "23"  # quality (lower = better, 18-28 typical)

# ─── Workers ───
NUM_WORKERS = 8

# ─── Feature names for stats ───
POS_FEATURE_NAMES = [
    f"{name}_{c}" for name in KEYPOINT_NAMES for c in ("px", "py", "pz")
]
QUAT_FEATURE_NAMES = [
    f"{name}_{c}" for name in KEYPOINT_NAMES for c in ("qw", "qx", "qy", "qz")
]
