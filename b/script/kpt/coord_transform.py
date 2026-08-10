"""Coordinate transforms for GeoPredict voxel space."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import VOXEL_CENTER, VOXEL_RANGE_MAX, VOXEL_RANGE_MIN


def compute_auto_offset(
    global_min: np.ndarray,
    global_max: np.ndarray,
    target_center: np.ndarray = VOXEL_CENTER,
) -> np.ndarray:
    workspace_center = (global_min + global_max) / 2.0
    return (workspace_center - target_center).astype(np.float32)


def apply_offset(keypoints: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (keypoints - offset).astype(np.float32)


def validate_range(
    keypoints: np.ndarray,
    range_min: np.ndarray = VOXEL_RANGE_MIN,
    range_max: np.ndarray = VOXEL_RANGE_MAX,
) -> Tuple[bool, Dict]:
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.shape[-1] != 3:
        kpts = kpts.reshape(-1, 3)

    actual_min = kpts.min(axis=0)
    actual_max = kpts.max(axis=0)
    in_range = (
        (actual_min >= range_min).all()
        and (actual_max <= range_max).all()
    )
    out_of_range_count = int(
        np.logical_or(
            (kpts < range_min).any(axis=-1),
            (kpts > range_max).any(axis=-1),
        ).sum()
    )
    stats = {
        "actual_min": actual_min.tolist(),
        "actual_max": actual_max.tolist(),
        "out_of_range_count": out_of_range_count,
        "total_points": int(kpts.shape[0]),
    }
    return in_range, stats
