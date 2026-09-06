"""Keypoint + pose (quaternion) extraction, extending KeypointExtractor."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .config import (
    K,
    LEFT_ARM_LINK_NAMES,
    LEFT_EE_JOINT_NAME,
    QUAT_CONVENTION,
    QUAT_DIM,
    QUAT_FILENAME,
    RIGHT_ARM_LINK_NAMES,
    RIGHT_EE_JOINT_NAME,
)
from .eef_calculator import compute_tcp_pose
from .keypoint_extractor import KeypointExtractor


class KeypointPoseExtractor(KeypointExtractor):
    """Extract 3D keypoint positions AND orientation quaternions.

    Inherits all position-extraction and offset logic from KeypointExtractor.
    Adds per-keypoint quaternion output saved as a separate .npy file
    alongside the existing keypoints.npy.

    Output per episode:
        keypoints.npy       — [T, K*3] float32  (positions, from parent)
        keypoint_quats.npy  — [T, K*4] float32  (quaternions, this class)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quat_cache: Dict[int, np.ndarray] = {}
        self._current_ep_quats: List[np.ndarray] = []

    def _compute_step_keypoints(self, state_14: np.ndarray) -> np.ndarray:
        """Override: also extract link/TCP quaternions per step."""
        qpos = self.joint_mapper.map_state_to_qpos(state_14)
        self.fk_scene.set_qpos(qpos)

        left_pos, left_quat = self.fk_scene.get_link_poses(LEFT_ARM_LINK_NAMES)
        right_pos, right_quat = self.fk_scene.get_link_poses(RIGHT_ARM_LINK_NAMES)

        left_ee_pos, left_ee_quat = self.fk_scene.get_joint_global_pose(
            LEFT_EE_JOINT_NAME
        )
        right_ee_pos, right_ee_quat = self.fk_scene.get_joint_global_pose(
            RIGHT_EE_JOINT_NAME
        )
        left_tcp_pos, left_tcp_quat = compute_tcp_pose(left_ee_pos, left_ee_quat)
        right_tcp_pos, right_tcp_quat = compute_tcp_pose(right_ee_pos, right_ee_quat)

        keypoints = np.zeros((K, 3), dtype=np.float32)
        keypoints[:6] = left_pos
        keypoints[6] = left_tcp_pos
        keypoints[7:13] = right_pos
        keypoints[13] = right_tcp_pos

        quats = np.zeros((K, 4), dtype=np.float32)
        quats[:6] = left_quat.astype(np.float32)
        quats[6] = np.asarray(left_tcp_quat, dtype=np.float32)
        quats[7:13] = right_quat.astype(np.float32)
        quats[13] = np.asarray(right_tcp_quat, dtype=np.float32)
        self._current_ep_quats.append(quats)

        return keypoints

    def extract_episode(self, episode_idx: int) -> np.ndarray:
        self._current_ep_quats = []
        keypoints = super().extract_episode(episode_idx)
        self._quat_cache[episode_idx] = np.stack(self._current_ep_quats, axis=0)
        self._current_ep_quats = []
        return keypoints

    def _save_episode_keypoints(
        self, episode_idx: int, keypoints_flat: np.ndarray
    ) -> None:
        super()._save_episode_keypoints(episode_idx, keypoints_flat)
        if episode_idx in self._quat_cache:
            ep_dir = self.output_dir / f"episode_{episode_idx:06d}"
            quats = self._quat_cache[episode_idx]
            np.save(
                ep_dir / QUAT_FILENAME,
                quats.reshape(quats.shape[0], K * QUAT_DIM).astype(np.float32),
            )

    def _save_meta(
        self, offset, global_min, global_max, final_min, final_max, total_episodes
    ) -> None:
        super()._save_meta(
            offset, global_min, global_max, final_min, final_max, total_episodes
        )
        meta_path = self.output_dir / "keypoints_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta.update(
            {
                "has_quaternions": True,
                "quat_dim": QUAT_DIM,
                "quat_convention": QUAT_CONVENTION,
                "quat_filename": QUAT_FILENAME,
                "quat_coord_frame": "world",
            }
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
