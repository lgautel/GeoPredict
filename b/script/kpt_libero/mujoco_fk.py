"""MuJoCo FK wrapper for Franka Panda (from robosuite MJCF)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from .config_libero import KEYPOINT_BODY_NAMES, PANDA_XML_PATH, ROBOT_NQPOS


class MujocoFKScene:
    """Load Franka Panda in a minimal MuJoCo scene for FK queries.

    Adapted from 3dkptraj_lbrpls_1.md §6.3 MujocoFKScene design,
    with body_id caching for performance.
    """

    def __init__(self, model_xml: str | Path = PANDA_XML_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self._body_id_cache: Dict[str, int] = {}
        for name in KEYPOINT_BODY_NAMES:
            self._body_id_cache[name] = self.model.body(name).id

    def set_qpos_and_forward(self, qpos: np.ndarray) -> None:
        """Set robot qpos (first ROBOT_NQPOS entries) and run FK."""
        n = min(len(qpos), ROBOT_NQPOS)
        self.data.qpos[:n] = qpos[:n]
        mujoco.mj_forward(self.model, self.data)

    def get_body_poses(
        self, body_names: List[str] | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get world-frame positions [K,3] and quaternions [K,4] (wxyz).

        Must call set_qpos_and_forward() first.
        """
        if body_names is None:
            body_names = KEYPOINT_BODY_NAMES
        n = len(body_names)
        positions = np.zeros((n, 3), dtype=np.float32)
        quaternions = np.zeros((n, 4), dtype=np.float32)
        for i, name in enumerate(body_names):
            bid = self._body_id_cache.get(name)
            if bid is None:
                bid = self.model.body(name).id
                self._body_id_cache[name] = bid
            positions[i] = self.data.xpos[bid]
            quaternions[i] = self.data.xquat[bid]  # wxyz
        return positions, quaternions

    def extract_episode(
        self, qpos_seq: np.ndarray, body_names: List[str] | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract keypoints for an entire episode.

        Args:
            qpos_seq: [T, 9] joint positions (arm + gripper)
            body_names: override keypoint bodies

        Returns:
            positions: [T, K, 3] float32
            quaternions: [T, K, 4] float32 (wxyz, hemisphere-normalized)
        """
        if body_names is None:
            body_names = KEYPOINT_BODY_NAMES
        T = qpos_seq.shape[0]
        K = len(body_names)
        all_pos = np.zeros((T, K, 3), dtype=np.float32)
        all_quat = np.zeros((T, K, 4), dtype=np.float32)

        for t in range(T):
            self.set_qpos_and_forward(qpos_seq[t])
            pos, quat = self.get_body_poses(body_names)
            # Hemisphere normalization: ensure qw >= 0 (wxyz → qw is index 0)
            neg_mask = quat[:, 0] < 0
            quat[neg_mask] = -quat[neg_mask]
            all_pos[t] = pos
            all_quat[t] = quat

        return all_pos, all_quat

    def close(self):
        self.model = None
        self.data = None
