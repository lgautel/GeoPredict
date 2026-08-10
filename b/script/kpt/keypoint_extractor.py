"""Core keypoint extraction from RoboTwin LeRobot dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    DATASET_DIR,
    K,
    KEYPOINT_NAMES,
    LEFT_ARM_LINK_NAMES,
    LEFT_EE_JOINT_NAME,
    OUTPUT_DIR,
    RIGHT_ARM_LINK_NAMES,
    RIGHT_EE_JOINT_NAME,
    ROBOT_ROOT_POS,
    ROBOT_ROOT_QUAT,
    URDF_PATH,
)
from .coord_transform import apply_offset, compute_auto_offset, validate_range
from .eef_calculator import compute_tcp_position
from .joint_mapper import JointMapper
from .sapien_env import AlohaFKScene


class KeypointExtractor:
    """Extract 3D keypoint trajectories with SAPIEN FK."""

    def __init__(
        self,
        urdf_path: str | Path = URDF_PATH,
        dataset_dir: str | Path = DATASET_DIR,
        output_dir: str | Path = OUTPUT_DIR,
        root_pos: np.ndarray = ROBOT_ROOT_POS,
        root_quat: np.ndarray = ROBOT_ROOT_QUAT,
        offset: Optional[np.ndarray] = None,
    ):
        self.urdf_path = Path(urdf_path)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.manual_offset = None if offset is None else np.asarray(offset, dtype=np.float32)
        self._world_cache: Dict[int, np.ndarray] = {}

        self.fk_scene = AlohaFKScene(self.urdf_path, root_pos, root_quat)
        self.joint_mapper = JointMapper(self.fk_scene)

    def close(self) -> None:
        self.fk_scene.close()

    def _read_parquet_states(self, episode_idx: int) -> np.ndarray:
        parquet_path = (
            self.dataset_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        states = np.array(df["observation.state"].tolist(), dtype=np.float32)
        return states

    def _compute_step_keypoints(self, state_14: np.ndarray) -> np.ndarray:
        qpos = self.joint_mapper.map_state_to_qpos(state_14)
        self.fk_scene.set_qpos(qpos)

        left_links = self.fk_scene.get_link_positions(LEFT_ARM_LINK_NAMES)
        right_links = self.fk_scene.get_link_positions(RIGHT_ARM_LINK_NAMES)

        left_ee_pos, left_ee_quat = self.fk_scene.get_joint_global_pose(LEFT_EE_JOINT_NAME)
        right_ee_pos, right_ee_quat = self.fk_scene.get_joint_global_pose(RIGHT_EE_JOINT_NAME)
        left_tcp = compute_tcp_position(left_ee_pos, left_ee_quat)
        right_tcp = compute_tcp_position(right_ee_pos, right_ee_quat)

        keypoints = np.zeros((K, 3), dtype=np.float32)
        keypoints[:6] = left_links
        keypoints[6] = left_tcp
        keypoints[7:13] = right_links
        keypoints[13] = right_tcp
        return keypoints

    def extract_episode(self, episode_idx: int) -> np.ndarray:
        states = self._read_parquet_states(episode_idx)
        keypoints = np.stack(
            [self._compute_step_keypoints(states[t]) for t in range(states.shape[0])],
            axis=0,
        )
        self._world_cache[episode_idx] = keypoints
        return keypoints

    def _save_episode_keypoints(self, episode_idx: int, keypoints_flat: np.ndarray) -> None:
        ep_dir = self.output_dir / f"episode_{episode_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.save(ep_dir / "keypoints.npy", keypoints_flat.astype(np.float32))

    def _save_meta(
        self,
        offset: np.ndarray,
        global_min: np.ndarray,
        global_max: np.ndarray,
        final_min: np.ndarray,
        final_max: np.ndarray,
        total_episodes: int,
    ) -> None:
        meta = {
            "K": K,
            "keypoint_names": KEYPOINT_NAMES,
            "coord_offset": offset.tolist(),
            "world_range_min": global_min.tolist(),
            "world_range_max": global_max.tolist(),
            "transformed_range_min": final_min.tolist(),
            "transformed_range_max": final_max.tolist(),
            "urdf_path": str(self.urdf_path),
            "dataset_dir": str(self.dataset_dir),
            "total_episodes": total_episodes,
        }
        with open(self.output_dir / "keypoints_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def extract_all(self, episode_indices: Optional[List[int]] = None) -> None:
        info_path = self.dataset_dir / "meta" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        total_episodes = info["total_episodes"]
        if episode_indices is None:
            episode_indices = list(range(total_episodes))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        global_min = np.full(3, np.inf, dtype=np.float32)
        global_max = np.full(3, -np.inf, dtype=np.float32)

        for ep_idx in episode_indices:
            kpts = self.extract_episode(ep_idx)
            global_min = np.minimum(global_min, kpts.min(axis=(0, 1)))
            global_max = np.maximum(global_max, kpts.max(axis=(0, 1)))
            print(f"[INFO] Episode {ep_idx + 1}/{total_episodes} extracted, steps={kpts.shape[0]}")

        if self.manual_offset is not None:
            offset = self.manual_offset
        else:
            offset = compute_auto_offset(global_min, global_max)
        print(f"[INFO] Using offset: {offset}")

        all_transformed = []
        for ep_idx in episode_indices:
            kpts = apply_offset(self._world_cache[ep_idx], offset)
            kpts_flat = kpts.reshape(kpts.shape[0], K * 3)
            self._save_episode_keypoints(ep_idx, kpts_flat)
            all_transformed.append(kpts)

        all_transformed_arr = np.concatenate(all_transformed, axis=0)
        final_min = all_transformed_arr.min(axis=(0, 1))
        final_max = all_transformed_arr.max(axis=(0, 1))
        is_valid, stats = validate_range(all_transformed_arr)
        print(f"[INFO] Transformed range min={final_min}, max={final_max}")
        print(f"[INFO] Range validation: {'PASS' if is_valid else 'FAIL'} ({stats})")

        self._save_meta(
            offset, global_min, global_max, final_min, final_max, total_episodes
        )
        print(f"[INFO] Saved keypoints to {self.output_dir}")
