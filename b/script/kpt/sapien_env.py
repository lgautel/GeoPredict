"""Minimal SAPIEN scene for ALOHA FK queries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sapien.core as sapien


class AlohaFKScene:
    """Load ALOHA URDF in a minimal SAPIEN scene and expose FK helpers."""

    def __init__(
        self,
        urdf_path: str | Path,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
    ):
        self.urdf_path = Path(urdf_path)
        self._urdf_dir = self.urdf_path.parent
        self._prev_cwd = os.getcwd()

        os.chdir(self._urdf_dir)
        try:
            self.engine = sapien.Engine()
            self.scene = self.engine.create_scene()
            self.scene.set_timestep(1.0 / 240)

            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            self.robot = loader.load(str(self.urdf_path))
            self.robot.set_root_pose(
                sapien.Pose(root_pos.tolist(), root_quat.tolist())
            )

            self._active_joints = self.robot.get_active_joints()
            self._joint_name_to_idx: Dict[str, int] = {
                joint.get_name(): idx for idx, joint in enumerate(self._active_joints)
            }
            self._link_cache: Dict[str, sapien.PhysxArticulationLinkComponent] = {}
            self._joint_cache: Dict[str, sapien.PhysxArticulationJointComponent] = {}
            for link in self.robot.get_links():
                self._link_cache[link.get_name()] = link
            for joint in self.robot.get_joints():
                self._joint_cache[joint.get_name()] = joint

            self.scene.step()
        finally:
            os.chdir(self._prev_cwd)

    def get_active_joint_names(self) -> List[str]:
        return [joint.get_name() for joint in self._active_joints]

    def get_link_names(self) -> List[str]:
        return list(self._link_cache.keys())

    def get_num_active_joints(self) -> int:
        return len(self._active_joints)

    def get_joint_name_to_idx(self) -> Dict[str, int]:
        return dict(self._joint_name_to_idx)

    def set_qpos(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape[0] != len(self._active_joints):
            raise ValueError(
                f"Expected qpos shape [{len(self._active_joints)}], got {qpos.shape[0]}"
            )
        self.robot.set_qpos(qpos)
        self.scene.step()

    def get_link_positions(self, link_names: List[str]) -> np.ndarray:
        positions = np.zeros((len(link_names), 3), dtype=np.float32)
        for i, name in enumerate(link_names):
            link = self._link_cache.get(name)
            if link is None:
                link = self.robot.find_link_by_name(name)
                if link is None:
                    raise ValueError(f"Link '{name}' not found")
                self._link_cache[name] = link
            positions[i] = link.get_entity_pose().p
        return positions

    def get_link_poses(
        self, link_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get world-frame positions AND quaternions for the specified links."""
        n = len(link_names)
        positions = np.zeros((n, 3), dtype=np.float32)
        quaternions = np.zeros((n, 4), dtype=np.float64)
        for i, name in enumerate(link_names):
            link = self._link_cache.get(name)
            if link is None:
                link = self.robot.find_link_by_name(name)
                if link is None:
                    raise ValueError(f"Link '{name}' not found")
                self._link_cache[name] = link
            pose = link.get_entity_pose()
            positions[i] = pose.p
            quaternions[i] = pose.q
        return positions, quaternions

    def get_joint_global_pose(self, joint_name: str) -> Tuple[np.ndarray, np.ndarray]:
        joint = self._joint_cache.get(joint_name)
        if joint is None:
            joint = self.robot.find_joint_by_name(joint_name)
            if joint is None:
                raise ValueError(f"Joint '{joint_name}' not found")
            self._joint_cache[joint_name] = joint
        pose = joint.global_pose
        return np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)

    def close(self) -> None:
        self.scene = None
        self.engine = None
        self.robot = None
