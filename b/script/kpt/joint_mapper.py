"""Map dataset 14-dim state to SAPIEN 38-dim qpos."""

from __future__ import annotations

import numpy as np

from .config import (
    GRIPPER_SCALE,
    LEFT_ARM_JOINT_NAMES,
    LEFT_ARM_STATE_SLICE,
    LEFT_GRIPPER_JOINT_NAMES,
    LEFT_GRIPPER_STATE_IDX,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_STATE_SLICE,
    RIGHT_GRIPPER_JOINT_NAMES,
    RIGHT_GRIPPER_STATE_IDX,
)
from .sapien_env import AlohaFKScene


class JointMapper:
    """Map LeRobot observation.state (14) to SAPIEN active-joint qpos (38)."""

    def __init__(self, fk_scene: AlohaFKScene):
        self.num_active_joints = fk_scene.get_num_active_joints()
        joint_name_to_idx = fk_scene.get_joint_name_to_idx()

        self.left_arm_indices = [
            joint_name_to_idx[name] for name in LEFT_ARM_JOINT_NAMES
        ]
        self.right_arm_indices = [
            joint_name_to_idx[name] for name in RIGHT_ARM_JOINT_NAMES
        ]
        self.left_gripper_indices = [
            joint_name_to_idx[name] for name in LEFT_GRIPPER_JOINT_NAMES
        ]
        self.right_gripper_indices = [
            joint_name_to_idx[name] for name in RIGHT_GRIPPER_JOINT_NAMES
        ]

    def _denormalize_gripper(self, normalized_val: float) -> float:
        return normalized_val * (GRIPPER_SCALE[1] - GRIPPER_SCALE[0]) + GRIPPER_SCALE[0]

    def map_state_to_qpos(self, state_14: np.ndarray) -> np.ndarray:
        state_14 = np.asarray(state_14, dtype=np.float64)
        if state_14.shape[0] != 14:
            raise ValueError(f"Expected state shape [14], got {state_14.shape}")

        qpos = np.zeros(self.num_active_joints, dtype=np.float64)

        for i, idx in enumerate(self.left_arm_indices):
            qpos[idx] = state_14[LEFT_ARM_STATE_SLICE][i]
        for i, idx in enumerate(self.right_arm_indices):
            qpos[idx] = state_14[RIGHT_ARM_STATE_SLICE][i]

        left_gripper_q = self._denormalize_gripper(state_14[LEFT_GRIPPER_STATE_IDX])
        right_gripper_q = self._denormalize_gripper(state_14[RIGHT_GRIPPER_STATE_IDX])
        for idx in self.left_gripper_indices:
            qpos[idx] = left_gripper_q
        for idx in self.right_gripper_indices:
            qpos[idx] = right_gripper_q

        return qpos

    def map_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        return np.stack([self.map_state_to_qpos(state) for state in states], axis=0)
