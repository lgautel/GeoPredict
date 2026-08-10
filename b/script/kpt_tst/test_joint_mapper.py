"""Verify 14-dim state to 38-dim qpos mapping."""

import numpy as np
import pytest

from b.script.kpt.config import GRIPPER_SCALE, ROBOT_ROOT_POS, ROBOT_ROOT_QUAT, URDF_PATH
from b.script.kpt.joint_mapper import JointMapper
from b.script.kpt.sapien_env import AlohaFKScene


@pytest.fixture(scope="module")
def joint_mapper():
    scene = AlohaFKScene(URDF_PATH, ROBOT_ROOT_POS, ROBOT_ROOT_QUAT)
    mapper = JointMapper(scene)
    yield mapper
    scene.close()


class TestJointMapper:
    def test_zero_state_zero_qpos(self, joint_mapper):
        qpos = joint_mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        assert qpos.shape[0] == 38
        assert np.allclose(qpos[joint_mapper.left_arm_indices], 0.0)
        assert np.allclose(qpos[joint_mapper.right_arm_indices], 0.0)

    def test_left_arm_joint1_only(self, joint_mapper):
        state = np.zeros(14, dtype=np.float32)
        state[0] = 1.0
        qpos = joint_mapper.map_state_to_qpos(state)
        assert qpos[joint_mapper.left_arm_indices[0]] == pytest.approx(1.0)
        assert qpos[joint_mapper.right_arm_indices].sum() == pytest.approx(0.0)

    def test_right_arm_joint3_only(self, joint_mapper):
        state = np.zeros(14, dtype=np.float32)
        state[9] = -0.5
        qpos = joint_mapper.map_state_to_qpos(state)
        assert qpos[joint_mapper.right_arm_indices[2]] == pytest.approx(-0.5)

    def test_left_gripper_closed(self, joint_mapper):
        state = np.zeros(14, dtype=np.float32)
        qpos = joint_mapper.map_state_to_qpos(state)
        for idx in joint_mapper.left_gripper_indices:
            assert qpos[idx] == pytest.approx(GRIPPER_SCALE[0])

    def test_left_gripper_open(self, joint_mapper):
        state = np.zeros(14, dtype=np.float32)
        state[6] = 1.0
        qpos = joint_mapper.map_state_to_qpos(state)
        for idx in joint_mapper.left_gripper_indices:
            assert qpos[idx] == pytest.approx(GRIPPER_SCALE[1])

    def test_batch_mapping_shape(self, joint_mapper):
        states = np.zeros((5, 14), dtype=np.float32)
        qpos_batch = joint_mapper.map_batch(states)
        assert qpos_batch.shape == (5, 38)

    def test_unmapped_joints_remain_zero(self, joint_mapper):
        state = np.zeros(14, dtype=np.float32)
        state[0] = 0.3
        state[7] = -0.2
        qpos = joint_mapper.map_state_to_qpos(state)
        mapped = set(
            joint_mapper.left_arm_indices
            + joint_mapper.right_arm_indices
            + joint_mapper.left_gripper_indices
            + joint_mapper.right_gripper_indices
        )
        for idx in range(len(qpos)):
            if idx not in mapped:
                assert qpos[idx] == pytest.approx(0.0)
