"""Verify home-position FK sanity."""

import numpy as np
import pytest

from b.script.kpt.config import (
    LEFT_ARM_LINK_NAMES,
    RIGHT_ARM_LINK_NAMES,
    ROBOT_ROOT_POS,
    ROBOT_ROOT_QUAT,
    URDF_PATH,
)
from b.script.kpt.joint_mapper import JointMapper
from b.script.kpt.sapien_env import AlohaFKScene


@pytest.fixture(scope="module")
def fk_setup():
    scene = AlohaFKScene(URDF_PATH, ROBOT_ROOT_POS, ROBOT_ROOT_QUAT)
    mapper = JointMapper(scene)
    yield scene, mapper
    scene.close()


class TestFKHome:
    def test_fl_link1_above_table(self, fk_setup):
        scene, mapper = fk_setup
        qpos = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos)
        pos = scene.get_link_positions(["fl_link1"])[0]
        assert pos[2] > 0.7

    def test_fl_link6_reasonable_height(self, fk_setup):
        scene, mapper = fk_setup
        qpos = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos)
        pos = scene.get_link_positions(["fl_link6"])[0]
        assert pos[2] > 0.6

    def test_left_arm_y_negative(self, fk_setup):
        scene, mapper = fk_setup
        qpos = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos)
        left = scene.get_link_positions(LEFT_ARM_LINK_NAMES)
        assert left[:, 1].mean() < 0.0

    def test_arms_y_similar(self, fk_setup):
        scene, mapper = fk_setup
        qpos = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos)
        left = scene.get_link_positions(LEFT_ARM_LINK_NAMES)
        right = scene.get_link_positions(RIGHT_ARM_LINK_NAMES)
        # Both arms sit on the same side of the world Y axis after root rotation.
        assert abs(left[:, 1].mean() - right[:, 1].mean()) < 0.02

    def test_positions_in_physical_range(self, fk_setup):
        scene, mapper = fk_setup
        qpos = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos)
        pos = scene.get_link_positions(LEFT_ARM_LINK_NAMES + RIGHT_ARM_LINK_NAMES)
        assert np.all(np.abs(pos[:, :2]) < 2.0)
        assert np.all((pos[:, 2] > 0.0) & (pos[:, 2] < 2.0))

    def test_joint_change_moves_link(self, fk_setup):
        scene, mapper = fk_setup
        qpos0 = mapper.map_state_to_qpos(np.zeros(14, dtype=np.float32))
        scene.set_qpos(qpos0)
        pos0 = scene.get_link_positions(["fl_link6"])[0].copy()

        state = np.zeros(14, dtype=np.float32)
        state[2] = 0.8
        qpos1 = mapper.map_state_to_qpos(state)
        scene.set_qpos(qpos1)
        pos1 = scene.get_link_positions(["fl_link6"])[0]
        assert np.linalg.norm(pos1 - pos0) > 0.01
