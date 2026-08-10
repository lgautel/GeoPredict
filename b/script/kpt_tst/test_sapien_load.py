"""Verify SAPIEN can load ALOHA URDF."""

import pytest

from b.script.kpt.config import (
    LEFT_ARM_JOINT_NAMES,
    LEFT_ARM_LINK_NAMES,
    LEFT_EE_JOINT_NAME,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_LINK_NAMES,
    RIGHT_EE_JOINT_NAME,
    ROBOT_ROOT_POS,
    ROBOT_ROOT_QUAT,
    URDF_PATH,
)
from b.script.kpt.sapien_env import AlohaFKScene


@pytest.fixture(scope="module")
def fk_scene():
    scene = AlohaFKScene(URDF_PATH, ROBOT_ROOT_POS, ROBOT_ROOT_QUAT)
    yield scene
    scene.close()


class TestSapienLoad:
    def test_urdf_loads(self, fk_scene):
        assert fk_scene.get_num_active_joints() == 38

    def test_left_arm_joints_exist(self, fk_scene):
        names = set(fk_scene.get_active_joint_names())
        for name in LEFT_ARM_JOINT_NAMES:
            assert name in names

    def test_right_arm_joints_exist(self, fk_scene):
        names = set(fk_scene.get_active_joint_names())
        for name in RIGHT_ARM_JOINT_NAMES:
            assert name in names

    def test_left_arm_links_exist(self, fk_scene):
        names = set(fk_scene.get_link_names())
        for name in LEFT_ARM_LINK_NAMES:
            assert name in names

    def test_right_arm_links_exist(self, fk_scene):
        names = set(fk_scene.get_link_names())
        for name in RIGHT_ARM_LINK_NAMES:
            assert name in names

    def test_ee_joints_exist(self, fk_scene):
        names = set(fk_scene.get_joint_name_to_idx().keys())
        assert LEFT_EE_JOINT_NAME in names
        assert RIGHT_EE_JOINT_NAME in names

    def test_root_link_is_footprint(self, fk_scene):
        assert "footprint" in fk_scene.get_link_names()
