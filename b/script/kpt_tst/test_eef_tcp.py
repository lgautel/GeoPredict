"""Verify EEF TCP calculation."""

import numpy as np
import pytest

from b.script.kpt.config import GLOBAL_TRANS_MATRIX
from b.script.kpt.eef_calculator import compute_tcp_position


class TestEEFTcp:
    def test_identity_rotation_offsets_x(self):
        pos = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tcp = compute_tcp_position(pos, quat)
        assert tcp[0] == pytest.approx(0.12, abs=1e-5)
        assert tcp[1] == pytest.approx(0.0, abs=1e-5)
        assert tcp[2] == pytest.approx(0.0, abs=1e-5)

    def test_z_rotation_changes_offset_direction(self):
        pos = np.zeros(3)
        quat = np.array([0.7071068, 0.0, 0.0, 0.7071068])
        tcp = compute_tcp_position(pos, quat)
        assert abs(tcp[0]) < 0.02
        assert abs(tcp[1] - 0.12) < 0.02

    def test_y_rotation_with_global_trans(self):
        pos = np.zeros(3)
        quat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        tcp = compute_tcp_position(pos, quat)
        assert abs(tcp[2] + 0.12) < 0.02

    def test_zero_bias_equals_ee_position(self):
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        tcp = compute_tcp_position(pos, quat, gripper_bias=0.0)
        assert np.allclose(tcp, pos, atol=1e-6)

    def test_float32_precision(self):
        pos = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        tcp = compute_tcp_position(pos, quat)
        assert tcp.dtype == np.float32

    def test_global_trans_matrix_values(self):
        assert GLOBAL_TRANS_MATRIX[1, 1] == -1
        assert GLOBAL_TRANS_MATRIX[2, 2] == -1
