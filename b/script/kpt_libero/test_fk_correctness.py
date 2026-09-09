#!/usr/bin/env python3
"""Unit tests for FK correctness.

Usage:
    MUJOCO_GL=egl conda run -n phantom python test_fk_correctness.py

Tests:
  1. FK at home position (base offset check)
  2. FK matches robosuite internal state
  3. RLDS parse smoke test
  4. FK + RLDS end-to-end
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

XML_PATH = "/tmp/panda_robosuite_full.xml"
RLDS_ROOT = "/home/luogang/DATA/libero_plus_rlds/libero_mix/1.0.0"


def test_fk_home():
    """Test FK at home position: verify robot base offset."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    base_id = model.body("robot0_base").id
    base_pos = data.xpos[base_id]
    print(f"  robot0_base pos: {base_pos}")
    assert abs(base_pos[0] - (-0.56)) < 0.02, f"Base X wrong: {base_pos[0]}"
    assert abs(base_pos[2] - 0.912) < 0.02, f"Base Z wrong: {base_pos[2]}"

    eef_id = model.body("gripper0_eef").id
    eef_quat = data.xquat[eef_id]
    assert abs(np.linalg.norm(eef_quat) - 1.0) < 1e-5, f"EEF quat not unit: {eef_quat}"
    print(f"  gripper0_eef pos: {data.xpos[eef_id]}")
    print("  PASS: FK at home position")


def test_fk_vs_robosuite():
    """Test FK matches robosuite internal state (same MJCF)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite
    import mujoco

    env = robosuite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    env.reset()
    rs_eef_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_eef")].copy()
    rs_qpos = env.sim.data.qpos[:9].copy()
    env.close()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    data.qpos[:9] = rs_qpos
    mujoco.mj_forward(model, data)

    fk_eef_id = model.body("gripper0_eef").id
    fk_eef_pos = data.xpos[fk_eef_id]

    err = np.linalg.norm(rs_eef_pos - fk_eef_pos)
    print(f"  robosuite EEF: {rs_eef_pos}")
    print(f"  standalone FK: {fk_eef_pos}")
    print(f"  Position error: {err:.3e} m")
    assert err < 1e-5, f"FK mismatch: {err} m"
    print("  PASS: FK matches robosuite")


def test_rlds_parse():
    """Smoke test RLDS parsing."""
    from kpt_libero.rlds_reader import iter_episodes

    ep_iter = iter_episodes(RLDS_ROOT)
    ep = next(ep_iter)

    assert ep["qpos"].shape == (ep["n_steps"], 9), f"qpos shape: {ep['qpos'].shape}"
    assert ep["state"].shape == (ep["n_steps"], 8)
    assert ep["joint_state"].shape == (ep["n_steps"], 7)
    assert ep["action"].shape == (ep["n_steps"], 7)
    assert len(ep["image"]) == ep["n_steps"]
    assert len(ep["wrist_image"]) == ep["n_steps"]
    assert isinstance(ep["language"], str) and len(ep["language"]) > 0

    print(f"  Episode: steps={ep['n_steps']}, lang='{ep['language'][:60]}...'")
    print(f"  qpos range: [{ep['qpos'].min():.4f}, {ep['qpos'].max():.4f}]")
    print("  PASS: RLDS parse")


def test_fk_on_rlds():
    """End-to-end FK on RLDS episode."""
    from kpt_libero.mujoco_fk import MujocoFKScene
    from kpt_libero.rlds_reader import iter_episodes
    from kpt_libero.config_libero import K

    fk = MujocoFKScene(XML_PATH)
    ep = next(iter_episodes(RLDS_ROOT))
    positions, quaternions = fk.extract_episode(ep["qpos"])

    assert positions.shape == (ep["n_steps"], K, 3), f"pos shape: {positions.shape}"
    assert quaternions.shape == (ep["n_steps"], K, 4), f"quat shape: {quaternions.shape}"

    # Hemisphere: qw is index 0 in wxyz
    qw = quaternions[:, :, 0]
    assert (qw >= -1e-6).all(), f"Hemisphere violation: min qw={qw.min()}"

    norms = np.linalg.norm(quaternions.reshape(-1, 4), axis=1)
    assert np.abs(norms - 1.0).max() < 0.001, f"Norm err: {np.abs(norms-1).max()}"

    assert np.abs(positions).max() < 3.0, f"Pos OOB: {np.abs(positions).max()}"

    print(f"  Pos: shape={positions.shape}, range=[{positions.min():.4f}, {positions.max():.4f}]")
    print(f"  Quat: qw range=[{qw.min():.4f}, {qw.max():.4f}], norm err max={np.abs(norms-1).max():.2e}")
    fk.close()
    print("  PASS: FK on RLDS")


if __name__ == "__main__":
    tests = [
        ("1. FK at home position", test_fk_home),
        ("2. FK vs robosuite", test_fk_vs_robosuite),
        ("3. RLDS parse", test_rlds_parse),
        ("4. FK on RLDS", test_fk_on_rlds),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[Test {name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
