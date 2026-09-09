#!/usr/bin/env python3
"""Validate LIBERO-plus LeRobot v3.0 dataset with FK keypoints.

8 checks:
  1. info.json schema & version
  2. Parquet column presence & dimensions
  3. Frame count consistency
  4. Episode continuity
  5. Keypoint position range
  6. Quaternion unit norm
  7. Quaternion hemisphere constraint
  8. Video file existence (if videos present)

Usage:
    conda run -n phantom python validate_libero.py
    conda run -n phantom python validate_libero.py --dataset_root /path/to/output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = "/home/luogang/DATA/libero_plus_lrb3"
K = 8
QUAT_DIM = 4
POS_DIM = K * 3       # 24
QUAT_TOTAL_DIM = K * QUAT_DIM  # 32


def check_info_json(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"  FAIL: {info_path} not found")
        return False
    with open(info_path) as f:
        info = json.load(f)
    checks = [
        ("codebase_version == v3.0", info.get("codebase_version") == "v3.0"),
        ("robot_type == franka", info.get("robot_type") == "franka"),
        ("fps == 20", info.get("fps") == 20),
        ("observation.keypoint_3d in features", "observation.keypoint_3d" in info.get("features", {})),
        ("observation.keypoint_quat in features", "observation.keypoint_quat" in info.get("features", {})),
        ("no reward feature", "reward" not in info.get("features", {})),
        ("no discount feature", "discount" not in info.get("features", {})),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")
        if not passed:
            ok = False
    return ok


def check_parquet_columns(root: Path) -> bool:
    pq_path = root / "data" / "chunk-000" / "file-000.parquet"
    if not pq_path.exists():
        print(f"  FAIL: {pq_path} not found")
        return False
    df = pd.read_parquet(pq_path)
    required = [
        "observation.state", "observation.joint_state", "action",
        "observation.keypoint_3d", "observation.keypoint_quat",
        "language_instruction", "timestamp", "frame_index",
        "episode_index", "index", "task_index",
    ]
    ok = True
    for col in required:
        present = col in df.columns
        print(f"  [{'OK' if present else 'FAIL'}] Column '{col}'")
        if not present:
            ok = False

    # Check no reward/discount
    for bad_col in ("reward", "discount"):
        if bad_col in df.columns:
            print(f"  [FAIL] Column '{bad_col}' should NOT be present")
            ok = False
        else:
            print(f"  [OK] '{bad_col}' not present")

    # Check dimensions
    sample = df.iloc[0]
    for col, expected_dim in [
        ("observation.state", 8),
        ("observation.joint_state", 7),
        ("action", 7),
        ("observation.keypoint_3d", POS_DIM),
        ("observation.keypoint_quat", QUAT_TOTAL_DIM),
    ]:
        if col in df.columns:
            actual = len(sample[col])
            match = actual == expected_dim
            print(f"  [{'OK' if match else 'FAIL'}] {col} dim={actual} (expected {expected_dim})")
            if not match:
                ok = False
    return ok


def check_frame_counts(root: Path) -> bool:
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    expected = info["total_frames"]
    actual = 0
    for pq_path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["index"])
        actual += len(df)
    ok = actual == expected
    print(f"  [{'OK' if ok else 'FAIL'}] Frame count: {actual} (expected {expected})")
    return ok


def check_episode_continuity(root: Path) -> bool:
    ok = True
    errors = 0
    checked_eps = 0
    for pq_path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["episode_index", "frame_index"])
        for ep_idx, group in df.groupby("episode_index"):
            frames = sorted(group["frame_index"].tolist())
            expected = list(range(len(frames)))
            if frames != expected:
                if errors < 3:
                    print(f"  [FAIL] Episode {ep_idx}: frame_index not contiguous (first few: {frames[:5]})")
                errors += 1
                ok = False
            checked_eps += 1
    if ok:
        print(f"  [OK] {checked_eps} episodes: all frame indices contiguous")
    elif errors >= 3:
        print(f"  ... {errors} total errors")
    return ok


def check_keypoint_positions(root: Path) -> bool:
    all_pos = []
    for pq_path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["observation.keypoint_3d"])
        pos = np.stack(df["observation.keypoint_3d"].values).astype(np.float32)
        all_pos.append(pos)
    all_pos = np.concatenate(all_pos, axis=0)

    kpt_meta_path = root / "meta" / "keypoints_meta.json"
    norm_method = "unknown"
    if kpt_meta_path.exists():
        with open(kpt_meta_path) as f:
            kpt_meta = json.load(f)
        norm_method = kpt_meta.get("normalization_method", "unknown")

    pos_min = float(all_pos.min())
    pos_max = float(all_pos.max())
    pos_abs_max = float(np.abs(all_pos).max())

    if norm_method == "rpad":
        ok = pos_abs_max <= 1.01
        print(f"  [{'OK' if ok else 'FAIL'}] Position range [{pos_min:.4f}, {pos_max:.4f}], "
              f"|max|={pos_abs_max:.4f} (R_pad, expect ≤1.0)")
    else:
        ok = True  # auto-offset: just report
        print(f"  [INFO] Position range [{pos_min:.4f}, {pos_max:.4f}] (auto-offset norm={norm_method})")

    return ok


def check_quaternion_norms(root: Path) -> bool:
    all_quat = []
    for pq_path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["observation.keypoint_quat"])
        quat = np.stack(df["observation.keypoint_quat"].values).astype(np.float32)
        all_quat.append(quat)
    all_quat = np.concatenate(all_quat, axis=0).reshape(-1, QUAT_DIM)
    norms = np.linalg.norm(all_quat, axis=1)
    err = np.abs(norms - 1.0)
    max_err = float(err.max())
    ok = max_err < 0.01
    print(f"  [{'OK' if ok else 'FAIL'}] Quat norm error: max={max_err:.2e} (threshold 0.01)")
    return ok


def check_quaternion_hemisphere(root: Path) -> bool:
    violations = 0
    total = 0
    for pq_path in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq_path, columns=["observation.keypoint_quat"])
        quat = np.stack(df["observation.keypoint_quat"].values).reshape(-1, K, QUAT_DIM)
        qw = quat[:, :, 0]  # wxyz → qw is first
        violations += int((qw < 0).sum())
        total += qw.size
    ok = violations == 0
    print(f"  [{'OK' if ok else 'FAIL'}] Hemisphere: {violations}/{total} violations (qw<0)")
    return ok


def check_videos(root: Path) -> bool:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    num_chunks = info["total_chunks"]
    has_video_features = any(
        v.get("dtype") == "video" for v in info.get("features", {}).values()
    )
    if not has_video_features:
        print("  [INFO] No video features in info.json (--no_video mode)")
        return True

    ok = True
    missing = 0
    for cam in ["observation.images.image", "observation.images.wrist_image"]:
        for chunk_idx in range(num_chunks):
            vp = root / "videos" / cam / f"chunk-{chunk_idx:03d}" / "file-000.mp4"
            if not vp.exists():
                missing += 1
                if missing <= 3:
                    print(f"  [FAIL] Missing: {vp}")
                ok = False
            elif vp.stat().st_size < 1000:
                print(f"  [WARN] Very small video: {vp} ({vp.stat().st_size} bytes)")

    if ok:
        print(f"  [OK] All {num_chunks * 2} video files present")
    elif missing > 3:
        print(f"  ... {missing} total missing video files")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = Path(args.dataset_root)

    if not root.exists():
        print(f"ERROR: Dataset root not found: {root}")
        sys.exit(1)

    checks = [
        ("1. info.json schema", check_info_json),
        ("2. Parquet columns & dims", check_parquet_columns),
        ("3. Frame count consistency", check_frame_counts),
        ("4. Episode continuity", check_episode_continuity),
        ("5. Keypoint positions", check_keypoint_positions),
        ("6. Quaternion unit norm", check_quaternion_norms),
        ("7. Quaternion hemisphere", check_quaternion_hemisphere),
        ("8. Video files", check_videos),
    ]

    print(f"\n{'='*60}")
    print(f"  Validating: {root}")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    for name, check_fn in checks:
        print(f"\n[Check {name}]")
        try:
            if check_fn(root):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Result: {passed} PASSED, {failed} FAILED (of {len(checks)} checks)")
    print(f"{'='*60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
