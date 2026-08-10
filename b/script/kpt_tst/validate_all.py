#!/usr/bin/env python3
"""Acceptance validation for all extracted keypoint episodes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b.script.kpt.config import K, OUTPUT_DIR, VOXEL_RANGE_MAX, VOXEL_RANGE_MIN
from b.script.kpt.coord_transform import validate_range


def validate_all(output_dir: Path = OUTPUT_DIR) -> bool:
    output_dir = Path(output_dir)
    meta_path = output_dir / "keypoints_meta.json"
    if not meta_path.exists():
        print(f"[FAIL] Missing meta file: {meta_path}")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    total_episodes = meta["total_episodes"]

    all_kpts = []
    passed = True
    for ep_idx in range(total_episodes):
        npy_path = output_dir / f"episode_{ep_idx:06d}" / "keypoints.npy"
        if not npy_path.exists():
            print(f"[FAIL] Missing {npy_path}")
            passed = False
            continue

        kpts_flat = np.load(npy_path)
        if kpts_flat.dtype != np.float32 or kpts_flat.shape[1] != K * 3:
            print(f"[FAIL] Bad format at episode {ep_idx}: {kpts_flat.shape}, {kpts_flat.dtype}")
            passed = False
            continue

        kpts = kpts_flat.reshape(-1, K, 3)
        is_valid, stats = validate_range(kpts)
        if not is_valid:
            print(f"[FAIL] Episode {ep_idx} out of range: {stats}")
            passed = False

        if kpts.shape[0] > 1:
            diffs = np.linalg.norm(np.diff(kpts, axis=0), axis=-1)
            if diffs.max() >= 0.05:
                print(f"[FAIL] Episode {ep_idx} discontinuity: max step {diffs.max():.4f}m")
                passed = False

        all_kpts.append(kpts)

    if not all_kpts:
        return False

    all_arr = np.concatenate(all_kpts, axis=0)
    global_min = all_arr.min(axis=(0, 1))
    global_max = all_arr.max(axis=(0, 1))
    print(f"[INFO] Verified {len(all_kpts)}/{total_episodes} episodes")
    print(f"[INFO] Global range: min={global_min}, max={global_max}")

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    _save_visualizations(all_kpts, vis_dir)
    print(f"[INFO] Visualization saved to {vis_dir}")

    if passed:
        print("[PASS] All acceptance checks passed")
    else:
        print("[FAIL] Some acceptance checks failed")
    return passed


def _save_visualizations(all_kpts, vis_dir: Path) -> None:
    rng = np.random.default_rng(0)
    sample_eps = rng.choice(len(all_kpts), size=min(5, len(all_kpts)), replace=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for ep_idx in sample_eps:
        kpts = all_kpts[ep_idx]
        for t in np.linspace(0, kpts.shape[0] - 1, 10, dtype=int):
            left = kpts[t, :7]
            right = kpts[t, 7:]
            ax.plot(left[:, 0], left[:, 1], left[:, 2], "b.", markersize=3)
            ax.plot(right[:, 0], right[:, 1], right[:, 2], "r.", markersize=3)
    ax.set_xlim(VOXEL_RANGE_MIN[0], VOXEL_RANGE_MAX[0])
    ax.set_ylim(VOXEL_RANGE_MIN[1], VOXEL_RANGE_MAX[1])
    ax.set_zlim(VOXEL_RANGE_MIN[2], VOXEL_RANGE_MAX[2])
    ax.set_title("Sampled 3D Keypoints")
    fig.savefig(vis_dir / "keypoints_3d_samples.png", dpi=120)
    plt.close(fig)

    ep0 = all_kpts[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    for i in range(3):
        axes[i].plot(ep0[:, 6, i], label="left_tcp")
        axes[i].plot(ep0[:, 13, i], label="right_tcp")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    axes[-1].set_xlabel("frame")
    fig.suptitle("Episode 0 EEF TCP trajectories")
    fig.tight_layout()
    fig.savefig(vis_dir / "eef_trajectories_ep0.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ok = validate_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
