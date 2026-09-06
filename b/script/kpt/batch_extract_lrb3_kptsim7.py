#!/usr/bin/env python3
"""Batch extract keypoints (positions + quaternions) and convert to LeRobot v3 format.

For each RoboTwin task:
  1. SAPIEN FK extracts 14 keypoint positions [K*3=42] + quaternions [K*4=56]
  2. Combines with original v2.1 data into a single LeRobot v3 parquet
  3. Creates v3 meta structure (info.json, episodes parquet, tasks parquet, stats, etc.)
  4. Symlinks original video directories

Output directory: {task}_lrb3_kptsim7/

Usage:
    conda activate RoboTwin
    cd /home/luogang/SRC/Robot/GeoPredict
    python b/script/kpt/batch_extract_lrb3_kptsim7.py [--tasks scan_object hanging_mug ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b.script.kpt.config import (
    K,
    KEYPOINT_NAMES,
    QUAT_CONVENTION,
    QUAT_DIM,
    ROBOT_ROOT_POS,
    ROBOT_ROOT_QUAT,
    VOXEL_RANGE_MAX,
    VOXEL_RANGE_MIN,
)
from b.script.kpt.coord_transform import apply_offset, compute_auto_offset, validate_range
from b.script.kpt.keypoint_pose_extractor import KeypointPoseExtractor

# ===== Defaults =====
DATA_ROOT = Path("/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean")
URDF_PATH = Path(
    "/home/luogang/share/zwy/Projects/RoboTwin"
    "/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
)
OUTPUT_SUFFIX = "_lrb3_kptsim7"

KPT_POS_NAMES = []
for name in KEYPOINT_NAMES:
    for axis in ("x", "y", "z"):
        KPT_POS_NAMES.append(f"{name}_{axis}")

KPT_QUAT_NAMES = []
for name in KEYPOINT_NAMES:
    for comp in ("w", "x", "y", "z"):
        KPT_QUAT_NAMES.append(f"{name}_q{comp}")


def discover_tasks(data_root: Path) -> List[str]:
    """Find all original task directories (no kptsim/lrb/lrbv30/lrb3 suffixes)."""
    skip_suffixes = ("_kptsim", "_lrb", "_lrbv30", "_lrb3_kptsim7")
    tasks = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        if (d / "meta" / "info.json").exists():
            tasks.append(name)
    return tasks


def read_v21_info(task_dir: Path) -> dict:
    with open(task_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def read_v21_tasks(task_dir: Path) -> List[dict]:
    tasks_file = task_dir / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_file.exists():
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    return tasks


def read_v21_episodes(task_dir: Path) -> List[dict]:
    ep_file = task_dir / "meta" / "episodes.jsonl"
    episodes = []
    if ep_file.exists():
        with open(ep_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def create_shared_extractor(urdf_path: Path) -> KeypointPoseExtractor:
    """Create a single extractor whose SAPIEN scene is reused across tasks."""
    return KeypointPoseExtractor(
        urdf_path=urdf_path,
        dataset_dir=Path("/tmp"),
        output_dir=Path("/tmp/_kptsim_tmp"),
        offset=None,
    )


def extract_keypoints_for_task(
    task_name: str,
    task_dir: Path,
    extractor: KeypointPoseExtractor,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract keypoints (pos + quat) for all episodes of a task.

    Reuses the extractor's SAPIEN scene to avoid GPU resource leaks.

    Returns:
        pos_cache: {ep_idx: [T, K, 3]}
        quat_cache: {ep_idx: [T, K, 4]}
        offset: [3]
        global_min: [3]
        global_max: [3]
    """
    extractor.dataset_dir = task_dir
    extractor._world_cache.clear()
    extractor._quat_cache.clear()

    info = read_v21_info(task_dir)
    total_episodes = info["total_episodes"]

    global_min = np.full(3, np.inf, dtype=np.float32)
    global_max = np.full(3, -np.inf, dtype=np.float32)

    for ep_idx in range(total_episodes):
        kpts = extractor.extract_episode(ep_idx)
        global_min = np.minimum(global_min, kpts.min(axis=(0, 1)))
        global_max = np.maximum(global_max, kpts.max(axis=(0, 1)))

    offset = compute_auto_offset(global_min, global_max)

    pos_cache = {}
    quat_cache = {}
    for ep_idx in range(total_episodes):
        pos_world = extractor._world_cache[ep_idx]
        pos_transformed = apply_offset(pos_world, offset)
        pos_cache[ep_idx] = pos_transformed
        quat_cache[ep_idx] = extractor._quat_cache[ep_idx]

    return pos_cache, quat_cache, offset, global_min, global_max


def compute_column_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute min/max/mean/std/count for a column."""
    return {
        "min": values.min(axis=0).tolist() if values.ndim > 1 else float(values.min()),
        "max": values.max(axis=0).tolist() if values.ndim > 1 else float(values.max()),
        "mean": values.mean(axis=0).tolist() if values.ndim > 1 else float(values.mean()),
        "std": values.std(axis=0).tolist() if values.ndim > 1 else float(values.std()),
        "count": int(len(values)),
    }


def compute_ep_column_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute per-episode stats for an array column."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist() if len(values) > 1 else (values[0] * 0).tolist(),
        "count": int(len(values)),
    }


def build_v3_dataset(
    task_name: str,
    task_dir: Path,
    output_dir: Path,
    pos_cache: Dict[int, np.ndarray],
    quat_cache: Dict[int, np.ndarray],
    offset: np.ndarray,
    global_min: np.ndarray,
    global_max: np.ndarray,
) -> dict:
    """Build LeRobot v3 dataset from original v2.1 + extracted keypoints."""
    info = read_v21_info(task_dir)
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    fps = info.get("fps", 15)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build combined data parquet ----
    all_rows = []
    ep_metadata = []
    global_idx = 0

    for ep_idx in range(total_episodes):
        parquet_path = task_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet_path)
        ep_len = len(df)

        pos_data = pos_cache[ep_idx]
        quat_data = quat_cache[ep_idx]

        assert pos_data.shape[0] == ep_len, (
            f"Episode {ep_idx}: pos rows {pos_data.shape[0]} != parquet rows {ep_len}"
        )
        assert quat_data.shape[0] == ep_len

        pos_flat = pos_data.reshape(ep_len, K * 3).astype(np.float32)
        quat_flat = quat_data.reshape(ep_len, K * QUAT_DIM).astype(np.float32)

        for t in range(ep_len):
            row = {
                "observation.state": np.asarray(df["observation.state"].iloc[t], dtype=np.float32),
                "action": np.asarray(df["action"].iloc[t], dtype=np.float32),
                "timestamp": float(df["timestamp"].iloc[t]),
                "frame_index": int(df["frame_index"].iloc[t]),
                "episode_index": int(ep_idx),
                "index": global_idx,
                "task_index": int(df["task_index"].iloc[t]),
                "observation.keypoint_3d": pos_flat[t],
                "observation.keypoint_quat": quat_flat[t],
            }
            all_rows.append(row)
            global_idx += 1

        ep_start = global_idx - ep_len
        ep_metadata.append({
            "episode_index": ep_idx,
            "length": ep_len,
            "dataset_from_index": ep_start,
            "dataset_to_index": global_idx,
        })

    data_df = pd.DataFrame(all_rows)

    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_df.to_parquet(data_dir / "file-000.parquet", index=False)
    print(f"    Data parquet: {len(data_df)} rows, {len(data_df.columns)} cols")

    # ---- 2. Symlink videos ----
    src_video_dir = task_dir / "videos"
    dst_video_dir = output_dir / "videos"
    if dst_video_dir.exists():
        if dst_video_dir.is_symlink():
            dst_video_dir.unlink()
        else:
            shutil.rmtree(dst_video_dir)

    if src_video_dir.exists():
        for cam_dir_v21 in sorted(src_video_dir.iterdir()):
            if cam_dir_v21.name == "chunk-000":
                for cam_subdir in sorted(cam_dir_v21.iterdir()):
                    cam_name = cam_subdir.name
                    v3_cam_dir = dst_video_dir / cam_name / "chunk-000"
                    v3_cam_dir.mkdir(parents=True, exist_ok=True)
                    for vid_file in sorted(cam_subdir.iterdir()):
                        if vid_file.suffix in (".mp4", ".avi", ".mkv"):
                            ep_num = vid_file.stem.replace("episode_", "")
                            dst_file = v3_cam_dir / f"file-{int(ep_num):03d}.mp4"
                            if not dst_file.exists():
                                dst_file.symlink_to(vid_file.resolve())
            else:
                cam_name = cam_dir_v21.name
                v3_cam_dir = dst_video_dir / cam_name / "chunk-000"
                v3_cam_dir.mkdir(parents=True, exist_ok=True)
                chunk_subdir = cam_dir_v21 / "chunk-000" if (cam_dir_v21 / "chunk-000").exists() else cam_dir_v21
                for vid_file in sorted(chunk_subdir.iterdir()):
                    if vid_file.suffix in (".mp4", ".avi", ".mkv"):
                        ep_num = vid_file.stem.replace("episode_", "")
                        dst_file = v3_cam_dir / f"file-{int(ep_num):03d}.mp4"
                        if not dst_file.exists():
                            dst_file.symlink_to(vid_file.resolve())

    # ---- 3. Build meta ----
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 3a. info.json
    v21_features = info.get("features", {})
    v3_features = {}
    for feat_name in ("observation.state", "action"):
        if feat_name in v21_features:
            feat = dict(v21_features[feat_name])
            feat["fps"] = fps
            v3_features[feat_name] = feat

    for img_key in ("observation.images.cam_high", "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist"):
        if img_key in v21_features:
            v3_features[img_key] = v21_features[img_key]

    for scalar_key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if scalar_key in v21_features:
            feat = dict(v21_features[scalar_key])
            feat["fps"] = fps
            v3_features[scalar_key] = feat
        else:
            dtype = "float32" if scalar_key == "timestamp" else "int64"
            v3_features[scalar_key] = {"dtype": dtype, "shape": [1], "names": None, "fps": fps}

    v3_features["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [K * 3],
        "names": KPT_POS_NAMES,
        "fps": fps,
    }
    v3_features["observation.keypoint_quat"] = {
        "dtype": "float32",
        "shape": [K * QUAT_DIM],
        "names": KPT_QUAT_NAMES,
        "fps": fps,
    }

    all_pos = np.concatenate([pos_cache[i].reshape(-1, K * 3) for i in range(total_episodes)], axis=0)
    all_quat = np.concatenate([quat_cache[i].reshape(-1, K * QUAT_DIM) for i in range(total_episodes)], axis=0)
    trans_min = all_pos.min(axis=0)
    trans_max = all_pos.max(axis=0)

    v3_info = {
        "codebase_version": "v3.0",
        "robot_type": info.get("robot_type", "aloha"),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": info.get("total_tasks", total_episodes),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": v3_features,
        "keypoint_coord_mode": "voxel",
        "keypoint_coord_offset": offset.tolist(),
        "keypoint_has_quaternions": True,
        "keypoint_quat_convention": QUAT_CONVENTION,
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(v3_info, f, indent=4)

    # 3b. keypoints_meta.json
    kpt_meta = {
        "K": K,
        "keypoint_names": list(KEYPOINT_NAMES),
        "coord_offset": offset.tolist(),
        "world_range_min": global_min.tolist(),
        "world_range_max": global_max.tolist(),
        "transformed_range_min": trans_min.reshape(K, 3).min(axis=0).tolist(),
        "transformed_range_max": trans_max.reshape(K, 3).max(axis=0).tolist(),
        "urdf_path": str(URDF_PATH),
        "dataset_dir": str(task_dir),
        "total_episodes": total_episodes,
        "has_quaternions": True,
        "quat_dim": QUAT_DIM,
        "quat_convention": QUAT_CONVENTION,
        "quat_filename": "observation.keypoint_quat",
        "quat_coord_frame": "world",
    }
    with open(meta_dir / "keypoints_meta.json", "w", encoding="utf-8") as f:
        json.dump(kpt_meta, f, indent=2)

    # 3c. tasks.parquet
    v21_tasks = read_v21_tasks(task_dir)
    if v21_tasks:
        tasks_rows = {task_entry["task"]: task_entry["task_index"] for task_entry in v21_tasks}
        tasks_df = pd.DataFrame({"task_index": list(tasks_rows.values())}, index=list(tasks_rows.keys()))
        tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # 3d. stats.json
    all_states = np.stack(data_df["observation.state"].values)
    all_actions = np.stack(data_df["action"].values)
    stats = {
        "observation.state": compute_column_stats(all_states),
        "action": compute_column_stats(all_actions),
        "observation.keypoint_3d": compute_column_stats(all_pos),
        "observation.keypoint_quat": compute_column_stats(all_quat),
        "timestamp": compute_column_stats(np.array(data_df["timestamp"].values, dtype=np.float32)),
        "frame_index": compute_column_stats(np.array(data_df["frame_index"].values, dtype=np.int64)),
        "episode_index": compute_column_stats(np.array(data_df["episode_index"].values, dtype=np.int64)),
        "index": compute_column_stats(np.array(data_df["index"].values, dtype=np.int64)),
        "task_index": compute_column_stats(np.array(data_df["task_index"].values, dtype=np.int64)),
    }
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    # 3e. norm_stat.json
    norm_stat = {
        "observation.state": {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        },
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        },
        "observation.keypoint_3d": {
            "mean": all_pos.mean(axis=0).tolist(),
            "std": all_pos.std(axis=0).tolist(),
        },
        "observation.keypoint_quat": {
            "mean": all_quat.mean(axis=0).tolist(),
            "std": all_quat.std(axis=0).tolist(),
        },
    }
    with open(output_dir / "norm_stat.json", "w", encoding="utf-8") as f:
        json.dump(norm_stat, f, indent=4)

    # 3f. episodes metadata parquet
    v21_episodes = read_v21_episodes(task_dir)
    ep_meta_dir = meta_dir / "episodes" / "chunk-000"
    ep_meta_dir.mkdir(parents=True, exist_ok=True)

    ep_records = []
    for em in ep_metadata:
        ep_idx = em["episode_index"]
        ep_pos = pos_cache[ep_idx].reshape(-1, K * 3)
        ep_quat = quat_cache[ep_idx].reshape(-1, K * QUAT_DIM)

        parquet_path = task_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df_ep = pd.read_parquet(parquet_path)
        ep_states = np.stack(df_ep["observation.state"].values)
        ep_actions = np.stack(df_ep["action"].values)
        ep_timestamps = np.array(df_ep["timestamp"].values, dtype=np.float32)
        ep_frame_idx = np.array(df_ep["frame_index"].values, dtype=np.int64)

        task_text = ""
        if ep_idx < len(v21_episodes) and "tasks" in v21_episodes[ep_idx]:
            task_list = v21_episodes[ep_idx]["tasks"]
            task_text = task_list if isinstance(task_list, str) else task_list[0] if task_list else ""

        record = {
            "episode_index": ep_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": em["dataset_from_index"],
            "dataset_to_index": em["dataset_to_index"],
            "tasks": [task_text] if isinstance(task_text, str) else task_text,
            "length": em["length"],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            cam_key = f"observation.images.{cam}"
            record[f"videos/{cam_key}/chunk_index"] = 0
            record[f"videos/{cam_key}/file_index"] = ep_idx
            record[f"videos/{cam_key}/from_timestamp"] = float(ep_timestamps[0]) if len(ep_timestamps) > 0 else 0.0
            record[f"videos/{cam_key}/to_timestamp"] = float(ep_timestamps[-1]) if len(ep_timestamps) > 0 else 0.0

        for col_name, col_data in [
            ("observation.state", ep_states),
            ("action", ep_actions),
            ("observation.keypoint_3d", ep_pos),
            ("observation.keypoint_quat", ep_quat),
        ]:
            s = compute_ep_column_stats(col_data)
            record[f"stats/{col_name}/min"] = s["min"]
            record[f"stats/{col_name}/max"] = s["max"]
            record[f"stats/{col_name}/mean"] = s["mean"]
            record[f"stats/{col_name}/std"] = s["std"]
            record[f"stats/{col_name}/count"] = s["count"]

        for col_name, col_data in [
            ("timestamp", ep_timestamps),
            ("frame_index", ep_frame_idx),
            ("episode_index", np.full(len(df_ep), ep_idx, dtype=np.int64)),
            ("index", np.arange(em["dataset_from_index"], em["dataset_to_index"], dtype=np.int64)),
            ("task_index", np.array(df_ep["task_index"].values, dtype=np.int64)),
        ]:
            s = compute_column_stats(col_data)
            record[f"stats/{col_name}/min"] = s["min"]
            record[f"stats/{col_name}/max"] = s["max"]
            record[f"stats/{col_name}/mean"] = s["mean"]
            record[f"stats/{col_name}/std"] = s["std"]
            record[f"stats/{col_name}/count"] = s["count"]

        ep_records.append(record)

    ep_df = pd.DataFrame(ep_records)
    ep_df.to_parquet(ep_meta_dir / "file-000.parquet", index=False)

    return {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "offset": offset.tolist(),
        "pos_range": [trans_min.reshape(K, 3).min(axis=0).tolist(),
                      trans_max.reshape(K, 3).max(axis=0).tolist()],
    }


def validate_task_output(task_name: str, output_dir: Path) -> Tuple[bool, List[str]]:
    """Validate a single task's v3 output."""
    issues = []

    if not (output_dir / "meta" / "info.json").exists():
        issues.append("Missing meta/info.json")
        return False, issues

    with open(output_dir / "meta" / "info.json") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]

    data_path = output_dir / "data" / "chunk-000" / "file-000.parquet"
    if not data_path.exists():
        issues.append("Missing data parquet")
        return False, issues

    df = pd.read_parquet(data_path)
    if len(df) != total_frames:
        issues.append(f"Frame count mismatch: parquet={len(df)}, info={total_frames}")

    if "observation.keypoint_3d" not in df.columns:
        issues.append("Missing observation.keypoint_3d column")
    else:
        sample_pos = np.array(df["observation.keypoint_3d"].iloc[0])
        if len(sample_pos) != K * 3:
            issues.append(f"keypoint_3d shape: expected {K*3}, got {len(sample_pos)}")

    if "observation.keypoint_quat" not in df.columns:
        issues.append("Missing observation.keypoint_quat column")
    else:
        sample_quat = np.array(df["observation.keypoint_quat"].iloc[0])
        if len(sample_quat) != K * QUAT_DIM:
            issues.append(f"keypoint_quat shape: expected {K*QUAT_DIM}, got {len(sample_quat)}")

        all_quats = np.stack(df["observation.keypoint_quat"].values).reshape(-1, K, QUAT_DIM)
        norms = np.linalg.norm(all_quats, axis=-1)
        norm_err = np.abs(norms - 1.0).max()
        if norm_err > 1e-3:
            issues.append(f"Non-unit quaternion: max |norm-1|={norm_err:.6f}")

    all_pos = np.stack(df["observation.keypoint_3d"].values).reshape(-1, K, 3)
    is_valid, stats = validate_range(all_pos)
    if not is_valid:
        issues.append(f"Keypoints out of voxel range: {stats['out_of_range_count']} points")

    for meta_file in ("keypoints_meta.json", "stats.json", "tasks.parquet"):
        if not (output_dir / "meta" / meta_file).exists():
            issues.append(f"Missing meta/{meta_file}")

    if not (output_dir / "norm_stat.json").exists():
        issues.append("Missing norm_stat.json")

    return len(issues) == 0, issues


def process_single_task(
    task_name: str,
    data_root: Path,
    extractor: KeypointPoseExtractor,
) -> dict:
    """Process a single task end-to-end: extract + convert + validate."""
    task_dir = data_root / task_name
    output_dir = data_root / f"{task_name}{OUTPUT_SUFFIX}"

    result = {
        "task": task_name,
        "status": "pending",
        "time_s": 0,
        "total_episodes": 0,
        "total_frames": 0,
        "error": None,
        "validation_issues": [],
    }

    t0 = time.time()
    try:
        print(f"\n{'='*60}")
        print(f"  Processing: {task_name}")
        print(f"{'='*60}")

        info = read_v21_info(task_dir)
        result["total_episodes"] = info["total_episodes"]
        result["total_frames"] = info["total_frames"]

        print(f"  Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
        print(f"  Extracting keypoints (pos + quat) via SAPIEN FK...")

        pos_cache, quat_cache, offset, g_min, g_max = extract_keypoints_for_task(
            task_name, task_dir, extractor
        )
        t_extract = time.time() - t0
        print(f"  Extraction done in {t_extract:.1f}s, offset={offset}")

        print(f"  Building LeRobot v3 dataset...")
        build_result = build_v3_dataset(
            task_name, task_dir, output_dir,
            pos_cache, quat_cache, offset, g_min, g_max,
        )
        t_build = time.time() - t0 - t_extract
        print(f"  Build done in {t_build:.1f}s")

        print(f"  Validating...")
        valid, issues = validate_task_output(task_name, output_dir)
        result["validation_issues"] = issues
        if valid:
            result["status"] = "success"
            print(f"  PASS")
        else:
            result["status"] = "validation_failed"
            for iss in issues:
                print(f"  [ISSUE] {iss}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    result["time_s"] = round(time.time() - t0, 1)
    print(f"  Total time: {result['time_s']}s | Status: {result['status']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract keypoints + quaternions → LeRobot v3"
    )
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--urdf_path", type=str, default=str(URDF_PATH))
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Specific tasks to process (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip tasks that already have output dirs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    urdf_path = Path(args.urdf_path)

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = discover_tasks(data_root)

    if args.skip_existing:
        tasks = [t for t in tasks if not (data_root / f"{t}{OUTPUT_SUFFIX}").exists()]

    print(f"Tasks to process: {len(tasks)}")
    print(f"Data root: {data_root}")
    print(f"URDF: {urdf_path}")
    print(f"Output suffix: {OUTPUT_SUFFIX}")

    print("Creating shared SAPIEN FK scene (reused across all tasks)...")
    extractor = create_shared_extractor(urdf_path)

    results = []
    t_total = time.time()

    try:
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] ", end="")
            r = process_single_task(task, data_root, extractor)
            results.append(r)
    finally:
        extractor.close()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {len(results)} tasks in {elapsed:.0f}s")
    print(f"{'='*60}")

    success = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    print(f"  Success: {len(success)}/{len(results)}")
    if failed:
        print(f"  Failed:")
        for r in failed:
            print(f"    {r['task']}: {r['status']} — {r.get('error') or r.get('validation_issues')}")

    summary_path = data_root / "_batch_lrb3_kptsim7_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_time_s": round(elapsed, 1)}, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
