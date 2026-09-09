#!/usr/bin/env python3
"""Convert LIBERO-plus RLDS TFRecord → LeRobot v3.0 with FK keypoints.

Two-phase pipeline:
  Phase 1: Parse RLDS + MuJoCo FK → intermediate cache + global bbox
  Phase 2: Normalize → write parquet + video + metadata

Usage:
    conda run -n phantom python convert_libero_rlds.py
    conda run -n phantom python convert_libero_rlds.py --normalization rpad
    conda run -n phantom python convert_libero_rlds.py --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ─── Ensure imports work when run as script ───
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from kpt_libero.config_libero import (
    BBOX_MARGIN,
    CACHE_DIR,
    CHUNKS_SIZE,
    FPS,
    K,
    KEYPOINT_BODY_NAMES,
    KEYPOINT_NAMES,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PANDA_XML_PATH,
    POS_FEATURE_NAMES,
    QUAT_CONVENTION,
    QUAT_DIM,
    QUAT_FEATURE_NAMES,
    RLDS_ROOT,
    VIDEO_CODEC,
    VIDEO_CRF,
    VIDEO_PIX_FMT,
    VOXEL_CENTER,
    VOXEL_RANGE_MAX,
    VOXEL_RANGE_MIN,
)
from kpt_libero.mujoco_fk import MujocoFKScene
from kpt_libero.rlds_reader import iter_episodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Parse RLDS + FK → cache + global bbox
# ═══════════════════════════════════════════════════════════════

def phase1_parse_and_fk(
    rlds_root: Path,
    xml_path: Path,
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, int, int, Dict[str, int]]:
    """Parse all RLDS episodes, run FK, cache to disk, collect global bbox."""
    fk = MujocoFKScene(xml_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_episodes = 0
    total_frames = 0
    task_map: Dict[str, int] = {}

    qw_min_global = 1.0
    quat_norm_err_max = 0.0

    t0 = time.time()
    for ep_data in iter_episodes(rlds_root):
        ep_idx = total_episodes

        # FK
        positions, quaternions = fk.extract_episode(ep_data["qpos"])

        # Track global bbox
        pos_flat = positions.reshape(-1, 3)
        global_min = np.minimum(global_min, pos_flat.min(axis=0))
        global_max = np.maximum(global_max, pos_flat.max(axis=0))

        # Quaternion checks
        qw_vals = quaternions[:, :, 0]  # wxyz → qw is index 0
        qw_min_global = min(qw_min_global, float(qw_vals.min()))
        quat_norms = np.linalg.norm(quaternions.reshape(-1, 4), axis=1)
        quat_norm_err_max = max(
            quat_norm_err_max, float(np.abs(quat_norms - 1.0).max())
        )

        # Task mapping
        lang = ep_data["language"]
        if lang not in task_map:
            task_map[lang] = len(task_map)

        # Cache episode data
        cache_path = cache_dir / f"ep_{ep_idx:06d}.pkl"
        cache_data = {
            "state": ep_data["state"],
            "joint_state": ep_data["joint_state"],
            "action": ep_data["action"],
            "positions": positions,
            "quaternions": quaternions,
            "language": lang,
            "file_path": ep_data["file_path"],
            "n_steps": ep_data["n_steps"],
            "image": ep_data["image"],
            "wrist_image": ep_data["wrist_image"],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        total_frames += ep_data["n_steps"]
        total_episodes += 1

        if total_episodes % 200 == 0:
            elapsed = time.time() - t0
            rate = total_episodes / elapsed
            eta_min = (14347 - total_episodes) / max(rate, 0.01) / 60
            logger.info(
                "Phase 1: %d episodes, %d frames (%.1f ep/s, ETA %.0f min)",
                total_episodes, total_frames, rate, eta_min,
            )

    fk.close()

    logger.info("Phase 1 complete: %d episodes, %d frames", total_episodes, total_frames)
    logger.info("Global pos min: %s", global_min.astype(np.float32))
    logger.info("Global pos max: %s", global_max.astype(np.float32))
    logger.info("qw_min=%.6f, quat_norm_err_max=%.2e", qw_min_global, quat_norm_err_max)

    if qw_min_global < -1e-6:
        raise RuntimeError(f"Hemisphere normalization failed: qw_min={qw_min_global}")
    if quat_norm_err_max > 0.01:
        raise RuntimeError(f"Quaternion norm error too large: {quat_norm_err_max}")

    return (
        global_min.astype(np.float32),
        global_max.astype(np.float32),
        total_episodes,
        total_frames,
        task_map,
    )


# ═══════════════════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════════════════

def compute_auto_offset(
    global_min: np.ndarray,
    global_max: np.ndarray,
    target_center: np.ndarray = VOXEL_CENTER,
) -> np.ndarray:
    workspace_center = (global_min + global_max) / 2.0
    return (workspace_center - target_center).astype(np.float32)


def compute_r_pad(
    global_min: np.ndarray,
    global_max: np.ndarray,
    margin: float = BBOX_MARGIN,
) -> float:
    abs_extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    R = float(abs_extremes.max())
    return R * (1.0 + margin)


def normalize_positions(
    positions: np.ndarray, normalization: str, norm_params: Dict
) -> np.ndarray:
    if normalization == "auto_offset":
        return (positions - norm_params["offset"]).astype(np.float32)
    elif normalization == "rpad":
        return (positions / norm_params["r_pad"]).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


# ═══════════════════════════════════════════════════════════════
#  Video encoding
# ═══════════════════════════════════════════════════════════════

def encode_video_chunk(
    video_dir: Path,
    episodes: List[Dict],
    cam_key: str,
    chunk_idx: int,
) -> Path:
    """Encode all frames from episodes in a chunk into one MP4 file."""
    video_key = f"observation.images.{cam_key}"
    out_dir = video_dir / video_key / f"chunk-{chunk_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file-000.mp4"

    # Collect all frames
    all_frames = []
    for ep in episodes:
        for jpeg_bytes in ep[cam_key]:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                logger.warning("Failed to decode JPEG frame, using black frame")
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            all_frames.append(img)

    if not all_frames:
        return out_path

    h, w = all_frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", VIDEO_CODEC,
        "-pix_fmt", VIDEO_PIX_FMT,
        "-crf", VIDEO_CRF,
        str(out_path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    for frame in all_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed for {out_path}: {stderr[-500:]}")

    return out_path


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Normalize + write LeRobot v3.0
# ═══════════════════════════════════════════════════════════════

def phase2_write_dataset(
    cache_dir: Path,
    output_root: Path,
    total_episodes: int,
    total_frames: int,
    task_map: Dict[str, int],
    normalization: str,
    norm_params: Dict[str, Any],
    encode_video: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Read cached episodes, normalize, write parquet + video + meta."""
    output_root.mkdir(parents=True, exist_ok=True)

    vec_stats: Dict[str, Dict] = {}
    global_frame_idx = 0
    ep_meta_rows = []

    video_dir = output_root / "videos"
    num_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE

    for chunk_idx in range(num_chunks):
        ep_start = chunk_idx * CHUNKS_SIZE
        ep_end = min(ep_start + CHUNKS_SIZE, total_episodes)
        chunk_episodes = []

        parquet_rows = []
        for ep_idx in range(ep_start, ep_end):
            cache_path = cache_dir / f"ep_{ep_idx:06d}.pkl"
            with open(cache_path, "rb") as f:
                ep = pickle.load(f)
            chunk_episodes.append(ep)

            T = ep["n_steps"]
            task_idx = task_map[ep["language"]]

            positions = normalize_positions(ep["positions"], normalization, norm_params)
            quaternions = ep["quaternions"]  # already hemisphere-normalized

            pos_flat = positions.reshape(T, K * 3)
            quat_flat = quaternions.reshape(T, K * QUAT_DIM)

            for t in range(T):
                row = {
                    "observation.state": ep["state"][t].tolist(),
                    "observation.joint_state": ep["joint_state"][t].tolist(),
                    "action": ep["action"][t].tolist(),
                    "observation.keypoint_3d": pos_flat[t].tolist(),
                    "observation.keypoint_quat": quat_flat[t].tolist(),
                    "language_instruction": ep["language"],
                    "timestamp": float(t) / FPS,
                    "frame_index": t,
                    "episode_index": ep_idx,
                    "index": global_frame_idx,
                    "task_index": task_idx,
                }
                parquet_rows.append(row)
                global_frame_idx += 1

            # Accumulate stats
            for col_name, arr in [
                ("observation.state", ep["state"]),
                ("observation.joint_state", ep["joint_state"]),
                ("action", ep["action"]),
                ("observation.keypoint_3d", pos_flat),
                ("observation.keypoint_quat", quat_flat),
            ]:
                if col_name not in vec_stats:
                    dim = arr.shape[1]
                    vec_stats[col_name] = {
                        "min": np.full(dim, np.inf, dtype=np.float64),
                        "max": np.full(dim, -np.inf, dtype=np.float64),
                        "sum": np.zeros(dim, dtype=np.float64),
                        "sum_sq": np.zeros(dim, dtype=np.float64),
                        "count": 0,
                    }
                vs = vec_stats[col_name]
                vs["min"] = np.minimum(vs["min"], arr.min(axis=0))
                vs["max"] = np.maximum(vs["max"], arr.max(axis=0))
                vs["sum"] += arr.sum(axis=0)
                vs["sum_sq"] += (arr.astype(np.float64) ** 2).sum(axis=0)
                vs["count"] += arr.shape[0]

            ep_meta_rows.append({
                "episode_index": ep_idx,
                "task_index": task_idx,
                "length": T,
                "task": ep["language"],
                "file_path": ep["file_path"],
            })

        # Write parquet
        data_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(parquet_rows)
        pq_path = data_dir / "file-000.parquet"
        df.to_parquet(pq_path, index=False)
        logger.info(
            "Chunk %d/%d: wrote %d rows → %s",
            chunk_idx + 1, num_chunks, len(df), pq_path,
        )

        # Encode videos
        if encode_video:
            for cam_key in ("image", "wrist_image"):
                encode_video_chunk(video_dir, chunk_episodes, cam_key, chunk_idx)
                logger.info("  Video %s chunk-%03d done", cam_key, chunk_idx)

        del chunk_episodes, parquet_rows, df

    # Compute final stats
    final_stats = {}
    for col_name, vs in vec_stats.items():
        mean = vs["sum"] / vs["count"]
        std = np.sqrt(np.maximum(vs["sum_sq"] / vs["count"] - mean ** 2, 0))
        final_stats[col_name] = {
            "min": vs["min"].tolist(),
            "max": vs["max"].tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": int(vs["count"]),
        }

    return final_stats, ep_meta_rows


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Write metadata
# ═══════════════════════════════════════════════════════════════

def phase3_write_metadata(
    output_root: Path,
    total_episodes: int,
    total_frames: int,
    task_map: Dict[str, int],
    stats: Dict,
    ep_meta_rows: List[Dict],
    normalization: str,
    norm_params: Dict[str, Any],
    global_min: np.ndarray,
    global_max: np.ndarray,
    xml_path: str,
    encode_video: bool = True,
) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    num_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE

    # info.json
    video_features = {}
    if encode_video:
        for cam in ("image", "wrist_image"):
            video_features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": 256, "video.width": 256,
                    "video.codec": "h264", "video.pix_fmt": VIDEO_PIX_FMT,
                    "video.is_depth_map": False, "video.fps": FPS,
                    "video.channels": 3, "has_audio": False,
                },
            }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_map),
        "total_videos": total_episodes * 2 if encode_video else 0,
        "total_chunks": num_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": [["x", "y", "z", "r1", "r2", "r3", "gripper_L", "gripper_R"]],
            },
            "observation.joint_state": {
                "dtype": "float32", "shape": [7],
                "names": [["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]],
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": [["dx", "dy", "dz", "dr1", "dr2", "dr3", "gripper"]],
            },
            "observation.keypoint_3d": {
                "dtype": "float32", "shape": [K * 3],
                "names": [POS_FEATURE_NAMES],
            },
            "observation.keypoint_quat": {
                "dtype": "float32", "shape": [K * QUAT_DIM],
                "names": [QUAT_FEATURE_NAMES],
            },
            **video_features,
            "language_instruction": {"dtype": "string", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Wrote meta/info.json")

    # tasks.parquet
    tasks_rows = [{"task_index": idx, "task": lang}
                  for lang, idx in sorted(task_map.items(), key=lambda x: x[1])]
    pd.DataFrame(tasks_rows).to_parquet(meta_dir / "tasks.parquet", index=False)
    logger.info("Wrote meta/tasks.parquet (%d tasks)", len(tasks_rows))

    # stats.json
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote meta/stats.json")

    # keypoints_meta.json
    kpt_meta = {
        "K": K,
        "keypoint_names": KEYPOINT_NAMES,
        "keypoint_body_names": KEYPOINT_BODY_NAMES,
        "position_dim": 3,
        "quaternion_dim": QUAT_DIM,
        "quaternion_convention": QUAT_CONVENTION,
        "hemisphere_constraint": "qw >= 0; negate all 4 if qw < 0",
        "normalization_method": normalization,
        "global_pos_min": global_min.tolist(),
        "global_pos_max": global_max.tolist(),
        "mjcf_xml": str(xml_path),
        "robot_base_offset": "embedded in MJCF ([-0.56, 0, 0.912] for robosuite Lift)",
        "fps": FPS,
        "source": "LIBERO-plus RLDS (libero_mix)",
    }
    if normalization == "auto_offset":
        kpt_meta["auto_offset"] = norm_params["offset"].tolist()
        kpt_meta["voxel_range_min"] = VOXEL_RANGE_MIN.tolist()
        kpt_meta["voxel_range_max"] = VOXEL_RANGE_MAX.tolist()
    elif normalization == "rpad":
        kpt_meta["r_pad"] = norm_params["r_pad"]
        kpt_meta["bbox_margin"] = BBOX_MARGIN
    with open(meta_dir / "keypoints_meta.json", "w") as f:
        json.dump(kpt_meta, f, indent=2)
    logger.info("Wrote meta/keypoints_meta.json")

    # episodes/
    ep_df = pd.DataFrame(ep_meta_rows)
    for chunk_idx in range(num_chunks):
        ep_start = chunk_idx * CHUNKS_SIZE
        ep_end = min(ep_start + CHUNKS_SIZE, total_episodes)
        chunk_ep_df = ep_df[
            (ep_df["episode_index"] >= ep_start) & (ep_df["episode_index"] < ep_end)
        ].copy()
        ep_dir = meta_dir / "episodes" / f"chunk-{chunk_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        chunk_ep_df.to_parquet(ep_dir / "file-000.parquet", index=False)
    logger.info("Wrote meta/episodes/ (%d chunks)", num_chunks)

    # norm_stat.json (GeoPredict compat)
    norm_stat = {
        "keypoint_3d": stats.get("observation.keypoint_3d", {}),
        "keypoint_quat": stats.get("observation.keypoint_quat", {}),
    }
    with open(output_root / "norm_stat.json", "w") as f:
        json.dump(norm_stat, f, indent=2)
    logger.info("Wrote norm_stat.json")


# ═══════════════════════════════════════════════════════════════
#  Cache helpers
# ═══════════════════════════════════════════════════════════════

def save_bbox_cache(cache_dir, global_min, global_max, total_episodes, total_frames, task_map):
    meta = {
        "global_min": global_min.tolist(),
        "global_max": global_max.tolist(),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "task_map": task_map,
    }
    with open(cache_dir / "_bbox_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_bbox_cache(cache_dir):
    with open(cache_dir / "_bbox_meta.json") as f:
        meta = json.load(f)
    return (
        np.array(meta["global_min"], dtype=np.float32),
        np.array(meta["global_max"], dtype=np.float32),
        meta["total_episodes"],
        meta["total_frames"],
        meta["task_map"],
    )


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO-plus RLDS → LeRobot v3.0 with FK keypoints.",
    )
    parser.add_argument("--rlds_root", type=str, default=str(RLDS_ROOT))
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--xml_path", type=str, default=str(PANDA_XML_PATH))
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    parser.add_argument(
        "--normalization", choices=["auto_offset", "rpad"], default="auto_offset",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_video", action="store_true", help="Skip video encoding")
    parser.add_argument("--bbox_margin", type=float, default=BBOX_MARGIN)
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes for testing (default: all)")
    args = parser.parse_args()

    rlds_root = Path(args.rlds_root)
    output_root = Path(args.output)
    xml_path = Path(args.xml_path)
    cache_dir = Path(args.cache_dir)
    encode_video = not args.no_video

    if not xml_path.exists():
        logger.error("Panda MJCF not found at %s. Run export_panda_xml.py first.", xml_path)
        sys.exit(1)
    if not rlds_root.exists():
        logger.error("RLDS root not found: %s", rlds_root)
        sys.exit(1)

    # Phase 1
    if args.resume and (cache_dir / "_bbox_meta.json").exists():
        logger.info("=== Resuming from Phase 1 cache ===")
        global_min, global_max, total_episodes, total_frames, task_map = load_bbox_cache(cache_dir)
    else:
        logger.info("=== Phase 1: Parse RLDS + MuJoCo FK ===")
        global_min, global_max, total_episodes, total_frames, task_map = (
            phase1_parse_and_fk(rlds_root, xml_path, cache_dir)
        )
        save_bbox_cache(cache_dir, global_min, global_max, total_episodes, total_frames, task_map)

    if args.max_episodes:
        total_episodes = min(total_episodes, args.max_episodes)
        logger.info("Limiting to %d episodes", total_episodes)

    # Compute normalization params
    norm_params: Dict[str, Any] = {}
    if args.normalization == "auto_offset":
        offset = compute_auto_offset(global_min, global_max)
        norm_params["offset"] = offset
        logger.info("Auto-offset: %s", offset)
    elif args.normalization == "rpad":
        r_pad = compute_r_pad(global_min, global_max, margin=args.bbox_margin)
        norm_params["r_pad"] = r_pad
        logger.info("R_pad: %.6f m (margin=%.0f%%)", r_pad, args.bbox_margin * 100)

    # Phase 2
    logger.info("=== Phase 2: Normalize + Write LeRobot v3.0 (video=%s) ===", encode_video)
    stats, ep_meta_rows = phase2_write_dataset(
        cache_dir, output_root, total_episodes, total_frames,
        task_map, args.normalization, norm_params, encode_video,
    )

    # Phase 3
    logger.info("=== Phase 3: Write metadata ===")
    phase3_write_metadata(
        output_root, total_episodes, total_frames, task_map,
        stats, ep_meta_rows, args.normalization, norm_params,
        global_min, global_max, str(xml_path), encode_video,
    )

    logger.info("=== DONE ===")
    logger.info("  Output: %s", output_root)
    logger.info("  Episodes: %d, Frames: %d", total_episodes, total_frames)
    logger.info("  Tasks: %d", len(task_map))
    logger.info("  Normalization: %s", args.normalization)
    logger.info("  Keypoints: K=%d × (3 pos + 4 quat) = %d + %d dim",
                K, K * 3, K * QUAT_DIM)


if __name__ == "__main__":
    main()
