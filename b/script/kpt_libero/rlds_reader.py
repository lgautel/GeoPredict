"""Parse LIBERO-plus RLDS TFRecord files into Python dicts.

Each TFRecord example represents one episode with flattened step features.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterator, Any

import numpy as np
import tensorflow as tf


def parse_episode(raw_record: bytes) -> Dict[str, Any]:
    """Parse a single RLDS TFRecord example into numpy arrays.

    Returns dict with keys:
        qpos:        [T, 9] float32  (arm 7 + gripper 2)
        state:       [T, 8] float32  (EEF state)
        joint_state: [T, 7] float32  (arm joints)
        action:      [T, 7] float32  (delta action)
        image:       list of T JPEG bytes  (agentview)
        wrist_image: list of T JPEG bytes  (wrist cam)
        language:    str
        file_path:   str
        n_steps:     int
    """
    ex = tf.train.Example()
    ex.ParseFromString(raw_record)
    f = ex.features.feature

    n_steps = len(f["steps/is_first"].int64_list.value)

    joint_state = np.array(
        f["steps/observation/joint_state"].float_list.value, dtype=np.float32
    ).reshape(n_steps, 7)

    state = np.array(
        f["steps/observation/state"].float_list.value, dtype=np.float32
    ).reshape(n_steps, 8)

    action = np.array(
        f["steps/action"].float_list.value, dtype=np.float32
    ).reshape(n_steps, 7)

    # Construct 9-dim qpos: arm [7] + gripper fingers [2]
    gripper = state[:, 6:8]
    qpos = np.concatenate([joint_state, gripper], axis=1)  # [T, 9]

    # Language instruction (repeated per step, take first)
    language = f["steps/language_instruction"].bytes_list.value[0].decode("utf-8")

    # Episode metadata
    file_path = f["episode_metadata/file_path"].bytes_list.value[0].decode("utf-8")

    # Images as raw JPEG bytes (NOT decoded yet, to save memory)
    image_bytes = list(f["steps/observation/image"].bytes_list.value)
    wrist_image_bytes = list(f["steps/observation/wrist_image"].bytes_list.value)

    return {
        "qpos": qpos,
        "state": state,
        "joint_state": joint_state,
        "action": action,
        "image": image_bytes,
        "wrist_image": wrist_image_bytes,
        "language": language,
        "file_path": file_path,
        "n_steps": n_steps,
    }


def iter_episodes(rlds_root: str | Path) -> Iterator[Dict[str, Any]]:
    """Iterate over all episodes across all shards in RLDS root.

    Yields parsed episode dicts in shard order.
    """
    rlds_root = Path(rlds_root)
    shard_pattern = str(rlds_root / "libero_mix-train.tfrecord-*")
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        raise FileNotFoundError(
            f"No TFRecord shards found matching {shard_pattern}"
        )

    for shard_path in shard_files:
        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
            yield parse_episode(raw_record.numpy())


def count_shards(rlds_root: str | Path) -> int:
    """Count number of TFRecord shards."""
    rlds_root = Path(rlds_root)
    shard_pattern = str(rlds_root / "libero_mix-train.tfrecord-*")
    return len(sorted(glob.glob(shard_pattern)))
