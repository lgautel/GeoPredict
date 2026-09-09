#!/usr/bin/env python3
"""Inspect libero_plus RLDS features.json and print one sample episode schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _collect_feature_paths(obj: dict, prefix: str = "") -> list[tuple[str, str, str | None]]:
    """Return (path, dtype_or_type, shape) from TFDS features.json."""
    rows: list[tuple[str, str, str | None]] = []

    def walk(node: dict, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "tensor" in node:
            tensor = node["tensor"]
            shape = "x".join(tensor.get("shape", {}).get("dimensions", [])) or "scalar"
            rows.append((path, tensor.get("dtype", "?"), shape))
            return
        if "image" in node:
            image = node["image"]
            shape = "x".join(image.get("shape", {}).get("dimensions", []))
            rows.append((path, image.get("dtype", "image"), shape))
            return
        if "text" in node:
            rows.append((path, "text", None))
            return
        if "featuresDict" in node:
            for name, child in node["featuresDict"].get("features", {}).items():
                walk(child, f"{path}.{name}" if path else name)
            return
        if "sequence" in node:
            walk(node["sequence"].get("feature", {}), path)
            return
        for key in ("features", "feature"):
            if key in node and isinstance(node[key], dict):
                if key == "features":
                    for name, child in node[key].items():
                        walk(child, f"{path}.{name}" if path else name)
                else:
                    walk(node[key], path)

    walk(obj, prefix)
    return rows


def print_features(features_path: Path) -> dict:
    data = json.loads(features_path.read_text())
    print("=== features.json (top-level keys) ===")
    print(json.dumps(list(data.keys()), indent=2))

    rows = _collect_feature_paths(data)
    print("\n=== feature fields ===")
    for path, dtype, shape in rows:
        shape_str = shape if shape else ""
        print(f"  {path}: {dtype} {shape_str}")

    joint_rows = [r for r in rows if "joint" in r[0].lower()]
    print("\n=== joint-related fields in features.json ===")
    if joint_rows:
        for path, dtype, shape in joint_rows:
            print(f"  {path}: {dtype} {shape}")
    else:
        print("  (none)")
    print(f"  has joint_states: {any(r[0].endswith('joint_states') for r in rows)}")
    print(f"  has joint_state: {any(r[0].endswith('joint_state') for r in rows)}")
    return data


def inspect_episode_tfrecord(tfds_dir: Path) -> None:
    import tensorflow as tf

    shards = sorted(tfds_dir.glob("libero_mix-train.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"No train shards under {tfds_dir}")

    raw = next(iter(tf.data.TFRecordDataset([str(shards[0])]).take(1))).numpy()
    example = tf.train.Example()
    example.ParseFromString(raw)

    keys = sorted(example.features.feature.keys())
    print("\n=== one episode via TFRecord (flattened keys) ===")
    print(f"  shard: {shards[0].name}")
    print(f"  keys ({len(keys)}): {keys}")

    def feature_info(name: str) -> tuple[str, int, object]:
        feat = example.features.feature[name]
        if feat.bytes_list.value:
            lens = [len(x) for x in feat.bytes_list.value]
            return "bytes", len(lens), lens[:3]
        if feat.float_list.value:
            vals = list(feat.float_list.value)
            return "float32", len(vals), vals[:7]
        if feat.int64_list.value:
            vals = list(feat.int64_list.value)
            return "int64", len(vals), vals[:7]
        return "?", 0, []

    step_count = feature_info("steps/is_first")[1]
    print(f"\n=== episode summary ===")
    print(f"  num_steps: {step_count}")

    print("\n=== per-step tensor shapes (inferred) ===")
    for key in keys:
        if "image" in key:
            dtype, count, sample = feature_info(key)
            print(f"  {key}: jpeg bytes, steps={count}, sample_sizes={sample}")
            continue
        dtype, count, sample = feature_info(key)
        if key.startswith("steps/") and dtype == "float32" and count % step_count == 0:
            dim = count // step_count
            print(f"  {key}: float32 [{dim}] x {step_count} steps, first={sample}")
        elif key.startswith("steps/") and dtype == "int64" and count == step_count:
            print(f"  {key}: int64/bool scalar x {step_count} steps, first={sample}")
        elif key.startswith("steps/language_instruction"):
            print(f"  {key}: text x {count} steps")
        elif key.startswith("episode_metadata/"):
            val = example.features.feature[key].bytes_list.value[0].decode("utf-8", errors="replace")
            print(f"  {key}: {val[:120]}{'...' if len(val) > 120 else ''}")
        else:
            print(f"  {key}: {dtype} count={count}")

    print("\n=== joint_states check (TFRecord) ===")
    print(f"  has joint_states: {'steps/observation/joint_states' in keys}")
    print(f"  has joint_state: {'steps/observation/joint_state' in keys}")
    if "steps/observation/joint_state" in keys:
        dtype, count, sample = feature_info("steps/observation/joint_state")
        dim = count // step_count
        print(f"  joint_state inferred shape per step: [{dim}]")


def inspect_episode_tfds(tfds_dir: Path) -> None:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(tfds_dir))
    info = builder.info
    print("\n=== builder.info ===")
    print(f"  name: {info.name}")
    print(f"  version: {info.version}")
    print(f"  splits: {list(info.splits.keys())}")
    print(f"  supervised_keys: {info.supervised_keys}")

    ds = builder.as_dataset(split="train")
    episode = next(iter(ds.take(1)))

    print("\n=== one episode (top-level keys) ===")
    print(list(episode.keys()))

    steps = episode["steps"]
    step = next(iter(steps.take(1)))

    print("\n=== one step (keys) ===")
    print(list(step.keys()))

    obs = step.get("observation")
    if obs is not None:
        print("\n=== observation keys ===")
        obs_keys = list(obs.keys()) if hasattr(obs, "keys") else list(obs)
        print(obs_keys)
        for key in sorted(obs_keys):
            if "joint" in key.lower() or "state" in key.lower() or "proprio" in key.lower():
                val = obs[key] if hasattr(obs, "__getitem__") else getattr(obs, key)
                shape = getattr(val, "shape", None)
                print(f"  ** {key}: shape={shape}")

    print("\n=== joint_states check ===")
    step_keys = [k.decode() if isinstance(k, bytes) else str(k) for k in step.keys()]
    has_step_joint = "joint_states" in step_keys or "joint_state" in step_keys
    print(f"  step has joint_states/joint_state: {has_step_joint}")

    if obs is not None:
        obs_key_strs = []
        for k in obs.keys() if hasattr(obs, "keys") else []:
            obs_key_strs.append(k.decode() if isinstance(k, bytes) else str(k))
        joint_like = [k for k in obs_key_strs if "joint" in k.lower()]
        state_like = [k for k in obs_key_strs if "state" in k.lower() or "proprio" in k.lower()]
        print(f"  observation keys containing 'joint': {joint_like}")
        print(f"  observation keys containing 'state'/'proprio': {state_like}")

    print("\n=== step tensor shapes (non-image) ===")
    for k in step.keys():
        name = k.decode() if isinstance(k, bytes) else str(k)
        if "image" in name.lower():
            continue
        v = step[k]
        if hasattr(v, "shape"):
            print(f"  {name}: shape={tuple(v.shape)}, dtype={v.dtype}")


def inspect_episode(tfds_dir: Path, backend: str) -> None:
    if backend == "tfrecord":
        inspect_episode_tfrecord(tfds_dir)
        return

    if backend == "tfds":
        inspect_episode_tfds(tfds_dir)
        return

    if backend == "auto":
        try:
            import tensorflow_datasets as tfds  # noqa: F401
            inspect_episode_tfds(tfds_dir)
        except Exception as exc:
            print(f"\nTFDS unavailable ({exc}); falling back to raw TFRecord parsing.")
            inspect_episode_tfrecord(tfds_dir)
        return

    raise ValueError(f"Unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfds-dir",
        type=Path,
        default=Path.home() / "DATA/libero_plus_rlds/libero_mix/1.0.0",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "tfds", "tfrecord"),
        default="auto",
        help="Episode reader: auto tries TFDS then TFRecord fallback.",
    )
    args = parser.parse_args()
    tfds_dir = args.tfds_dir
    features_path = tfds_dir / "features.json"
    if not features_path.is_file():
        print(f"features.json not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    print_features(features_path)
    inspect_episode(tfds_dir, args.backend)


if __name__ == "__main__":
    main()
