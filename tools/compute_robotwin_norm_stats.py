#!/usr/bin/env python3
"""Compute z-score normalization stats for RoboTwin stack_bowls_three."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(dataset_dir: Path) -> dict:
    parquet_dir = dataset_dir / "data" / "chunk-000"
    states = []
    actions = []
    for parquet_path in sorted(parquet_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(parquet_path)
        states.append(np.stack(df["observation.state"].tolist(), axis=0))
        actions.append(np.stack(df["action"].tolist(), axis=0))

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)

    def _stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": np.maximum(arr.std(axis=0), 1e-6).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    return {"state": _stats(states), "actions": _stats(actions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./ckpts/robotwin_norm_stats.json",
    )
    args = parser.parse_args()

    stats = compute_stats(Path(args.dataset_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved norm stats to {output_path}")


if __name__ == "__main__":
    main()
