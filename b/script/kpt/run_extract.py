#!/usr/bin/env python3
"""CLI entry for SAPIEN FK keypoint extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from b.script.kpt.config import DATASET_DIR, OUTPUT_DIR, URDF_PATH
from b.script.kpt.keypoint_extractor import KeypointExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3D keypoint trajectories from RoboTwin LeRobot dataset via SAPIEN FK"
    )
    parser.add_argument("--urdf_path", type=str, default=str(URDF_PATH))
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--offset", type=float, nargs=3, default=None)
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    offset = None if args.offset is None else args.offset
    extractor = KeypointExtractor(
        urdf_path=args.urdf_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        offset=offset,
    )
    try:
        if args.episode is not None:
            kpts = extractor.extract_episode(args.episode)
            print(
                f"Episode {args.episode}: shape={kpts.shape}, "
                f"min={kpts.min(axis=(0, 1))}, max={kpts.max(axis=(0, 1))}"
            )
        else:
            extractor.extract_all()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
