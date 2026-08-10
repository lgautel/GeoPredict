"""Integration test: extract episode_000000."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from b.script.kpt.config import DATASET_DIR, K, OUTPUT_DIR, URDF_PATH
from b.script.kpt.coord_transform import apply_offset, compute_auto_offset, validate_range
from b.script.kpt.keypoint_extractor import KeypointExtractor


@pytest.fixture(scope="module")
def extracted_episode(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("kpt_extract")
    extractor = KeypointExtractor(
        urdf_path=URDF_PATH,
        dataset_dir=DATASET_DIR,
        output_dir=out_dir,
    )
    kpts = extractor.extract_episode(0)
    offset = compute_auto_offset(kpts.min(axis=(0, 1)), kpts.max(axis=(0, 1)))
    kpts_t = apply_offset(kpts, offset)
    extractor._save_episode_keypoints(0, kpts_t.reshape(kpts_t.shape[0], K * 3))
    extractor.close()
    return kpts, kpts_t, out_dir


class TestExtractSingle:
    def test_extract_shape_matches_parquet(self, extracted_episode):
        kpts, _, _ = extracted_episode
        parquet_path = DATASET_DIR / "data" / "chunk-000" / "episode_000000.parquet"
        df = pd.read_parquet(parquet_path)
        assert kpts.shape == (len(df), K, 3)

    def test_frame_continuity(self, extracted_episode):
        kpts, _, _ = extracted_episode
        diffs = np.linalg.norm(np.diff(kpts, axis=0), axis=-1)
        assert diffs.max() < 0.05

    def test_transformed_in_voxel_range(self, extracted_episode):
        _, kpts_t, _ = extracted_episode
        is_valid, _ = validate_range(kpts_t)
        assert is_valid

    def test_saved_npy_format(self, extracted_episode):
        _, _, out_dir = extracted_episode
        saved = np.load(out_dir / "episode_000000" / "keypoints.npy")
        assert saved.dtype == np.float32
        assert saved.shape[1] == K * 3

    def test_left_base_stable(self, extracted_episode):
        kpts, _, _ = extracted_episode
        base = kpts[:, 0, :]
        drift = np.linalg.norm(base - base[0], axis=1).max()
        assert drift < 0.02
