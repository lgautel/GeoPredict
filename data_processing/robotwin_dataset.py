"""RoboTwin LeRobot dataset loader with SAPIEN keypoint GT."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_processing.data_transform import _parse_image, load_norm_stats
from data_processing.transforms import pad_to_dim
from models.tokenizer import PaligemmaTokenizer


class RoboTwinDataset(Dataset):
    JOINT_NUM = 14
    VIDEO_KEYS = {
        "left_rgb": "observation.images.cam_high",
        "right_rgb": "observation.images.cam_right_wrist",
        "wrist_rgb": "observation.images.cam_left_wrist",
    }

    def __init__(
        self,
        dataset_dir: str | Path,
        keypoints_dir: str | Path,
        norm_stats_path: str | Path = "./ckpts/robotwin_norm_stats.json",
        tokenizer_path: str | Path = "./ckpts/paligemma_tokenizer.model",
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.keypoints_dir = Path(keypoints_dir)
        self.img_size = 224
        self.action_dim = 32
        self.action_horizon = 50
        self.max_token_len = 48
        self.delta_idx = list(range(self.action_horizon))
        self.norm_stats = load_norm_stats(norm_stats_path)
        self.tokenizer = PaligemmaTokenizer(self.max_token_len, path=str(tokenizer_path))

        episodes_path = self.dataset_dir / "meta" / "episodes.jsonl"
        self.episode_meta = {}
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.episode_meta[item["episode_index"]] = item

        self.infos = []
        for ep_idx, meta in sorted(self.episode_meta.items()):
            for step in range(meta["length"]):
                self.infos.append(
                    {
                        "episode_index": ep_idx,
                        "step": step,
                        "step_num": meta["length"],
                        "prompt": meta["tasks"][0],
                    }
                )

    def __len__(self) -> int:
        return len(self.infos)

    @staticmethod
    @lru_cache(maxsize=256)
    def _load_parquet(path_str: str) -> pd.DataFrame:
        return pd.read_parquet(path_str)

    def _read_video_frame(self, video_key: str, episode_index: int, frame_index: int) -> np.ndarray:
        video_path = (
            self.dataset_dir
            / "videos"
            / "chunk-000"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        return iio.imread(video_path, index=frame_index)

    def __getitem__(self, index: int):
        info = self.infos[index]
        ep_idx = info["episode_index"]
        step = info["step"]
        step_num = info["step_num"]
        prompt = info["prompt"]

        parquet_path = (
            self.dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        )
        df = self._load_parquet(str(parquet_path))
        row = df.iloc[step]

        images = {}
        for out_key, video_key in self.VIDEO_KEYS.items():
            frame = self._read_video_frame(video_key, ep_idx, int(row["frame_index"]))
            images[out_key] = _parse_image(frame, self.img_size, self.img_size)

        state_raw = np.asarray(row["observation.state"], dtype=np.float32).copy()
        state = torch.from_numpy(state_raw).float()
        state = (state - self.norm_stats["state"]["mean"]) / (self.norm_stats["state"]["std"] + 1e-6)
        state = pad_to_dim(state, self.action_dim)

        query_indices = [max(0, min(step_num - 1, step + delta)) for delta in self.delta_idx]
        actions_raw = np.stack(
            [np.asarray(df.iloc[i]["action"], dtype=np.float32) for i in query_indices],
            axis=0,
        )
        actions = torch.from_numpy(actions_raw).float()
        actions = (actions - self.norm_stats["actions"]["mean"]) / (self.norm_stats["actions"]["std"] + 1e-6)
        actions = pad_to_dim(actions, self.action_dim)

        tokens, token_masks = self.tokenizer.tokenize(str(prompt))

        keypoints = np.load(self.keypoints_dir / f"episode_{ep_idx:06d}" / "keypoints.npy")
        his_kpts = torch.zeros((1000, self.JOINT_NUM, 3), dtype=torch.float32)
        if step == 0:
            his_len = 1
        else:
            kpts_hist = torch.from_numpy(keypoints[:step].reshape(-1, self.JOINT_NUM, 3)).float()
            his_kpts[: kpts_hist.shape[0]] = kpts_hist
            his_len = kpts_hist.shape[0]

        kpt_t = torch.from_numpy(keypoints[step].reshape(self.JOINT_NUM, 3)).float()
        future_query_indices = [
            max(0, min(step_num - 1, step + 1 + delta)) for delta in range(self.action_horizon)
        ]
        future_kpts = torch.zeros((self.action_horizon, self.JOINT_NUM, 3), dtype=torch.float32)
        for i, query_step in enumerate(future_query_indices):
            future_kpts[i] = torch.from_numpy(
                keypoints[query_step].reshape(self.JOINT_NUM, 3)
            ).float()

        img_size = self.img_size
        left_depth_t = torch.zeros((img_size, img_size), dtype=torch.float32)
        right_depth_t = torch.zeros((img_size, img_size), dtype=torch.float32)
        left_depth_future = torch.zeros((self.action_horizon, img_size, img_size), dtype=torch.float32)
        right_depth_future = torch.zeros((self.action_horizon, img_size, img_size), dtype=torch.float32)

        identity_inner = torch.eye(3, dtype=torch.float32)
        identity_outer = torch.eye(4, dtype=torch.float32)

        return {
            "token": f"episode_{ep_idx:06d}",
            "step": step,
            "future_steps": torch.tensor(future_query_indices, dtype=torch.long),
            "state": state,
            "images": images,
            "image_masks": {
                "left_rgb": torch.tensor(True),
                "right_rgb": torch.tensor(True),
                "wrist_rgb": torch.tensor(True),
            },
            "actions": actions,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "his_kpts": his_kpts,
            "his_len": torch.tensor(his_len, dtype=torch.long),
            "kpt_t": kpt_t,
            "future_kpts": future_kpts,
            "depths_t": {
                "left_depth": left_depth_t,
                "right_depth": right_depth_t,
            },
            "depths_future": {
                "left_depth": left_depth_future,
                "right_depth": right_depth_future,
            },
            "cam_infos": {
                "left": {"inner": identity_inner, "outer": identity_outer},
                "right": {"inner": identity_inner, "outer": identity_outer},
            },
            "use_depth_loss": False,
        }
