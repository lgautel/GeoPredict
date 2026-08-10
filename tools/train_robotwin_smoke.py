#!/usr/bin/env python3
"""Smoke training script for GeoPredict on RoboTwin keypoint data."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.robotwin_dataset import RoboTwinDataset
from models.geopredict import GeoPredict
from utils.utils import move_to_device


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key == "images":
            out[key] = {
                sub_key: torch.stack([item[key][sub_key] for item in batch], dim=0)
                for sub_key in values[0]
            }
        elif key == "image_masks":
            out[key] = {
                sub_key: torch.stack([item[key][sub_key] for item in batch], dim=0)
                for sub_key in values[0]
            }
        elif isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        elif key == "cam_infos":
            out[key] = {
                cam_key: {
                    sub_key: torch.stack([item[key][cam_key][sub_key] for item in batch], dim=0)
                    for sub_key in values[0][cam_key]
                }
                for cam_key in values[0]
            }
        elif key == "step":
            out[key] = torch.tensor(values, dtype=torch.long)
        elif isinstance(values[0], dict):
            out[key] = {
                sub_key: torch.stack([item[key][sub_key] for item in batch], dim=0)
                for sub_key in values[0]
            }
        else:
            out[key] = values[0] if len(set(map(repr, values))) == 1 else values
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three",
    )
    parser.add_argument(
        "--keypoints_dir",
        type=str,
        default="/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim",
    )
    parser.add_argument("--num_train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RoboTwinDataset(
        dataset_dir=args.dataset_dir,
        keypoints_dir=args.keypoints_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    print(f"Dataset size: {len(dataset)}")

    model = GeoPredict(
        embed_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        joint_num=RoboTwinDataset.JOINT_NUM,
        use_depth_loss=False,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-10)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    data_iter = iter(dataloader)
    train_stats = defaultdict(float)

    for step in range(args.num_train_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            losses, loss_dict, _ = model(batch)

        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss at step {step + 1}: {loss_dict}")

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_stats["total_loss"] += losses.item()
        for key, value in loss_dict.items():
            train_stats[key] += value

        if (step + 1) % args.log_interval == 0:
            avg = {k: v / args.log_interval for k, v in train_stats.items()}
            msg = f"Step [{step + 1}/{args.num_train_steps}] "
            msg += " ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(msg)
            train_stats = defaultdict(float)

    print("Smoke training completed successfully.")


if __name__ == "__main__":
    main()
