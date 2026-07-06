import cv2
import json
import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset

from models.tokenizer import PaligemmaTokenizer
from .data_transform import _parse_image, load_norm_stats
from .transforms import pad_to_dim


class RobocasaDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()

        self.img_size = 224
        self.action_dim = 32
        self.action_horizon = 50
        self.max_token_len = 48
        self.delta_idx = [i for i in range(self.action_horizon)]
        self.norm_stats = load_norm_stats("./ckpts/robocasa_norm_stats.json")
        self.tokenizer = PaligemmaTokenizer(self.max_token_len)

        data_dir = Path(data_root) / 'data'
        prompt_path = Path(data_root) / 'meta' / 'episodes.json'
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_dict = json.load(file)

        infos = []
        for ep_name in os.listdir(data_dir):
            meta_infos = np.load(data_dir / ep_name / 'infos.npy')

            for step in range(meta_infos.shape[0]):
                info = {
                    'index': step,
                    'token': ep_name,
                    'data_dir': data_dir,
                    'prompt': prompt_dict[ep_name],
                }

                infos.append(info)
        
        self.infos = infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        info = self.infos[index]

        step = info['index']
        ep_name = info['token']
        data_dir = info['data_dir']
        prompt = info['prompt']

        labels = np.load(data_dir / ep_name / 'infos.npy')
        step_num = labels.shape[0]

        left_image = cv2.cvtColor(cv2.imread(str(data_dir / ep_name / 'agentview_left_image' / f'step_{step:04}.png')), cv2.COLOR_BGR2RGB)
        left_image = _parse_image(left_image, self.img_size, self.img_size)  # 224x224
        right_image = cv2.cvtColor(cv2.imread(str(data_dir / ep_name / 'agentview_right_image' / f'step_{step:04}.png')), cv2.COLOR_BGR2RGB)
        right_image = _parse_image(right_image, self.img_size, self.img_size)  # 224x224
        wrist_image = cv2.cvtColor(cv2.imread(str(data_dir / ep_name / 'eye_in_hand_image' / f'step_{step:04}.png')), cv2.COLOR_BGR2RGB)
        wrist_image = _parse_image(wrist_image, self.img_size, self.img_size)  # 224x224

        state = torch.from_numpy(labels[step][:8]).float()
        state = pad_to_dim(state, self.action_dim)
        state = (state - self.norm_stats['state']['mean']) / (self.norm_stats['state']['std'] + 1e-6)

        query_indices = [max(0, min(step_num - 1, step + delta)) for delta in self.delta_idx]
        actions = torch.from_numpy(labels[query_indices, 8:20]).float()
        actions = pad_to_dim(actions, self.action_dim)
        actions = (actions - self.norm_stats['actions']['mean']) / (self.norm_stats['actions']['std'] + 1e-6)

        tokens, token_masks = self.tokenizer.tokenize(str(prompt))

        # History keypoint
        keypoints = np.load(data_dir / ep_name / 'keypoints.npy')  # [step_num, 8*3]
        his_kpts = torch.zeros((1000, 8, 3), dtype=torch.float32)
        if step == 0:
            his_len = 1
        else:
            kpts_ = torch.from_numpy(keypoints[:step].reshape(-1, 8, 3)).float()
            his_kpts[:kpts_.shape[0]] = kpts_
            his_len = kpts_.shape[0]

        # current keypoint and depth
        kpt_t = torch.from_numpy(keypoints[step].reshape(8, 3)).float()
        depth_np = np.load(data_dir / ep_name / 'agentview_left_depth' / f'step_{step:04}.npy')  # 256x256
        left_depth_t = torch.from_numpy(cv2.resize(
            depth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)).float()  # 224x224
        depth_np = np.load(data_dir / ep_name / 'agentview_right_depth' / f'step_{step:04}.npy')  # 256x256
        right_depth_t = torch.from_numpy(cv2.resize(
            depth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)).float()  # 224x224
        
        # future keypoint and depth
        future_query_indices = [max(0, min(step_num - 1, step + 1 + delta)) for delta in range(self.action_horizon)]

        future_kpts = torch.zeros((self.action_horizon, 8, 3), dtype=torch.float32)
        for i, query_step in enumerate(future_query_indices):
            kpts_ = torch.from_numpy(keypoints[query_step].reshape(8, 3)).float()
            future_kpts[i] = kpts_

        left_depth_future = torch.zeros((self.action_horizon, self.img_size, self.img_size), dtype=torch.float)
        right_depth_future = torch.zeros((self.action_horizon, self.img_size, self.img_size), dtype=torch.float)
        for i, query_step in enumerate(future_query_indices):
            depth_np = np.load(data_dir / ep_name / 'agentview_left_depth' / f'step_{query_step:04}.npy')  # 256x256
            depth_resized = torch.from_numpy(cv2.resize(
                depth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)).float()  # 224x224
            left_depth_future[i] = depth_resized
            depth_np = np.load(data_dir / ep_name / 'agentview_right_depth' / f'step_{query_step:04}.npy')  # 256x256
            depth_resized = torch.from_numpy(cv2.resize(
                depth_np, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)).float()  # 224x224
            right_depth_future[i] = depth_resized
        
        replay_size = 256
        scale_matrix = torch.tensor([
            [self.img_size/replay_size,                         0, 0],
            [                        0, self.img_size/replay_size, 0], 
            [                        0,                         0, 1]
        ], dtype=torch.float32)

        agentview_left_cam = torch.from_numpy(np.load(data_dir / ep_name / 'cams.npy')).float()[1]
        left_inner = scale_matrix @ agentview_left_cam[:9].reshape(3, 3)
        left_outer = agentview_left_cam[9:].reshape(4, 4)
        agentview_right_cam = torch.from_numpy(np.load(data_dir / ep_name / 'cams.npy')).float()[3]
        right_inner = scale_matrix @ agentview_right_cam[:9].reshape(3, 3)
        right_outer = agentview_right_cam[9:].reshape(4, 4)

        item = {
            "token": ep_name,
            "step": step,
            "future_steps": torch.tensor(future_query_indices, dtype=torch.long),
            "state": state,
            "images": {
                "left_rgb": left_image,
                "right_rgb": right_image,
                "wrist_rgb": wrist_image,
            },
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
            "cam_infos" : {
                "left": {
                    "inner": left_inner,
                    "outer": left_outer
                },
                "right": {
                    "inner": right_inner,
                    "outer": right_outer
                },
            },
        }

        return item
