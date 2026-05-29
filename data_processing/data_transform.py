import json
import numpy as np
import torch
from PIL import Image

from models.tokenizer import PaligemmaTokenizer
from .transforms import compose, resize_with_pad_pil, Normalize, RobocasaInputs, TokenizePrompt


def _parse_image(image, height, width, method=Image.BILINEAR):
    pil_image = Image.fromarray(image)
    resized_pil = resize_with_pad_pil(pil_image, height, width, method=method)
    resized_hwc = np.array(resized_pil).astype(np.float32) / 255.0  # Back to [0,1]
    resized_chw = torch.from_numpy(resized_hwc).permute(2, 0, 1)  # [C, H, W]

    # chw, float32, [0,1]
    return resized_chw


def convert_to_batch(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    elif isinstance(data, dict):
        return {key: convert_to_batch(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_to_batch(item) for item in data)
    else:
        return data


def load_norm_stats(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    norm_stats = {}
    for key, stats in data.items():
        norm_stats[key] = {
            "mean": torch.tensor(stats["mean"], dtype=torch.float32),
            "std": torch.tensor(stats["std"], dtype=torch.float32),
            "q01": torch.tensor(stats["q01"], dtype=torch.float32),
            "q99": torch.tensor(stats["q99"], dtype=torch.float32),
        }
    
    return norm_stats


class RobocasaInputTransform(object):
    def __init__(self,
                 action_dim=32,
                 max_token_len=48):
        transforms = [
            RobocasaInputs(action_dim=action_dim),
            Normalize(norm_stats=load_norm_stats("./ckpts/robocasa_norm_stats.json")),
            TokenizePrompt(PaligemmaTokenizer(max_token_len))
        ]
        self._transform = compose(transforms)

    def __call__(self, data):
        resize_size = data.pop("resize_size")
        left_image = _parse_image(data["observation/left_image"], resize_size, resize_size)
        right_image = _parse_image(data["observation/right_image"], resize_size, resize_size)
        wrist_image = _parse_image(data["observation/wrist_image"], resize_size, resize_size)
        state = torch.from_numpy(data["observation/state"]).float()
        his_kpts = torch.from_numpy(data["his_kpts"]).float()
        his_len = torch.tensor(data["his_len"], dtype=torch.long)

        inputs = {
            "observation/left_image": left_image,  # 3x224x224, float32, [0,1]
            "observation/right_image": right_image,  # 3x224x224, float32, [0,1]
            "observation/wrist_image": wrist_image,  # 3x224x224, float32, [0,1]
            "observation/state": state, # 8, float
            "prompt": data["prompt"],  # N words, string
        }

        inputs = self._transform(inputs)
        inputs["his_kpts"] = his_kpts
        inputs["his_len"] = his_len
        outputs = convert_to_batch(inputs)

        return outputs


class RobocasaOutputTransform(object):
    def __init__(self):
        self.norm_stats = load_norm_stats("./ckpts/robocasa_norm_stats.json")

    def __call__(self, data):
        # First step
        outputs = {}
        for key, value in data.items():
            outputs[key] = value[0]
        
        # Second Step
        if self.norm_stats is None:
            pass
        else:
            for key, value in self.norm_stats.items():
                if key not in outputs:
                    continue
                
                outputs[key] = outputs[key] * (value["std"] + 1e-6) + value["mean"]
        
        return {"actions": np.asarray(outputs["actions"][:, :12])}
