import torch
from PIL import Image


def compose(transforms):
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


def pad_to_dim(x, target_dim, axis=-1):
    """Pad a tensor to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [0] * (len(x.shape) * 2)
        pad_index = (len(x.shape) - 1 - (axis % len(x.shape))) * 2
        pad_width[pad_index + 1] = target_dim - current_dim
        return torch.nn.functional.pad(x, pad_width)
    return x


def resize_with_pad_pil(image, height, width, method):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


class CompositeTransform(object):
    """A composite transform that applies a sequence of transforms in order."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class Normalize(object):
    def __init__(self, norm_stats, use_quantiles=False):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles

    def __call__(self, data):
        if self.norm_stats is None:
            return data

        for key, value in self.norm_stats.items():
            if key not in data:
                continue
            
            data[key] = (data[key] - value["mean"]) / (value["std"] + 1e-6)
        
        return data


class RobocasaInputs(object):
    def __init__(self, action_dim, model_type="PI0"):
        self.action_dim = action_dim
        self.model_type = model_type

    def __call__(self, data):
        state = pad_to_dim(data["observation/state"], self.action_dim)

        left_image = data["observation/left_image"]  # chw, float32, [0,1]
        right_image = data["observation/right_image"]  # chw, float32, [0,1]
        wrist_image = data["observation/wrist_image"]  # chw, float32, [0,1]

        inputs = {
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
        }

        if "actions" in data:
            actions = pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


class TokenizePrompt(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
