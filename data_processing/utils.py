import torch
import torchvision.transforms as transforms


IMAGE_KEYS = (
    "left_rgb",
    "right_rgb",
    "wrist_rgb",
)


def transform_images(images, image_keys, train):
    out_images = {}

    for key in image_keys:
        image = images[key]  # BCHW or BNCHW, [0, 1]

        if train:
            original_shape = image.shape
            image = image.reshape(-1, *original_shape[-3:])  # [N, C, H, W]
            batch_shape = image.shape[0]

            if "wrist" not in key:
                height, width = image.shape[-2:]
                transform = transforms.Compose([
                    transforms.RandomCrop((int(height * 0.95), int(width * 0.95))),
                    transforms.Resize((height, width)),
                    transforms.RandomRotation(degrees=5),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ])
            
            # Apply transforms to each image in the batch
            transformed_images = []
            for i in range(batch_shape):
                transformed = transform(image[i])
                transformed_images.append(transformed)
            
            image = torch.stack(transformed_images)
            image = image.reshape(*original_shape[:-3], *image.shape[-3:])
            
        # Convert to [-1, 1]
        image = image * 2.0 - 1.0
        out_images[key] = image
    
    return out_images


def preprocess_observation(observation, train=False, image_keys=IMAGE_KEYS):
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """
    if not set(image_keys).issubset(observation["images"]):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation['images'])}")
    
    results = {
        "state": observation["state"],
        "tokenized_prompt": observation["tokenized_prompt"],
        "tokenized_prompt_mask": observation["tokenized_prompt_mask"],
        "his_kpts": observation["his_kpts"],
        "his_len": observation["his_len"],
    }

    results["image_masks"] = observation["image_masks"]
    results["images"] = transform_images(observation["images"], image_keys, train)
        
    return results
