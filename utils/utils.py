import logging
import torch


def build_logger(filepath=None, verbose=False, mode="w"):
    logger = logging.getLogger("logger")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if verbose:
        handler1 = logging.StreamHandler()
        handler1.setLevel(logging.INFO)
        handler1.setFormatter(formatter)
        logger.addHandler(handler1)
    
    if filepath is not None:
        handler2 = logging.FileHandler(filename=filepath, mode=mode)
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)

    return logger


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data
