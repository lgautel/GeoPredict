import deepspeed
import numpy as np
import os
import os.path as osp
import random
import shutil
import sys
import time
import torch
import torch.distributed as dist
from collections import defaultdict
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.robocasa_dataset import RobocasaDataset
from models.geopredict import GeoPredict
from utils.arguments import parse_args
from utils.ema import moving_average, save_zero_three_model
from utils.optimizer import build_lr_scheduler, build_optimizer
from utils.utils import build_logger, move_to_device


def all_reduce_stats(stats_dict, world_size, device):
    reduced_stats = {}
    
    for key, value in stats_dict.items():
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced_stats[key] = tensor.item() / world_size
    
    return reduced_stats


def main():
    args = parse_args()
    rank = args.local_rank

    master_port = int(os.environ.get('MASTER_PORT', None))
    print(f"deepspeed init with master_port: {master_port}")
    deepspeed.init_distributed()
    
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    world_size = int(os.environ["WORLD_SIZE"])

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{args.name}"
    experiment_dir = osp.join("experiments", experiment_name)
    checkpoints_dir = osp.join(experiment_dir, "checkpoints")
    if rank == 0:
        if osp.exists(experiment_dir):
            print(f"Experiment directory exists but no valid checkpoints found. Starting fresh training.")
            shutil.rmtree(experiment_dir)

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        logger = build_logger(filepath=osp.join(experiment_dir, "log.txt"), verbose=False, mode="w")
    
    dist.barrier()

    dataset = RobocasaDataset(data_root='./data/robocasa')

    model = GeoPredict(embed_dtype=torch.bfloat16)
    missing_keys, unexpected_keys = model.load_state_dict(
        torch.load(args.pretrain, map_location='cpu'), strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys in state_dict: {unexpected_keys}")
    if missing_keys:
        white_keyword = ['keypoint', 'spatial', 'gs_decoder', 'renderer', 'refine']
        non_white_missing = [key for key in missing_keys if not any(keyword in key for keyword in white_keyword)]
        if non_white_missing:
            raise RuntimeError(f"Missing critical keys: {non_white_missing}")
        else:
            if rank == 0:
                logger.info(f"Ignored {white_keyword}-related missing keys: {missing_keys}")
    if rank == 0:
        logger.info(f"Using DDP with {world_size} GPUs")
    
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    model_ema, _, _, _ = deepspeed.initialize(args=args, model=model)
    model, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        training_data=dataset,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )

    dist.barrier()

    model.train()
    dataloader_iter = iter(dataloader)
    train_stats = defaultdict(float)
    start_time = time.time()
    if rank == 0:
        logger.info(f"Starting training for {args.num_train_steps} steps")
        logger.info(f"Length of dataloader is {len(dataloader)}")
    
    for step in range(args.num_train_steps):
        try:
            batch_data = next(dataloader_iter)
        except StopIteration:
            if rank == 0:
                logger.info(f"Starting new epoch: {step // len(dataloader)}")
            dataloader_iter = iter(dataloader)
            batch_data = next(dataloader_iter)
        batch_data = move_to_device(batch_data, device)

        losses, loss_dict, acc_dict = model(batch_data)
        model.backward(losses)
        model.step()
        moving_average(model, model_ema)
        train_stats["total_loss"] += losses.item()
        for k, v in loss_dict.items():
            train_stats[k] += v
        for k, v in acc_dict.items():
            train_stats[k] += v

        if (step + 1) % args.log_interval == 0:
            local_avg_stats = {key: value / args.log_interval for key, value in train_stats.items()}
            global_avg_stats = all_reduce_stats(local_avg_stats, world_size, rank)

            if rank == 0:
                elapsed_time = time.time() - start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                log_msg = f"Step [{step + 1}/{args.num_train_steps}] "
                log_msg += f"Time: {elapsed_time:.2f}s "
                log_msg += f"LR: {current_lr:.2e} "
                for key, value in global_avg_stats.items():
                    log_msg += f"{key}: {value:.4f} "
                logger.info(log_msg)
            
            train_stats = defaultdict(float)
            start_time = time.time()
        
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = osp.join(checkpoints_dir, f"checkpoint_step_{step + 1}.pth")
            save_zero_three_model(model_ema, rank, checkpoint_path)
            if rank == 0:
                logger.info(f"Saved checkpoint at step {step + 1}")
    
    checkpoint_path = osp.join(checkpoints_dir, "final_checkpoint.pth")
    save_zero_three_model(model_ema, rank, checkpoint_path)
    if rank == 0:
        logger.info("Training completed!")


if __name__ == '__main__':
    main()
