import argparse
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser()

    ################ training setting
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--seed', default=75, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--num_train_steps', default=40000, type=int)

    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--base_lr', default=2.5e-5, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=1e-10, type=float)

    parser.add_argument('--lr_schedule', default='CosineDecay', type=str)
    parser.add_argument('--decay_lr', default=2.5e-6, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--decay_steps', default=30000, type=int)

    parser.add_argument('--ema_decay', default=0.99, type=float)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--save_interval', default=5000, type=int)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--action_horizon', default=50, type=int)
    parser.add_argument('--action_dim', default=32, type=int)
    parser.add_argument('--max_token_len', default=48, type=int)

    ################ deepspeed setting
    parser.add_argument('--local_rank', default=-1, type=int)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args
