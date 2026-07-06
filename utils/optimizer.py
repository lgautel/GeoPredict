import math
import torch.optim as optim


def build_optimizer(args, model):
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.base_lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError
    
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_schedule == "CosineDecay":
        base_lr = args.base_lr
        warmup_steps = args.warmup_steps
        init_lr = base_lr / (warmup_steps + 1)
        decay_steps = args.decay_steps
        decay_lr = args.decay_lr

        def lr_lambda(step):
            if step < warmup_steps:
                return (init_lr + (base_lr - init_lr) * step / warmup_steps) / base_lr
            elif step < decay_steps:
                cosine_steps = decay_steps - warmup_steps
                progress = (step - warmup_steps) / cosine_steps
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                current_lr = decay_lr + (base_lr - decay_lr) * cosine_factor
                return current_lr / base_lr
            else:
                return decay_lr / base_lr
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError

    return scheduler
