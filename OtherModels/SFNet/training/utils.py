"""Training utilities."""

import math
import torch
import torch.optim as optim


def setup_training(model, config):
    """Setup optimizer and scheduler for training."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def cosine_warmup_scheduler(optimizer, config, steps_per_epoch):
    """Create cosine annealing scheduler with warmup."""
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return config.min_lr / config.learning_rate + \
               (1 - config.min_lr / config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

