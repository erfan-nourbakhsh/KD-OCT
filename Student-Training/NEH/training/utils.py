"""Training utilities for optimizer and scheduler."""

import math
import torch.optim as optim


def create_optimizer(model, config):
    """Create optimizer for student model"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(1, config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = config.min_lr / config.learning_rate
        return min_factor + (1.0 - min_factor) * cosine
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

