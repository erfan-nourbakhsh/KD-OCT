"""Learning rate schedulers."""

import numpy as np
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer, max_lr, min_lr, warmup_steps, total_steps, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
            lrs.append(lr)
        
        return lrs

