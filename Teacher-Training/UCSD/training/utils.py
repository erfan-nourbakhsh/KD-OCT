"""Training utilities."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .losses import FocalLoss
from models import split_head_backbone_params

# Try to import timm
try:
    import timm
    from timm.loss import SoftTargetCrossEntropy
    try:
        from timm.data.mixup import Mixup
    except Exception:
        from timm.data import Mixup
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False
    Mixup = None


def create_optimizer(model: nn.Module, config):
    """Create optimizer with separate learning rates for head and backbone."""
    head_params, backbone_params = split_head_backbone_params(model)
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': config.learning_rate_head},
        {'params': backbone_params, 'lr': config.learning_rate_backbone},
    ], weight_decay=config.weight_decay)
    return optimizer


def cosine_with_warmup_lambda(epoch: int, total_epochs: int, warmup_epochs: int, min_lr: float, base_lr: float):
    """Cosine annealing with warmup learning rate schedule."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_factor = min_lr / base_lr
    return min_factor + (1.0 - min_factor) * cosine


def create_scheduler(optimizer: optim.Optimizer, config):
    """Create learning rate scheduler."""
    base_lr = max(config.learning_rate_head, config.learning_rate_backbone)
    lr_lambda = lambda e: cosine_with_warmup_lambda(e, config.epochs, config.warmup_epochs, config.min_lr, base_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def create_criteria(config, class_weights: np.ndarray, device):
    """Create training and validation loss functions."""
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    if config.use_focal_loss:
        train_criterion = FocalLoss(alpha=weight_tensor, gamma=config.focal_gamma)
        val_criterion = FocalLoss(alpha=weight_tensor, gamma=config.focal_gamma)
        print(f"Using Focal Loss (gamma={config.focal_gamma})")
    elif TIMM_AVAILABLE and config.use_mixup and Mixup is not None:
        train_criterion = SoftTargetCrossEntropy()
        val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        train_criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)
        val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    return train_criterion, val_criterion


def prepare_mixup(config):
    """Prepare Mixup/CutMix augmentation if enabled."""
    if TIMM_AVAILABLE and config.use_mixup and Mixup is not None and not config.use_focal_loss:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            switch_prob=config.mixup_switch_prob,
            mode=config.mixup_mode,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes,
        )
        return mixup_fn
    return None

