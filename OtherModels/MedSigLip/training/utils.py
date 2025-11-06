"""Training utilities."""

import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import class_weight
from .losses import LabelSmoothingCrossEntropy, FocalLoss
from .scheduler import CosineWarmupScheduler


def setup_optimization(model, config, steps_per_epoch, is_frozen_phase: bool):
    """Setup optimizer and scheduler."""
    
    if is_frozen_phase:
        # Only optimize classification head
        params = [p for p in model.head.parameters() if p.requires_grad]
        lr = config.learning_rate_head
    else:
        # Different learning rates for encoder and head
        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        
        params = [
            {'params': encoder_params, 'lr': config.learning_rate_encoder},
            {'params': head_params, 'lr': config.learning_rate_head}
        ]
        lr = config.learning_rate_head  # For scheduler
    
    optimizer = optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
    
    epochs = config.epochs_frozen if is_frozen_phase else config.epochs_unfrozen
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(config.warmup_epochs * steps_per_epoch))
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        lr,
        config.min_lr,
        warmup_steps,
        total_steps
    )
    
    return optimizer, scheduler


def setup_losses(config, train_labels_np):
    """Setup loss functions."""
    cls_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels_np),
        y=train_labels_np
    )
    
    primary = LabelSmoothingCrossEntropy(config.label_smoothing)
    focal = FocalLoss(alpha=cls_weights, gamma=config.focal_gamma)
    
    return primary, focal, cls_weights


def apply_mixup_cutmix(images, labels, mixup_alpha, cutmix_alpha, cutmix_prob):
    """Apply MixUp or CutMix augmentation."""
    if np.random.rand() < cutmix_prob:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        b, c, h, w = images.size()
        rand_index = torch.randperm(b, device=images.device)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cut_w = int(w * np.sqrt(1 - lam))
        cut_h = int(h * np.sqrt(1 - lam))
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        return images, (labels, labels[rand_index], lam)
    else:
        # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        b = images.size(0)
        rand_index = torch.randperm(b, device=images.device)
        mixed = lam * images + (1 - lam) * images[rand_index]
        
        return mixed, (labels, labels[rand_index], lam)

