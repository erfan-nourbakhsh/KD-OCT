"""Training loop implementation."""

import numpy as np
import torch
from tqdm import tqdm
from .utils import apply_mixup_cutmix


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    primary_loss,
    focal_loss,
    config,
    device,
    scaler,
    ema,
    epoch
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Train {epoch+1}')
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply MixUp/CutMix randomly
        if config.use_mixup and np.random.rand() < 0.5:
            images, mix = apply_mixup_cutmix(
                images, labels,
                config.mixup_alpha,
                config.cutmix_alpha,
                config.cutmix_prob
            )
        else:
            mix = None
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            logits = model(images)
            
            if mix is None:
                loss_cls = primary_loss(logits, labels)
                loss_f = focal_loss(logits, labels)
                loss = 0.7 * loss_cls + 0.3 * loss_f
            else:
                la, lb, lam = mix
                loss_a = primary_loss(logits, la)
                loss_b = primary_loss(logits, lb)
                loss = lam * loss_a + (1 - lam) * loss_b
        
        scaler.scale(loss).backward()
        
        if config.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema is not None:
            ema.update(model)
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / max(1, total):.2f}%"
        )
    
    return total_loss / len(train_loader), correct / max(1, total)

