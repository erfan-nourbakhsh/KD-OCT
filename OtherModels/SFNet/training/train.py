"""Training loop for SF-Net."""

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        
        if config.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / max(1, total):.2f}%"
        )
    
    return total_loss / len(loader), correct / max(1, total)

