"""Training loop implementation."""

import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, 
                    mixup_fn=None, log_interval=10, accumulation_steps=1, scaler=None):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Training')):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, labels)
        else:
            targets = labels

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            if mixup_fn is None:
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                if targets.ndim == 2:
                    hard_targets = torch.argmax(targets, dim=1)
                    total += hard_targets.size(0)
                    correct += (predicted == hard_targets).sum().item()
                else:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        if log_interval and (batch_idx + 1) % (log_interval * accumulation_steps) == 0:
            acc = 100.0 * correct / max(1, total)
            tqdm.write(f"Iter {batch_idx+1}/{len(dataloader)} - Loss: {loss.item()*accumulation_steps:.4f} - Acc: {acc:.2f}%")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc

