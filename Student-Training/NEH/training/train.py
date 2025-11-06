"""Training loop for knowledge distillation."""

import torch
from tqdm import tqdm


def train_one_epoch(student_model, teacher_model, dataloader, optimizer, 
                    distill_criterion, device, accumulation_steps=1, 
                    log_interval=10, scaler=None):
    """Train student for one epoch with knowledge distillation"""
    student_model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Training')):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_logits = student_model(images)
                loss, hard_loss, soft_loss = distill_criterion(
                    student_logits, teacher_logits, labels
                )
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            student_logits = student_model(images)
            loss, hard_loss, soft_loss = distill_criterion(
                student_logits, teacher_logits, labels
            )
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()
        
        with torch.no_grad():
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if log_interval and (batch_idx + 1) % (log_interval * accumulation_steps) == 0:
            acc = 100.0 * correct / max(1, total)
            tqdm.write(f"Iter {batch_idx+1}/{len(dataloader)} - Loss: {loss.item()*accumulation_steps:.4f} "
                      f"(Hard: {hard_loss.item():.4f}, Soft: {soft_loss.item():.4f}) - Acc: {acc:.2f}%")
    
    epoch_loss = total_loss / len(dataloader)
    epoch_hard_loss = total_hard_loss / len(dataloader)
    epoch_soft_loss = total_soft_loss / len(dataloader)
    epoch_acc = correct / max(1, total)
    
    return epoch_loss, epoch_hard_loss, epoch_soft_loss, epoch_acc

