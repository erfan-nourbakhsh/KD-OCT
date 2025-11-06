"""Evaluation utilities."""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(model, loader, primary_loss, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Eval'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            loss = primary_loss(logits, labels)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        total_loss / len(loader),
        correct / max(1, total),
        all_preds,
        all_labels,
        all_probs
    )


def tta_predict(model, loader, device):
    """Test-time augmentation predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='TTA'):
            # Original prediction
            images = images.to(device)
            logits_list = [model(images)]
            
            # Horizontal flip
            logits_list.append(model(torch.flip(images, dims=[3])))
            
            # Average predictions
            logits = torch.stack(logits_list).mean(0)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_labels, all_probs

