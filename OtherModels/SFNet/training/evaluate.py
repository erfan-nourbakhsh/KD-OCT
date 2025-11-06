"""Evaluation utilities."""

import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc='Evaluation'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return (
        total_loss / len(loader),
        correct / max(1, total),
        all_preds,
        all_labels
    )

