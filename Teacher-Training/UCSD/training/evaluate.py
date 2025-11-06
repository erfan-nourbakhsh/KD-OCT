"""Evaluation utilities."""

import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_tta=False):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_data in tqdm(dataloader, desc='Evaluation'):
        if use_tta:
            # TTA: batch_data contains list of augmented images
            images_list, labels = batch_data
            labels = labels.to(device, non_blocking=True)
            
            # Average predictions across all augmentations
            tta_logits = []
            for images in images_list:
                images = torch.stack([img for img in images]).to(device, non_blocking=True)
                logits = model(images)
                tta_logits.append(logits)
            
            # Average logits
            logits = torch.stack(tta_logits).mean(dim=0)
            loss = criterion(logits, labels)
        else:
            images, labels = batch_data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc, all_preds, all_labels

