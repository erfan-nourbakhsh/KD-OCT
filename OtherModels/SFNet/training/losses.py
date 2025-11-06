"""Loss functions for SF-Net training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassWeightedCrossEntropyLoss(nn.Module):
    """
    Class-weighted cross-entropy loss for imbalanced datasets.
    
    Assigns higher weights to minority classes to improve their 
    learning during training.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        if self.class_weights is not None:
            loss = F.cross_entropy(
                logits, 
                targets, 
                weight=self.class_weights,
                reduction='mean'
            )
        else:
            loss = F.cross_entropy(logits, targets, reduction='mean')
        
        return loss


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency weights for each class.
    
    Args:
        labels: All training labels
        num_classes: Number of classes
    
    Returns:
        Normalized class weights
    """
    # Count samples per class
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    
    # Inverse frequency
    weights = 1.0 / (class_counts + 1e-6)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights

