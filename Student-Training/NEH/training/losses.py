"""Loss functions for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    - Hard label loss (cross-entropy with ground truth)
    - Soft target loss (KL divergence with teacher predictions)
    """
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft target loss
        self.beta = beta    # Weight for hard label loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft target loss with temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.beta * hard_loss + self.alpha * soft_loss
        
        return total_loss, hard_loss, soft_loss

