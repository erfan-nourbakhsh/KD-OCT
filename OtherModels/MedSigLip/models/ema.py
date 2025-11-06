"""Exponential Moving Average for model parameters."""

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema = self._clone_model(model)
        self._set_requires_grad(self.ema, False)
    
    @torch.no_grad()
    def _clone_model(self, model: nn.Module):
        ema = type(model)(
            num_classes=model.num_classes,
            image_size=model.image_size,
            dropout=0.0,
            freeze_encoder=False,
            use_gradient_checkpointing=False
        )
        ema.load_state_dict(model.state_dict())
        ema = ema.to(next(model.parameters()).device)
        ema.eval()
        return ema
    
    @staticmethod
    def _set_requires_grad(model: nn.Module, requires_grad: bool):
        for p in model.parameters():
            p.requires_grad_(requires_grad)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        
        for k, v in self.ema.state_dict().items():
            if k in msd:
                src = msd[k].detach()
                if src.device != v.device:
                    src = src.to(v.device)
                
                if not torch.is_floating_point(v):
                    v.copy_(src)
                else:
                    v.copy_(v * d + src * (1.0 - d))

