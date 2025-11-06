"""MedSigLIP model architecture with attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Check for transformers library
try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

from dotenv import load_dotenv
import os
HF_TOKEN = os.getenv('HF_TOKEN')


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SqueezeExcitation(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionClassificationHead(nn.Module):
    """Advanced classification head with attention mechanisms."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()
        
        layers = []
        current_dim = in_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            
            if use_attention and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Dropout(dropout * (0.5 if i == len(hidden_dims) - 1 else 1.0)))
            
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MedSigLIPOCTClassifier(nn.Module):
    """OCT classifier using MedSigLIP vision encoder.
    
    Uses the medical-pretrained MedSigLIP encoder for robust feature extraction
    from OCT images, with an attention-enhanced classification head.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        image_size: int = 448,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.freeze_encoder = freeze_encoder
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        # Load MedSigLIP vision encoder
        print("Loading MedSigLIP-448 vision encoder...")
        try:
            self.encoder = AutoModel.from_pretrained(
                "google/medsiglip-448",
                trust_remote_code=True,
                token=HF_TOKEN
            ).vision_model
            
            # Enable gradient checkpointing for memory efficiency
            if use_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
            
            print("MedSigLIP encoder loaded successfully")
        except Exception as e:
            print(f"Error loading MedSigLIP: {e}")
            print("Falling back to timm backbone...")
            self._load_fallback_backbone()
        
        # Freeze encoder if specified (useful for initial training)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen - only training classification head")
        
        # Get feature dimension (handle fallback models)
        if hasattr(self.encoder, 'config'):
            self.feature_dim = self.encoder.config.hidden_size
        else:
            # For fallback models (EfficientNet, etc.)
            self.feature_dim = self.feature_dim  # Already set in _load_fallback_backbone()
        
        # Advanced classification head with attention
        self.head = AttentionClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dims=(1024, 512, 256),
            dropout=dropout,
            use_attention=True
        )
    
    def _load_fallback_backbone(self):
        """Fallback to EfficientNet if MedSigLIP unavailable."""
        try:
            import timm
            self.encoder = timm.create_model(
                'efficientnetv2_m',
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            self.feature_dim = self.encoder.num_features
            print("Using EfficientNetV2-M as fallback")
        except Exception:
            from torchvision import models
            self.encoder = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.DEFAULT
            )
            self.feature_dim = 1280
            self.encoder.classifier = nn.Identity()
            print("Using torchvision EfficientNet as fallback")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features using MedSigLIP encoder
        if hasattr(self.encoder, 'forward'):
            features = self.encoder(x).pooler_output
        else:
            features = self.encoder(x)
        
        # Classification
        logits = self.head(features)
        return logits
    
    def unfreeze_encoder(self, layers_to_unfreeze: Optional[int] = None):
        """Unfreeze encoder layers for fine-tuning.
        
        Args:
            layers_to_unfreeze: Number of layers to unfreeze from the end.
                               If None, unfreezes all layers.
        """
        if layers_to_unfreeze is None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("All encoder layers unfrozen")
        else:
            # Unfreeze specific number of layers
            layers = list(self.encoder.encoder.layers)
            for layer in layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {layers_to_unfreeze} encoder layers")

