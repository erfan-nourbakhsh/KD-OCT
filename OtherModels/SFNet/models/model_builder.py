"""SF-Net model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import timm


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    
    Adaptively recalibrates channel-wise features by modeling 
    inter-channel dependencies.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: Two FC layers
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: Multiply weights with input
        return x * y.expand_as(x)


class ConvNeXtBlockWithSE(nn.Module):
    """
    ConvNeXt Block with integrated SE module.
    
    Combines ConvNeXt's design principles with channel attention 
    for enhanced feature extraction.
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SEBlock(dim, reduction=16)
        
        # Stochastic depth
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Apply SE attention
        x = self.se(x)
        
        # Residual connection with drop path
        x = shortcut + self.drop_path(x)
        return x


class MultiScaleFeatureFusion(nn.Module):
    """
    Feature Pyramid Fusion Module.
    
    Fuses features from multiple stages to combine low-level details 
    with high-level semantics.
    """
    def __init__(self, stage_channels: List[int], target_channels: int = 384):
        super().__init__()
        self.target_channels = target_channels
        
        # 1x1 convolutions for channel alignment
        self.channel_adapters = nn.ModuleList([
            nn.Conv2d(ch, target_channels, kernel_size=1, bias=False)
            for ch in stage_channels
        ])
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(target_channels)
            for _ in stage_channels
        ])
    
    def forward(self, stage_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            stage_features: List of [stage1, stage2, stage3, stage4] features
        
        Returns:
            List of fused features ready for pooling
        """
        fused_features = []
        
        # Process each stage
        for i in range(len(stage_features)):
            # Align channels
            feat = self.channel_adapters[i](stage_features[i])
            feat = self.batch_norms[i](feat)
            
            # Add upsampled features from next stage (except for last stage)
            if i < len(stage_features) - 1:
                next_feat = self.channel_adapters[i + 1](stage_features[i + 1])
                next_feat = self.batch_norms[i + 1](next_feat)
                # Upsample to match current stage spatial dimensions
                next_feat = F.interpolate(
                    next_feat, 
                    size=feat.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                feat = feat + next_feat
            
            fused_features.append(feat)
        
        return fused_features


class SFNetOCTClassifier(nn.Module):
    """
    SF-Net OCT Image Classifier.
    
    Architecture:
    1. ConvNeXt-T backbone with SE modules
    2. Multi-scale feature fusion pyramid
    3. Global pooling and concatenation
    4. Classification head
    """
    def __init__(
        self, 
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ConvNeXt-T backbone
        self.backbone = timm.create_model(
            'convnext_tiny', 
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # Get all 4 stage outputs
        )
        
        # Get stage output channels
        # ConvNeXt-T: [96, 192, 384, 768]
        self.stage_channels = self.backbone.feature_info.channels()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add SE modules to each stage (integrate into forward pass)
        self.se_modules = nn.ModuleList([
            SEBlock(ch) for ch in self.stage_channels
        ])
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            self.stage_channels, 
            target_channels=384
        )
        
        # Global pooling for each fused feature
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classification head
        # Input: 384 * 4 (four stages fused) = 1536
        fusion_dim = 384 * 4
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier layers with proper initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract multi-stage features
        stage_features = self.backbone(x)
        
        # Apply SE attention to each stage
        stage_features = [
            self.se_modules[i](feat) 
            for i, feat in enumerate(stage_features)
        ]
        
        # Multi-scale feature fusion
        fused_features = self.feature_fusion(stage_features)
        
        # Global pooling for each fused feature
        pooled_features = [
            self.global_pool(feat).flatten(1) 
            for feat in fused_features
        ]
        
        # Concatenate all pooled features
        combined = torch.cat(pooled_features, dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Returns:
            List of SE attention weights for each stage
        """
        stage_features = self.backbone(x)
        attention_maps = []
        
        for i, feat in enumerate(stage_features):
            b, c, _, _ = feat.size()
            # Get SE weights
            y = self.se_modules[i].squeeze(feat).view(b, c)
            y = self.se_modules[i].excitation(y)
            attention_maps.append(y)
        
        return attention_maps

