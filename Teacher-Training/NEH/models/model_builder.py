"""Model building utilities."""

import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


def build_model(config):
    """Create the backbone model with the correct classifier head."""
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model(
                config.backbone,
                pretrained=True,
                num_classes=config.num_classes,
                drop_rate=config.dropout,
                drop_path_rate=config.drop_path_rate,
            )
            print(f"Using timm backbone: {config.backbone}")
            return model
        except Exception as e:
            print(f"Warning: failed to create timm model '{config.backbone}': {e}")

    # Fallback to torchvision ResNet50
    from torchvision import models
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.num_classes)
    print("Using torchvision ResNet50 fallback")
    return model

