"""Model building utilities."""

import torch.nn as nn

# Try to import timm
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


def build_model(config) -> nn.Module:
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


def split_head_backbone_params(model: nn.Module):
    """Return parameter groups: (head_params, backbone_params)."""
    head_modules = []
    for attr in ['head', 'fc', 'classifier', 'classif']:
        if hasattr(model, attr):
            head_modules.append(getattr(model, attr))
    if not head_modules:
        if hasattr(model, 'get_classifier'):
            try:
                head_modules.append(model.get_classifier())
            except Exception:
                pass

    head_param_ids = set()
    for m in head_modules:
        if m is None:
            continue
        for p in m.parameters(recurse=True):
            head_param_ids.add(id(p))

    head_params, backbone_params = [], []
    for p in model.parameters():
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return head_params, backbone_params


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Set backbone parameters trainable or frozen."""
    head_params, backbone_params = split_head_backbone_params(model)
    for p in backbone_params:
        p.requires_grad = trainable
    for p in head_params:
        p.requires_grad = True

