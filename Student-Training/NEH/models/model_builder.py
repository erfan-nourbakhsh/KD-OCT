"""Model building for teacher and student models."""

import os
import torch

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


def build_teacher_model(config, fold):
    """Build and load the teacher model for specific fold"""
    if not TIMM_AVAILABLE:
        raise RuntimeError("timm is required for teacher model")
    
    model = timm.create_model(
        config.teacher_backbone,
        pretrained=False,
        num_classes=config.num_classes,
    )
    
    # Load trained weights for this fold
    teacher_path = os.path.join(config.teacher_model_dir, f'best_model_fold_{fold+1}.pth')
    if os.path.exists(teacher_path):
        state_dict = torch.load(teacher_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded teacher model from {teacher_path}")
    else:
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def build_student_model(config):
    """Build the student model"""
    if TIMM_AVAILABLE:
        try:
            model = timm.create_model(
                config.student_backbone,
                pretrained=True,
                num_classes=config.num_classes,
                drop_rate=config.dropout,
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Student model: {config.student_backbone}")
            print(f"Total parameters: {total_params/1e6:.2f}M")
            print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
            
            if total_params > 20e6:
                print(f"WARNING: Model has {total_params/1e6:.2f}M parameters (target: <20M)")
            
            return model
        except Exception as e:
            print(f"Error creating student model: {e}")
            raise
    else:
        raise RuntimeError("timm is required for student model")

