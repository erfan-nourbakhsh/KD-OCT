"""Augmentation pipeline for OCT images."""

import torch
import torchvision.transforms as T
from PIL import Image


class OCTAugmentations:
    """Advanced augmentation pipeline optimized for OCT images."""
    
    def __init__(self, image_size: int = 448, is_training: bool = True, use_autoaugment: bool = True):
        self.is_training = is_training
        self.image_size = image_size
        
        if is_training:
            transforms = [
                T.Resize((image_size + 64, image_size + 64)),
                T.RandomResizedCrop(
                    image_size, 
                    scale=(0.75, 1.0), 
                    ratio=(0.95, 1.05),
                    interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]
            
            # Add medical-specific augmentations
            transforms.extend([
                T.RandomApply([
                    T.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.2,
                        hue=0.1
                    )
                ], p=0.5),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.3),
                T.RandomApply([
                    T.RandomAdjustSharpness(sharpness_factor=2)
                ], p=0.3),
                T.RandomRotation(
                    degrees=15,
                    interpolation=T.InterpolationMode.BICUBIC
                ),
            ])
            
            transforms.extend([
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                # Random erasing for robustness
                T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ])
            
            self.transform = T.Compose(transforms)
        else:
            self.transform = T.Compose([
                T.Resize(
                    (image_size, image_size),
                    interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)

