"""Data augmentation transforms."""

import numpy as np
import torchvision.transforms as transforms


class RandAugmentTransform:
    """Lighter RandAugment for student training"""
    
    def __init__(self, n=2, m=7):  # Reduced magnitude
        self.n = n
        self.m = m
        
    def __call__(self, img):
        ops = [
            lambda img: transforms.functional.adjust_brightness(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.adjust_contrast(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.rotate(img, angle=(self.m/30) * 20 * np.random.choice([-1, 1])),
            lambda img: transforms.functional.affine(img, angle=0, translate=(self.m/100, self.m/100), scale=1.0, shear=0),
        ]
        
        selected_ops = np.random.choice(ops, min(self.n, len(ops)), replace=False)
        for op in selected_ops:
            try:
                img = op(img)
            except:
                pass
        return img


class DataAugmentation:
    """Data augmentation for student model"""
    
    def __init__(self, image_size=224, is_training=True, use_randaugment=True):
        self.is_training = is_training
        self.image_size = image_size

        if is_training:
            base_transforms = [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
            ]
            
            if use_randaugment:
                base_transforms.append(RandAugmentTransform(n=2, m=7))
            
            base_transforms.extend([
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self.transform = transforms.Compose(base_transforms)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, image):
        return self.transform(image)

