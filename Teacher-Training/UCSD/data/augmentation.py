"""Data augmentation transforms."""

import numpy as np
import torchvision.transforms as transforms


class RandAugmentTransform:
    """RandAugment for medical images - more aggressive augmentation."""
    
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        
    def __call__(self, img):
        ops = [
            lambda img: transforms.functional.adjust_brightness(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.adjust_contrast(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.adjust_saturation(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.adjust_sharpness(img, 1 + (self.m/30) * np.random.choice([-1, 1])),
            lambda img: transforms.functional.rotate(img, angle=(self.m/30) * 30 * np.random.choice([-1, 1])),
            lambda img: transforms.functional.affine(img, angle=0, translate=(self.m/100, self.m/100), scale=1.0, shear=0),
        ]
        
        selected_ops = np.random.choice(ops, self.n, replace=False)
        for op in selected_ops:
            try:
                img = op(img)
            except:
                pass
        return img


class AdvancedDataAugmentation:
    """Enhanced data augmentation with RandAugment and stronger transforms."""

    def __init__(self, image_size=224, is_training=True, use_randaugment=True):
        self.is_training = is_training
        self.image_size = image_size

        if is_training:
            base_transforms = [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
            ]
            
            if use_randaugment:
                base_transforms.append(RandAugmentTransform(n=2, m=9))
            
            base_transforms.extend([
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, shear=15, scale=(0.85, 1.15)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
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


class TTATransform:
    """Test-Time Augmentation transforms."""
    
    def __init__(self, image_size=224):
        self.transforms = [
            # Original
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Vertical flip
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Center crop
            transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]
    
    def __call__(self, image):
        return [transform(image) for transform in self.transforms]

