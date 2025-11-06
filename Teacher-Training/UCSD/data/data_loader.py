"""Data loader creation utilities."""

import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from .dataset import FolderDataset
from .augmentation import AdvancedDataAugmentation, TTATransform


def create_data_loaders(config):
    """Create train, validation, and test data loaders from folder structure."""
    print("DEBUG: Loading full training dataset...")
    # Load full training dataset
    full_train_transform = AdvancedDataAugmentation(config.image_size, is_training=False)
    full_train_dataset = FolderDataset(config.train_dir, full_train_transform)
    
    print("DEBUG: Splitting into train/val...")
    # Split training data into train and validation
    train_size = int((1 - config.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Create stratified split
    train_indices, val_indices = [], []
    labels = [full_train_dataset.samples[i][1] for i in range(len(full_train_dataset))]
    
    print("DEBUG: Organizing class indices...")
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    print("DEBUG: Performing stratified split...")
    # Split each class proportionally
    torch.manual_seed(config.random_state)
    for class_idx, indices in class_indices.items():
        n_val = int(len(indices) * config.val_split)
        perm = torch.randperm(len(indices)).tolist()
        val_indices.extend([indices[i] for i in perm[:n_val]])
        train_indices.extend([indices[i] for i in perm[n_val:]])
    
    print("DEBUG: Creating transform objects...")
    # Create datasets with proper transforms
    train_transform = AdvancedDataAugmentation(config.image_size, is_training=True, 
                                               use_randaugment=config.use_randaugment)
    val_transform = AdvancedDataAugmentation(config.image_size, is_training=False)
    
    print("DEBUG: Creating subset datasets...")
    train_dataset = torch.utils.data.Subset(
        FolderDataset(config.train_dir, train_transform), 
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        FolderDataset(config.train_dir, val_transform), 
        val_indices
    )
    
    print("DEBUG: Creating test dataset...")
    # Test dataset
    if config.use_tta:
        test_transform = TTATransform(config.image_size)
        test_dataset = FolderDataset(config.test_dir, test_transform, use_tta=True)
    else:
        test_transform = AdvancedDataAugmentation(config.image_size, is_training=False)
        test_dataset = FolderDataset(config.test_dir, test_transform)
    
    print("DEBUG: Creating data loaders...")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                           num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            num_workers=config.num_workers, pin_memory=True)
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("DEBUG: Data loaders created successfully!")
    
    return train_loader, val_loader, test_loader, full_train_dataset.classes

