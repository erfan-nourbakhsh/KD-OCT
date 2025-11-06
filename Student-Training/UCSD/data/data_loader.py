"""Data loader creation utilities."""

import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from .dataset import FolderDataset
from .augmentation import DataAugmentation


def create_stratified_splits(dataset, config):
    """Create stratified k-fold splits from dataset"""
    # Get all labels
    all_labels = [dataset.samples[i][1] for i in range(len(dataset))]
    all_indices = np.arange(len(dataset))
    
    # Create stratified k-fold
    kfold = StratifiedKFold(
        n_splits=config.n_folds, 
        shuffle=True, 
        random_state=config.random_state
    )
    
    fold_splits = []
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(all_indices, all_labels)):
        # Further split train_val into train and val
        train_val_labels = [all_labels[i] for i in train_val_idx]
        
        # Stratified split for train/val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=config.val_split,
            random_state=config.random_state,
            stratify=train_val_labels
        )
        
        fold_splits.append({
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        })
    
    return fold_splits


def create_fold_loaders(config, fold_split, full_dataset):
    """Create data loaders for a specific fold"""
    train_transform = DataAugmentation(
        config.image_size, 
        is_training=True, 
        use_randaugment=config.use_randaugment
    )
    val_transform = DataAugmentation(config.image_size, is_training=False)
    
    # Create datasets with indices
    train_dataset = FolderDataset(config.train_dir, train_transform, fold_split['train'])
    val_dataset = FolderDataset(config.train_dir, val_transform, fold_split['val'])
    test_dataset = FolderDataset(config.test_dir, val_transform)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

