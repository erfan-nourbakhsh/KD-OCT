"""Data loader creation utilities."""

from torch.utils.data import DataLoader
from .dataset import RetinalOCTDataset
from .augmentation import DataAugmentation


def create_data_loaders(config, train_df, val_df, test_df):
    """Create data loaders"""
    train_transform = DataAugmentation(
        config.image_size, 
        is_training=True, 
        use_randaugment=config.use_randaugment
    )
    val_transform = DataAugmentation(config.image_size, is_training=False)
    
    train_ds = RetinalOCTDataset(train_df, config.data_root, train_transform)
    val_ds = RetinalOCTDataset(val_df, config.data_root, val_transform)
    test_ds = RetinalOCTDataset(test_df, config.data_root, val_transform)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

