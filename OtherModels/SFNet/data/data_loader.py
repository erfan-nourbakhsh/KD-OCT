"""Data loader creation utilities."""

from torch.utils.data import DataLoader
from .dataset import OCTDatasetSFNet


def create_data_loaders(config, train_df, val_df, test_df):
    """Create data loaders for training, validation, and testing."""
    train_ds = OCTDatasetSFNet(train_df, config.data_root, config.image_size, is_training=True)
    val_ds = OCTDatasetSFNet(val_df, config.data_root, config.image_size, is_training=False)
    test_ds = OCTDatasetSFNet(test_df, config.data_root, config.image_size, is_training=False)
    
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
        batch_size=config.batch_size * 2,
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

