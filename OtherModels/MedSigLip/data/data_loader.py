"""Data loader creation utilities."""

from torch.utils.data import DataLoader
from .dataset import MedOCTDataset
from .augmentation import OCTAugmentations


def create_data_loaders(config, train_df, val_df, test_df):
    """Create data loaders for training, validation, and testing."""
    train_tf = OCTAugmentations(config.image_size, is_training=True)
    eval_tf = OCTAugmentations(config.image_size, is_training=False)
    
    train_ds = MedOCTDataset(train_df, config.data_root, train_tf)
    val_ds = MedOCTDataset(val_df, config.data_root, eval_tf)
    test_ds = MedOCTDataset(test_df, config.data_root, eval_tf)
    
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
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

