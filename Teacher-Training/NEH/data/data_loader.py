"""Data loader creation utilities."""

from torch.utils.data import DataLoader
from .dataset import RetinalOCTDataset
from .augmentation import AdvancedDataAugmentation, TTATransform


def create_data_loaders(config, train_df, val_df, test_df):
    """Create training, validation, and test data loaders."""
    train_t = AdvancedDataAugmentation(config.image_size, is_training=True, use_randaugment=config.use_randaugment)
    val_t = AdvancedDataAugmentation(config.image_size, is_training=False)
    
    if config.use_tta:
        test_t = TTATransform(config.image_size)
        test_use_tta = True
    else:
        test_t = AdvancedDataAugmentation(config.image_size, is_training=False)
        test_use_tta = False

    train_ds = RetinalOCTDataset(train_df, config.data_root, train_t)
    val_ds = RetinalOCTDataset(val_df, config.data_root, val_t)
    test_ds = RetinalOCTDataset(test_df, config.data_root, test_t, use_tta=test_use_tta)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                           num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                            num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

