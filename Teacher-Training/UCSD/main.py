"""Main entry point for UCSD OCT Classification Training."""

import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from config import TrainingConfig
from workflow import train_and_evaluate


def main():
    """Main training function."""
    print("KD-OCT Fine-Tuning")
    print("=" * 60)
    
    config = TrainingConfig()
    
    print("\nConfiguration:")
    print(f"  Train directory: {config.train_dir}")
    print(f"  Test directory: {config.test_dir}")
    print(f"  Validation split: {config.val_split*100:.1f}%")
    print(f"  Number of classes: {config.num_classes}")
    
    print("\nFeatures:")
    print(f"  - Test-Time Augmentation (TTA): {config.use_tta}")
    print(f"  - RandAugment: {config.use_randaugment}")
    print(f"  - Stochastic Weight Averaging (SWA): {config.use_swa}")
    print(f"  - Focal Loss: {config.use_focal_loss}")
    print(f"  - Gradient Accumulation: {config.accumulation_steps} steps")
    print(f"  - Effective Batch Size: {config.batch_size * config.accumulation_steps}")
    print(f"  - Image Size: {config.image_size}x{config.image_size}")
    print(f"  - Backbone: {config.backbone}")
    
    start = time.time()
    results = train_and_evaluate(config)
    end = time.time()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total training time: {(end - start) / 3600:.2f} hours")
    print(f"Final test accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Epochs trained: {results['epochs_trained']}")


if __name__ == '__main__':
    main()

