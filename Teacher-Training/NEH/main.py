"""Main entry point for Retinal OCT Classification with Cross-Validation."""

import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from config import TrainingConfig
from workflow import run_cross_validation


def main():
    """Main function to run the training workflow."""
    print("KD-OCT Fine-Tuning")
    print("=" * 50)
    config = TrainingConfig()
    
    print("\nConfiguration:")
    print(f"  - Test-Time Augmentation (TTA): {config.use_tta}")
    print(f"  - RandAugment: {config.use_randaugment}")
    print(f"  - Stochastic Weight Averaging (SWA): {config.use_swa}")
    print(f"  - Focal Loss: {config.use_focal_loss}")
    print(f"  - Gradient Accumulation: {config.accumulation_steps} steps")
    print(f"  - Effective Batch Size: {config.batch_size * config.accumulation_steps}")
    print(f"  - Image Size: {config.image_size}x{config.image_size}")
    print(f"  - Backbone: {config.backbone}")
    
    start = time.time()
    run_cross_validation(config)
    end = time.time()
    print(f"\nTotal training time: {(end - start) / 3600:.2f} hours")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

