"""Main entry point for SF-Net OCT Classification Training."""

import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from config import SFNetTrainingConfig
from workflow import SFNetTrainer


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("SF-Net OCT Classification Training")
    print("="*60 + "\n")
    
    # Create config
    config = SFNetTrainingConfig()
    
    print("Configuration:")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Weight Decay: {config.weight_decay}")
    print(f"  Using AMP: {config.use_amp}")
    print(f"  Using Class Weights: {config.use_class_weights}")
    
    # Create trainer
    trainer = SFNetTrainer(config)
    
    # Run cross-validation
    start_time = time.time()
    results = trainer.run_cv()
    end_time = time.time()
    
    # Print timing
    total_hours = (end_time - start_time) / 3600
    print(f"\nTotal training time: {total_hours:.2f} hours")
    print(f"Average time per fold: {total_hours / config.n_folds:.2f} hours")


if __name__ == '__main__':
    main()

