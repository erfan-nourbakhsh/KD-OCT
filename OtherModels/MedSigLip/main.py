"""Main entry point for MedSigLIP OCT Classification Training."""

import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from config import MedSigLIPConfig
from workflow import MedSigLIPTrainer


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("MedSigLIP OCT Classification Training")
    print("="*60 + "\n")
    
    # Create config
    config = MedSigLIPConfig()
    
    print("Configuration:")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Total Epochs: {config.total_epochs}")
    print(f"    - Frozen: {config.epochs_frozen}")
    print(f"    - Unfrozen: {config.epochs_unfrozen}")
    print(f"  Learning Rates:")
    print(f"    - Head: {config.learning_rate_head}")
    print(f"    - Encoder: {config.learning_rate_encoder}")
    print(f"  Using AMP: {config.use_amp}")
    print(f"  Using EMA: {config.use_ema}")
    print(f"  Using MixUp/CutMix: {config.use_mixup}")
    
    # Create trainer
    trainer = MedSigLIPTrainer(config)
    
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

