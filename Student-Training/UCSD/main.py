"""Main entry point for Student Model Training with Knowledge Distillation."""

import os
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from config import StudentConfig
from workflow import run_cross_validation
from utils import compare_with_teacher


def main():
    """Main training function."""
    print("Knowledge Distillation - Student Model Training (Fold-by-Fold)")
    print("=" * 60)
    
    config = StudentConfig()
    
    # Check if teacher models exist for all folds
    print("\nChecking teacher models...")
    missing_folds = []
    for fold in range(config.n_folds):
        teacher_path = os.path.join(config.teacher_model_dir, f'best_model_fold_{fold+1}.pth')
        if not os.path.exists(teacher_path):
            missing_folds.append(fold + 1)
    
    if missing_folds:
        print(f"\nERROR: Teacher models not found for folds: {missing_folds}")
        print(f"Please train the teacher model first using the main training script.")
        print(f"Expected location: {config.teacher_model_dir}")
        return
    
    print("✓ All teacher models found")
    
    # Run cross-validation
    run_cross_validation(config)
    
    # Compare with teacher
    try:
        compare_with_teacher(config)
    except Exception as e:
        print(f"\nCould not compare with teacher: {e}")
    
    print("\n" + "="*60)
    print("All training complete!")
    print("="*60)


if __name__ == '__main__':
    main()

