#!/usr/bin/env python3
"""
KD-OCT: Knowledge Distillation for OCT Image Classification
Unified runner for all models.

Usage:
    python run.py <model_name>

Available Models:
    KD-OCT-Teacher-NEH     Teacher model on NEH-UT dataset
    KD-OCT-Student-NEH     Student model on NEH-UT dataset (requires teacher)
    KD-OCT-Teacher-UCSD    Teacher model on UCSD dataset
    KD-OCT-Student-UCSD    Student model on UCSD dataset (requires teacher)
    MedSigLip              MedSigLIP medical-specific model
    SFNet                  SF-Net multi-scale fusion model
"""

import os
import sys
import subprocess
from pathlib import Path


# Model configuration
MODELS = {
    'KD-OCT-Teacher-NEH': {
        'path': 'Teacher-Training',
        'description': 'Teacher model (ConvNextV2-Large) on NEH-UT dataset',
        'requires': None,
        'check_path': 'Teacher-Training/results/best_model_fold_1.pth'
    },
    'KD-OCT-Student-NEH': {
        'path': 'Student-Training',
        'description': 'Student model (EfficientNet-B2) on NEH-UT dataset',
        'requires': 'KD-OCT-Teacher-NEH',
        'check_path': 'Teacher-Training/results/best_model_fold_1.pth'
    },
    'KD-OCT-Teacher-UCSD': {
        'path': 'Teacher-Training/UCSD',
        'description': 'Teacher model (ConvNextV2-Large) on UCSD dataset',
        'requires': None,
        'check_path': 'Teacher-Training/UCSD/results/best_model_fold_1.pth'
    },
    'KD-OCT-Student-UCSD': {
        'path': 'Student-Training/UCSD',
        'description': 'Student model (EfficientNet-B2) on UCSD dataset',
        'requires': 'KD-OCT-Teacher-UCSD',
        'check_path': 'Teacher-Training/UCSD/results/best_model_fold_1.pth'
    },
    'MedSigLip': {
        'path': 'OtherModels/MedSigLip',
        'description': 'MedSigLIP-448 medical-specific pretrained model',
        'requires': None,
        'check_path': None
    },
    'SFNet': {
        'path': 'OtherModels/SFNet',
        'description': 'SF-Net with multi-scale feature fusion',
        'requires': None,
        'check_path': None
    }
}


def print_banner():
    """Print welcome banner"""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           KD-OCT: Knowledge Distillation for OCT               ║
║         Retinal OCT Image Classification Framework             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_usage():
    """Print usage information"""
    print("\nUsage:")
    print("  python run.py <model_name>\n")
    print("Available Models:")
    print("━" * 70)
    
    for model_name, config in MODELS.items():
        status = ""
        if config['requires']:
            status = f" [Requires: {config['requires']}]"
        print(f"  {model_name:<25} {config['description']}{status}")
    
    print("\nExamples:")
    print("  python run.py KD-OCT-Teacher-NEH")
    print("  python run.py KD-OCT-Student-NEH")
    print("  python run.py MedSigLip")
    print()


def check_dependency(model_name, model_config):
    """Check if required dependency (teacher model) is trained"""
    if not model_config['requires']:
        return True
    
    check_path = model_config['check_path']
    if not check_path:
        return True
    
    full_path = Path(__file__).parent / check_path
    
    if not full_path.exists():
        print(f"\n❌ ERROR: Required teacher model not found!")
        print(f"\nThe model '{model_name}' requires '{model_config['requires']}' to be trained first.")
        print(f"\nMissing file: {check_path}")
        print(f"\nPlease run the teacher model first:")
        print(f"  python run.py {model_config['requires']}")
        return False
    
    return True


def check_all_folds_trained(model_name, model_config):
    """Check if all 5 folds of teacher model are trained (for student models)"""
    if not model_config['requires']:
        return True
    
    # Extract base path from check_path
    if model_config['check_path']:
        base_dir = Path(__file__).parent / Path(model_config['check_path']).parent
        
        missing_folds = []
        for fold in range(1, 6):
            fold_path = base_dir / f'best_model_fold_{fold}.pth'
            if not fold_path.exists():
                missing_folds.append(fold)
        
        if missing_folds:
            print(f"\n⚠️  WARNING: Some teacher model folds are missing!")
            print(f"\nMissing folds: {missing_folds}")
            print(f"\nThe student model training may fail if not all teacher folds are available.")
            print(f"\nRecommended action: Train all 5 folds of '{model_config['requires']}' first.")
            
            response = input("\nDo you want to continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("\nAborted.")
                return False
    
    return True


def run_model(model_name):
    """Run the specified model"""
    if model_name not in MODELS:
        print(f"\n❌ ERROR: Unknown model '{model_name}'")
        print_usage()
        return False
    
    model_config = MODELS[model_name]
    
    print(f"\n{'='*70}")
    print(f"Running: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*70}\n")
    
    # Check dependencies
    if not check_dependency(model_name, model_config):
        return False
    
    if not check_all_folds_trained(model_name, model_config):
        return False
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    model_dir = script_dir / model_config['path']
    main_file = model_dir / 'main.py'
    
    # Check if model directory and main.py exist
    if not model_dir.exists():
        print(f"\n❌ ERROR: Model directory not found: {model_dir}")
        return False
    
    if not main_file.exists():
        print(f"\n❌ ERROR: main.py not found in: {model_dir}")
        return False
    
    # Run the model
    print(f"Starting training in: {model_dir}\n")
    
    try:
        # Change to model directory and run main.py
        result = subprocess.run(
            [sys.executable, 'main.py'],
            cwd=model_dir,
            check=True
        )
        
        print(f"\n{'='*70}")
        print(f"✅ {model_name} completed successfully!")
        print(f"{'='*70}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*70}")
        print(f"❌ {model_name} failed with error code: {e.returncode}")
        print(f"{'='*70}\n")
        return False
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print(f"{'='*70}\n")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def list_models():
    """List all available models with their status"""
    print_banner()
    print("\nAvailable Models:")
    print("━" * 70)
    
    for model_name, config in MODELS.items():
        # Check if model is ready to run
        status_icon = "✓"
        status_text = "Ready"
        
        if config['requires']:
            check_path = Path(__file__).parent / config['check_path']
            if not check_path.exists():
                status_icon = "✗"
                status_text = f"Requires {config['requires']}"
        
        print(f"  {status_icon} {model_name:<25} {status_text}")
        print(f"    {config['description']}")
        print()


def main():
    """Main entry point"""
    print_banner()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("❌ ERROR: No model specified\n")
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Handle special commands
    if command in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    if command in ['-l', '--list', 'list']:
        list_models()
        sys.exit(0)
    
    # Run the specified model
    model_name = command
    success = run_model(model_name)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

