"""Workflow orchestration for student model training with knowledge distillation."""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import StudentConfig
from models import build_teacher_model, build_student_model
from data import create_stratified_splits, create_fold_loaders, FolderDataset, DataAugmentation
from training import DistillationLoss, create_optimizer, create_scheduler, train_one_epoch, evaluate
from utils import (
    plot_fold_results, plot_confusion_matrix,
    compute_ensemble_predictions, compare_with_teacher
)


def run_fold(config, fold, fold_split, full_dataset, class_names, device):
    """Train student model for one fold"""
    print(f"\n{'='*60}")
    print(f"Starting Fold {fold + 1}/{config.n_folds}")
    print(f"{'='*60}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_fold_loaders(config, fold_split, full_dataset)
    
    print(f"Train samples: {len(fold_split['train'])}")
    print(f"Validation samples: {len(fold_split['val'])}")
    print(f"Test samples: {len(fold_split['test'])}")
    
    # Build models
    print("\nBuilding models...")
    teacher_model = build_teacher_model(config, fold).to(device)
    student_model = build_student_model(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(student_model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create distillation criterion
    distill_criterion = DistillationLoss(
        temperature=config.temperature,
        alpha=config.alpha,
        beta=config.beta
    )
    
    # Evaluation criterion
    eval_criterion = nn.CrossEntropyLoss()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, f'fold_{fold+1}'))
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = os.path.join(config.output_dir, f'best_student_model_fold_{fold+1}.pth')
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\nStarting training...")
    print(f"Temperature: {config.temperature}")
    print(f"Alpha (soft): {config.alpha}, Beta (hard): {config.beta}")
    
    fold_start_time = time.time()
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Train
        train_loss, hard_loss, soft_loss, train_acc = train_one_epoch(
            student_model, teacher_model, train_loader,
            optimizer, distill_criterion, device,
            accumulation_steps=config.accumulation_steps,
            log_interval=config.log_interval,
            scaler=scaler
        )
        
        scheduler.step()
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            student_model, val_loader, eval_criterion, device
        )
        
        # Log metrics
        writer.add_scalar('Loss/Train_Total', train_loss, epoch)
        writer.add_scalar('Loss/Train_Hard', hard_loss, epoch)
        writer.add_scalar('Loss/Train_Soft', soft_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{config.epochs}]")
        print(f"  Train Loss: {train_loss:.4f} (Hard: {hard_loss:.4f}, Soft: {soft_loss:.4f})")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(student_model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    writer.close()
    
    # Training complete for this fold
    fold_end_time = time.time()
    fold_training_time = (fold_end_time - fold_start_time) / 3600
    
    print(f"\nFold {fold+1} training complete!")
    print(f"Training time: {fold_training_time:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model
    if os.path.exists(best_model_path):
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(
        student_model, test_loader, eval_criterion, device
    )
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_fold_results(train_losses, val_losses, train_accs, val_accs, fold, config.output_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, fold, config.output_dir)
    
    # Save training history
    history = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    history.to_csv(os.path.join(config.output_dir, f'training_history_fold_{fold+1}.csv'), index=False)
    
    return {
        'fold': fold + 1,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(train_losses),
        'training_time_hours': fold_training_time,
        'model_state': student_model.state_dict(),
    }


def run_cross_validation(config):
    """Run 5-fold cross-validation for student model training"""
    print("Knowledge Distillation - Student Model 5-Fold Cross-Validation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load full training dataset
    print("\nLoading training dataset...")
    temp_transform = DataAugmentation(config.image_size, is_training=False)
    full_dataset = FolderDataset(config.train_dir, temp_transform)
    class_names = full_dataset.classes
    
    print(f"\nTotal training samples: {len(full_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Create stratified splits
    print("\nCreating stratified k-fold splits...")
    fold_splits = create_stratified_splits(full_dataset, config)
    
    # Verify splits
    for i, split in enumerate(fold_splits):
        print(f"Fold {i+1}: Train={len(split['train'])}, Val={len(split['val'])}, Test={len(split['test'])}")
    
    fold_results = []
    fold_models = []
    
    start_time = time.time()
    
    # Run each fold
    for fold, fold_split in enumerate(fold_splits):
        result = run_fold(config, fold, fold_split, full_dataset, class_names, device)
        fold_results.append(result)
        fold_models.append(result['model_state'])
        
        print(f"\nFold {fold + 1} Summary:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Best Val Accuracy: {result['best_val_acc']:.4f}")
        print(f"  Epochs Trained: {result['epochs_trained']}")
        print(f"  Training Time: {result['training_time_hours']:.2f} hours")
    
    # Ensemble predictions
    print("\nComputing ensemble predictions across all folds...")
    ensemble_accuracy = compute_ensemble_predictions(config, fold_models, class_names, device)
    print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    # Total training time
    end_time = time.time()
    total_training_time = (end_time - start_time) / 3600
    
    # Save results and generate summary
    _save_results_and_summary(config, fold_results, ensemble_accuracy, total_training_time)
    
    print(f"\nAll results saved to: {config.output_dir}")
    print("\nStudent model training completed successfully!")


def _save_results_and_summary(config, fold_results, ensemble_accuracy, total_training_time):
    """Save results and generate summary statistics"""
    # Extract metrics
    test_accuracies = [r['test_accuracy'] for r in fold_results]
    test_losses = [r['test_loss'] for r in fold_results]
    val_accuracies = [r['best_val_acc'] for r in fold_results]
    epochs_trained = [r['epochs_trained'] for r in fold_results]
    training_times = [r['training_time_hours'] for r in fold_results]
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print("\nTest Accuracy per fold:")
    for i, acc in enumerate(test_accuracies):
        print(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Mean: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    
    print(f"\nEnsemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    print(f"Total training time: {total_training_time:.2f} hours")
    
    # Save fold results
    os.makedirs(config.output_dir, exist_ok=True)
    results_df = pd.DataFrame(fold_results)
    results_df = results_df.drop('model_state', axis=1)
    results_df.to_csv(os.path.join(config.output_dir, 'cv_results.csv'), index=False)
    
    # Save summary statistics
    summary = {
        'student_backbone': config.student_backbone,
        'teacher_backbone': config.teacher_backbone,
        'image_size': config.image_size,
        'temperature': config.temperature,
        'alpha': config.alpha,
        'beta': config.beta,
        'mean_test_acc': np.mean(test_accuracies),
        'std_test_acc': np.std(test_accuracies),
        'mean_val_acc': np.mean(val_accuracies),
        'std_val_acc': np.std(val_accuracies),
        'ensemble_accuracy': ensemble_accuracy,
        'total_training_time_hours': total_training_time,
        'mean_epochs': np.mean(epochs_trained),
        'n_folds': config.n_folds,
    }
    
    pd.DataFrame([summary]).to_csv(
        os.path.join(config.output_dir, 'training_summary.csv'),
        index=False
    )
    
    # Create summary plots
    _create_summary_plots(config, test_accuracies, test_losses, epochs_trained, 
                         training_times, ensemble_accuracy)
    
    # Print model info
    student_model = build_student_model(config)
    student_params = sum(p.numel() for p in student_model.parameters()) / 1e6
    
    print("\n" + "="*60)
    print("STUDENT MODEL INFO")
    print("="*60)
    print(f"Architecture: {config.student_backbone}")
    print(f"Parameters: {student_params:.2f}M")
    print(f"Image Size: {config.image_size}x{config.image_size}")
    print(f"Distillation Temperature: {config.temperature}")


def _create_summary_plots(config, test_accuracies, test_losses, epochs_trained, 
                         training_times, ensemble_accuracy):
    """Create summary visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Test Accuracy
    ax1 = axes[0, 0]
    ax1.bar(range(1, len(test_accuracies) + 1), test_accuracies, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.axhline(y=np.mean(test_accuracies), color='red', linestyle='--', 
                label=f"Mean: {np.mean(test_accuracies):.4f}")
    ax1.axhline(y=ensemble_accuracy, color='green', linestyle='--', 
                label=f"Ensemble: {ensemble_accuracy:.4f}")
    ax1.set_title('Student Model - Test Accuracy by Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test Loss
    ax2 = axes[0, 1]
    ax2.bar(range(1, len(test_losses) + 1), test_losses, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.axhline(y=np.mean(test_losses), color='blue', linestyle='--', 
                label=f"Mean: {np.mean(test_losses):.4f}")
    ax2.set_title('Student Model - Test Loss by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Epochs Trained
    ax3 = axes[1, 0]
    ax3.bar(range(1, len(epochs_trained) + 1), epochs_trained, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax3.axhline(y=np.mean(epochs_trained), color='orange', linestyle='--', 
                label=f"Mean: {np.mean(epochs_trained):.1f}")
    ax3.set_title('Student Model - Epochs Trained by Fold')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Epochs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training Time
    ax4 = axes[1, 1]
    ax4.bar(range(1, len(training_times) + 1), training_times, color='plum', alpha=0.7, edgecolor='purple')
    ax4.axhline(y=np.mean(training_times), color='red', linestyle='--', 
                label=f"Mean: {np.mean(training_times):.2f}h")
    ax4.set_title('Student Model - Training Time by Fold')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('Time (hours)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'cv_summary.png'), dpi=300)
    plt.close()

