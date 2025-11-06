"""Cross-validation workflow and fold execution."""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

from config import TrainingConfig
from models import build_model
from data import create_data_loaders
from training import (
    create_optimizer, create_scheduler, create_criteria, prepare_mixup,
    set_backbone_trainable, train_one_epoch, evaluate
)
from utils import generate_plots_and_report, compute_ensemble_predictions


def run_fold(config: TrainingConfig, fold: int, train_df, val_df, test_df):
    """Train and evaluate model for a single fold."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = create_data_loaders(config, train_df, val_df, test_df)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    label_map = {'NORMAL': 0, 'DRUSEN': 1, 'CNV': 2}
    train_labels = train_df['Label'].map(label_map).values
    cls_weights = class_weight.compute_class_weight('balanced', classes=np.array([0,1,2]), y=train_labels)

    model = build_model(config).to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    train_criterion, val_criterion = create_criteria(config, cls_weights, device)

    mixup_fn = prepare_mixup(config)
    if mixup_fn is None and config.use_mixup:
        print("Mixup/CutMix disabled (using Focal Loss or timm not available).")

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # SWA
    swa_model = None
    swa_scheduler = None
    if config.use_swa:
        swa_model = optim.swa_utils.AveragedModel(model)
        swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=config.swa_lr)

    set_backbone_trainable(model, trainable=False)

    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, f'fold_{fold+1}'))
    best_val_loss = float('inf')
    best_model_path = os.path.join(config.output_dir, f'best_model_fold_{fold+1}.pth')
    patience_counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")

        if epoch == config.freeze_backbone_epochs:
            set_backbone_trainable(model, trainable=True)
            print("Backbone unfrozen for fine-tuning.")

        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, train_criterion, device,
            mixup_fn=mixup_fn, log_interval=config.log_interval, 
            accumulation_steps=config.accumulation_steps, scaler=scaler
        )

        # SWA update
        if config.use_swa and epoch >= config.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, val_criterion, device, use_tta=False)

        writer.add_scalar('Loss/Train', tr_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', tr_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{config.epochs}] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if config.save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    writer.close()

    # Use SWA model if enabled
    if config.use_swa and swa_model is not None:
        print("Updating batch normalization for SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model = swa_model.module  # Extract the averaged model
        print("Using SWA averaged model for final evaluation")

    # Load best weights
    if config.save_best and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")

    # Test with TTA
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, val_criterion, device, use_tta=config.use_tta
    )
    generate_plots_and_report(config, fold, test_labels, test_preds, 
                              train_losses, val_losses, train_accs, val_accs)

    return {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'model_state': model.state_dict() if config.use_tta else None,
    }


def run_cross_validation(config: TrainingConfig):
    """Run k-fold cross-validation."""
    print("Starting 5-fold cross-validation with enhanced training...")

    df = pd.read_csv(config.csv_path)
    print(f"Dataset loaded: {df.shape}")
    print("Class distribution:")
    print(df['Label'].value_counts())

    df = df[df['Class'] == df['Label']]
    print(f"Images after worst-case filtering: {len(df)}")

    patient_labels = []
    for _, group in df.groupby('Patient ID'):
        patient_labels.append(group['Label'].mode()[0])

    patient_ids = df['Patient ID'].unique()
    patient_labels = np.array(patient_labels)
    print(f"Number of patients: {len(patient_ids)}")

    kfold = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    fold_results = []
    fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(patient_ids, patient_labels)):
        train_patients = patient_ids[train_idx]
        test_patients = patient_ids[test_idx]
        train_patient_labels = patient_labels[train_idx]
        train_patients_split, val_patients_split = train_test_split(
            train_patients, test_size=0.2, random_state=config.random_state,
            stratify=train_patient_labels
        )

        train_df = df[df['Patient ID'].isin(train_patients_split)]
        val_df = df[df['Patient ID'].isin(val_patients_split)]
        test_df = df[df['Patient ID'].isin(test_patients)]

        result = run_fold(config, fold, train_df, val_df, test_df)
        fold_results.append(result)
        if result['model_state'] is not None:
            fold_models.append(result['model_state'])

        print(f"Fold {fold + 1} Results:")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  Epochs Trained: {result['epochs_trained']}")

    # Ensemble predictions (if models saved)
    if len(fold_models) > 0:
        print("\n" + "="*60)
        print("Computing ensemble predictions across all folds...")
        ensemble_accuracy = compute_ensemble_predictions(config, fold_models, df, patient_ids, patient_labels)
        print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")

    # Final summary
    test_accuracies = [r['test_accuracy'] for r in fold_results]
    test_losses = [r['test_loss'] for r in fold_results]
    epochs_trained = [r['epochs_trained'] for r in fold_results]

    print("\nFINAL CROSS-VALIDATION RESULTS")
    print("="*60)
    print("Test Accuracy per fold:")
    for i, acc in enumerate(test_accuracies):
        print(f"  Fold {i+1}: {acc:.4f}")
    print(f"Mean: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")

    print("\nTest Loss per fold:")
    for i, loss in enumerate(test_losses):
        print(f"  Fold {i+1}: {loss:.4f}")
    print(f"Mean: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")

    print("\nEpochs trained per fold:")
    for i, ep in enumerate(epochs_trained):
        print(f"  Fold {i+1}: {ep}")
    print(f"Mean: {np.mean(epochs_trained):.1f} ± {np.std(epochs_trained):.1f}")

    os.makedirs(config.output_dir, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(os.path.join(config.output_dir, 'cv_results.csv'), index=False)

    # Summary plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(range(1, len(test_accuracies) + 1), test_accuracies, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.axhline(y=np.mean(test_accuracies), color='red', linestyle='--', label=f"Mean: {np.mean(test_accuracies):.4f}")
    ax1.set_title('Test Accuracy by Fold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(1, len(test_losses) + 1), test_losses, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.axhline(y=np.mean(test_losses), color='blue', linestyle='--', label=f"Mean: {np.mean(test_losses):.4f}")
    ax2.set_title('Test Loss by Fold')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'cv_summary.png'), dpi=300)
    plt.close()

    print(f"\nResults saved to: {config.output_dir}")

