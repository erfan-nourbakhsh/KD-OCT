"""Workflow orchestration for UCSD OCT training."""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import class_weight

from config import TrainingConfig
from models import build_model, set_backbone_trainable
from data import create_data_loaders
from training import (
    create_optimizer, create_scheduler, create_criteria, prepare_mixup,
    train_one_epoch, evaluate
)
from utils import generate_plots_and_report


def train_and_evaluate(config: TrainingConfig):
    """Main training loop - train on all training data and evaluate on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print("DEBUG: Creating data loaders...")
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(config)
    
    print("DEBUG: Computing class weights...")
    # Compute class weights from training data
    train_labels = []
    for _, label in train_loader.dataset:
        if isinstance(label, int):
            train_labels.append(label)
    
    print("DEBUG: Calculating balanced class weights...")
    cls_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.arange(config.num_classes), 
        y=train_labels
    )
    print(f"\nClass weights: {cls_weights}")

    print("DEBUG: Building model...")
    # Build model
    model = build_model(config).to(device)
    
    print("DEBUG: Creating optimizer...")
    optimizer = create_optimizer(model, config)
    
    print("DEBUG: Creating scheduler...")
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

    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    best_val_loss = float('inf')
    best_model_path = os.path.join(config.output_dir, 'best_model.pth')
    patience_counter = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

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
        print("\nUpdating batch normalization for SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        model = swa_model.module  # Extract the averaged model
        print("Using SWA averaged model for final evaluation")

    # Load best weights
    if config.save_best and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from {best_model_path}")

    # Final test evaluation with TTA
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, val_criterion, device, use_tta=config.use_tta
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Generate plots and reports
    generate_plots_and_report(config, test_labels, test_preds, 
                              train_losses, val_losses, train_accs, val_accs, class_names)

    # Save final results
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(config.output_dir, 'training_results.csv'), index=False)
    
    print(f"\nResults saved to: {config.output_dir}")
    
    return results

