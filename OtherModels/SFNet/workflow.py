"""Workflow orchestration for SF-Net training."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import SFNetTrainingConfig
from models import SFNetOCTClassifier
from data import create_data_loaders
from training import (
    ClassWeightedCrossEntropyLoss, compute_class_weights,
    setup_training, cosine_warmup_scheduler,
    train_one_epoch, evaluate
)
from utils import generate_reports


class SFNetTrainer:
    """SF-Net trainer with cross-validation."""
    
    def __init__(self, config: SFNetTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._load_data()
    
    def _load_data(self):
        """Load and organize data for cross-validation."""
        self.df = pd.read_csv(self.config.csv_path)
        
        # Keep only rows where Class matches Label
        self.df = self.df[self.df['Class'] == self.df['Label']]
        print(f"Images after worst-case filtering: {len(self.df)}")
        
        # Extract patient-level labels
        patient_labels = []
        for _, group in self.df.groupby('Patient ID'):
            patient_labels.append(group['Label'].mode()[0])
        
        self.patient_ids = self.df['Patient ID'].unique()
        self.patient_labels = np.array(patient_labels)
        
        print(f"Loaded {len(self.df)} images from {len(self.patient_ids)} patients")
        print(f"Class distribution: {pd.Series(patient_labels).value_counts().to_dict()}")
    
    def train_fold(self, fold: int, train_df, val_df, test_df):
        """Train one fold."""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold+1}/{self.config.n_folds}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            self.config, train_df, val_df, test_df
        )
        
        # Create model
        model = SFNetOCTClassifier(
            num_classes=self.config.model_config.num_classes,
            pretrained=self.config.model_config.pretrained,
            dropout=self.config.model_config.dropout_rate,
            freeze_backbone=self.config.model_config.freeze_backbone
        ).to(self.device)
        
        # Setup training
        optimizer = setup_training(model, self.config)
        scheduler = cosine_warmup_scheduler(optimizer, self.config, len(train_loader))
        
        # Setup loss with class weights
        if self.config.use_class_weights:
            train_labels = torch.tensor(
                train_df['Label'].map({'NORMAL': 0, 'DRUSEN': 1, 'CNV': 2}).values
            )
            class_weights = compute_class_weights(train_labels, 3).to(self.device)
            criterion = ClassWeightedCrossEntropyLoss(class_weights)
        else:
            criterion = ClassWeightedCrossEntropyLoss()
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        
        # TensorBoard
        writer = SummaryWriter(
            log_dir=os.path.join(self.config.output_dir, f'fold_{fold+1}')
        )
        
        best_val_acc = 0.0
        best_path = os.path.join(self.config.output_dir, f'best_model_fold_{fold+1}.pth')
        patience = 0
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        print("\nStarting training...")
        
        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # Train
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, self.device, scaler, self.config
            )
            
            # Validate
            va_loss, va_acc, va_preds, va_labels = evaluate(
                model, val_loader, criterion, self.device
            )
            
            # Log metrics
            writer.add_scalar('Loss/Train', tr_loss, epoch)
            writer.add_scalar('Loss/Val', va_loss, epoch)
            writer.add_scalar('Acc/Train', tr_acc, epoch)
            writer.add_scalar('Acc/Val', va_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(va_acc)
            
            print(f'Train: {tr_loss:.4f}/{tr_acc:.4f} | Val: {va_loss:.4f}/{va_acc:.4f}')
            
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                patience = 0
                torch.save(model.state_dict(), best_path)
                print(f'✓ Saved best model (acc: {va_acc:.4f})')
            else:
                patience += 1
                if patience >= self.config.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        writer.close()
        
        # Load best model for testing
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))
            print(f'\n✓ Loaded best model for testing')
        
        # Test
        print("\nEvaluating on test set...")
        te_loss, te_acc, te_preds, te_labels = evaluate(
            model, test_loader, criterion, self.device
        )
        
        # Generate reports
        generate_reports(self.config, fold, te_labels, te_preds, history)
        
        results = {
            'test_accuracy': te_acc,
            'test_loss': te_loss,
            'best_val_accuracy': best_val_acc
        }
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Test Accuracy: {te_acc:.4f}")
        print(f"  Test Loss: {te_loss:.4f}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return results
    
    def run_cv(self):
        """Run 5-fold cross-validation."""
        print('='*60)
        print('SF-Net OCT Classification - 5-Fold Cross-Validation')
        print('='*60)
        
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(
            kfold.split(self.patient_ids, self.patient_labels)
        ):
            train_patients = self.patient_ids[train_idx]
            test_patients = self.patient_ids[test_idx]
            train_patient_labels = self.patient_labels[train_idx]
            
            # Further split train into train/val
            train_patients_split, val_patients_split = train_test_split(
                train_patients,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=train_patient_labels
            )
            
            # Create dataframes
            train_df = self.df[self.df['Patient ID'].isin(train_patients_split)]
            val_df = self.df[self.df['Patient ID'].isin(val_patients_split)]
            test_df = self.df[self.df['Patient ID'].isin(test_patients)]
            
            print(f"\nFold {fold+1} Data Split:")
            print(f"  Train: {len(train_df)} images")
            print(f"  Val:   {len(val_df)} images")
            print(f"  Test:  {len(test_df)} images")
            
            # Train fold
            res = self.train_fold(fold, train_df, val_df, test_df)
            results.append(res)
        
        # Final summary
        self._print_final_summary(results)
        
        return results
    
    def _print_final_summary(self, results):
        """Print final cross-validation summary."""
        print("\n" + "="*60)
        print("FINAL CROSS-VALIDATION RESULTS")
        print("="*60)
        
        test_accs = [r['test_accuracy'] for r in results]
        test_losses = [r['test_loss'] for r in results]
        
        print(f"\nTest Accuracies: {[f'{a:.4f}' for a in test_accs]}")
        print(f"Mean ± Std: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        print(f"\nTest Losses: {[f'{l:.4f}' for l in test_losses]}")
        print(f"Mean ± Std: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(self.config.output_dir, 'cv_results_summary.csv'),
            index=False
        )
        
        print(f"\n✓ Results saved to {self.config.output_dir}")
        print("="*60)

