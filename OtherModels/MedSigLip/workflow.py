"""Workflow orchestration for MedSigLIP training."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from config import MedSigLIPConfig
from models import MedSigLIPOCTClassifier, ModelEMA
from data import create_data_loaders
from training import (
    setup_optimization, setup_losses,
    train_one_epoch, evaluate, tta_predict
)
from utils import generate_reports


class MedSigLIPTrainer:
    """Advanced trainer with progressive unfreezing."""
    
    def __init__(self, config: MedSigLIPConfig):
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
        """Train one fold with progressive unfreezing."""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold+1}/{self.config.n_folds}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            self.config, train_df, val_df, test_df
        )
        
        # Create model
        model = MedSigLIPOCTClassifier(
            num_classes=self.config.num_classes,
            image_size=self.config.image_size,
            dropout=self.config.dropout,
            freeze_encoder=self.config.freeze_encoder_initially,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        ).to(self.device)
        
        # Setup losses
        train_labels_np = train_df['Label'].map({
            'NORMAL': 0, 'DRUSEN': 1, 'CNV': 2
        }).values
        primary_loss, focal_loss, cls_weights = setup_losses(self.config, train_labels_np)
        
        # Setup EMA
        ema = ModelEMA(model, decay=self.config.ema_decay) if self.config.use_ema else None
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        
        # TensorBoard
        writer = SummaryWriter(
            log_dir=os.path.join(self.config.output_dir, f'fold_{fold+1}')
        )
        
        best_val_acc = 0.0
        best_path = os.path.join(
            self.config.output_dir,
            f'best_model_fold_{fold+1}.pth'
        )
        patience = 0
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        # Phase 1: Train with frozen encoder
        print("\nPhase 1: Training classification head (encoder frozen)")
        print("-" * 60)
        
        steps_per_epoch = len(train_loader)
        optimizer, scheduler = setup_optimization(
            model, self.config, steps_per_epoch, is_frozen_phase=True
        )
        
        for epoch in range(self.config.epochs_frozen):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                primary_loss, focal_loss, self.config, self.device, scaler, ema, epoch
            )
            
            eval_model = ema.ema if ema is not None else model
            va_loss, va_acc, va_preds, va_labels, va_probs = evaluate(
                eval_model, val_loader, primary_loss, self.device
            )
            
            # Log metrics
            writer.add_scalar('Loss/Train', tr_loss, epoch)
            writer.add_scalar('Loss/Val', va_loss, epoch)
            writer.add_scalar('Acc/Train', tr_acc, epoch)
            writer.add_scalar('Acc/Val', va_acc, epoch)
            
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(va_acc)
            
            print(f'Epoch [{epoch+1}/{self.config.epochs_frozen}] - '
                  f'Train: {tr_loss:.4f}/{tr_acc:.4f} | '
                  f'Val: {va_loss:.4f}/{va_acc:.4f}')
            
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                patience = 0
                if self.config.save_best:
                    torch.save(eval_model.state_dict(), best_path)
                    print(f'✓ Saved best model (acc: {va_acc:.4f})')
            else:
                patience += 1
        
        # Phase 2: Fine-tune entire model
        print("\nPhase 2: Fine-tuning entire model (encoder unfrozen)")
        print("-" * 60)
        
        # Unfreeze encoder
        model.unfreeze_encoder()
        if ema is not None:
            ema.ema.unfreeze_encoder()
        
        # Reset optimizer with different learning rates
        optimizer, scheduler = setup_optimization(
            model, self.config, steps_per_epoch, is_frozen_phase=False
        )
        
        # Reset patience
        patience = 0
        
        for epoch in range(self.config.epochs_unfrozen):
            global_epoch = self.config.epochs_frozen + epoch
            
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                primary_loss, focal_loss, self.config, self.device, scaler, ema, global_epoch
            )
            
            eval_model = ema.ema if ema is not None else model
            va_loss, va_acc, va_preds, va_labels, va_probs = evaluate(
                eval_model, val_loader, primary_loss, self.device
            )
            
            # Log metrics
            writer.add_scalar('Loss/Train', tr_loss, global_epoch)
            writer.add_scalar('Loss/Val', va_loss, global_epoch)
            writer.add_scalar('Acc/Train', tr_acc, global_epoch)
            writer.add_scalar('Acc/Val', va_acc, global_epoch)
            
            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(va_acc)
            
            print(f'Epoch [{global_epoch+1}/{self.config.total_epochs}] - '
                  f'Train: {tr_loss:.4f}/{tr_acc:.4f} | '
                  f'Val: {va_loss:.4f}/{va_acc:.4f}')
            
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                patience = 0
                if self.config.save_best:
                    torch.save(eval_model.state_dict(), best_path)
                    print(f'✓ Saved best model (acc: {va_acc:.4f})')
            else:
                patience += 1
                if patience >= self.config.patience:
                    print(f'Early stopping at epoch {global_epoch+1}')
                    break
        
        writer.close()
        
        # Load best model for testing
        if os.path.exists(best_path):
            eval_model = ema.ema if ema is not None else model
            eval_model.load_state_dict(torch.load(best_path))
            print(f'\n✓ Loaded best model for testing')
        
        # Test with TTA
        print("\nEvaluating on test set with TTA...")
        te_preds, te_labels, te_probs = tta_predict(eval_model, test_loader, self.device)
        
        # Calculate metrics
        te_acc = np.mean(np.array(te_preds) == np.array(te_labels))
        
        # Calculate additional metrics
        te_probs = np.array(te_probs)
        te_labels_onehot = np.eye(self.config.num_classes)[te_labels]
        
        try:
            te_auc = roc_auc_score(
                te_labels_onehot, te_probs, average='macro', multi_class='ovr'
            )
        except:
            te_auc = 0.0
        
        te_f1 = f1_score(te_labels, te_preds, average='macro')
        
        # Generate reports and plots
        generate_reports(
            self.config, fold, te_labels, te_preds, te_probs, history
        )
        
        results = {
            'test_accuracy': te_acc,
            'test_auc': te_auc,
            'test_f1': te_f1,
            'best_val_accuracy': best_val_acc
        }
        
        print(f"\nFold {fold+1} Results:")
        print(f"  Test Accuracy: {te_acc:.4f}")
        print(f"  Test AUC: {te_auc:.4f}")
        print(f"  Test F1: {te_f1:.4f}")
        print(f"  Best Val Accuracy: {best_val_acc:.4f}")
        
        return results
    
    def run_cv(self):
        """Run 5-fold cross-validation."""
        print('='*60)
        print('MedSigLIP OCT Classification - 5-Fold Cross-Validation')
        print('='*60)
        
        # Use same splitting strategy as original
        kfold = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(
            kfold.split(self.patient_ids, self.patient_labels)
        ):
            # Split patients
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
            print(f"  Train: {len(train_df)} images from {len(train_patients_split)} patients")
            print(f"  Val:   {len(val_df)} images from {len(val_patients_split)} patients")
            print(f"  Test:  {len(test_df)} images from {len(test_patients)} patients")
            
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
        test_aucs = [r['test_auc'] for r in results]
        test_f1s = [r['test_f1'] for r in results]
        
        print(f"\nTest Accuracies: {[f'{a:.4f}' for a in test_accs]}")
        print(f"Mean ± Std: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        
        print(f"\nTest AUCs: {[f'{a:.4f}' for a in test_aucs]}")
        print(f"Mean ± Std: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        
        print(f"\nTest F1 Scores: {[f'{f:.4f}' for f in test_f1s]}")
        print(f"Mean ± Std: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            os.path.join(self.config.output_dir, 'cv_results_summary.csv'),
            index=False
        )
        
        print(f"\n✓ Results saved to {self.config.output_dir}")
        print("="*60)

