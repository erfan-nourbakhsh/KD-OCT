"""Visualization and reporting utilities."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def generate_plots_and_report(config, fold, test_labels, test_preds, 
                              train_losses, val_losses, train_accs, val_accs):
    """Generate confusion matrix, classification report, and training curves."""
    class_names = ['NORMAL', 'DRUSEN', 'CNV']
    report_text = classification_report(test_labels, test_preds, target_names=class_names)
    print(f"\nClassification Report for Fold {fold + 1}:\n{report_text}")

    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(config.output_dir, exist_ok=True)
    plt.savefig(os.path.join(config.output_dir, f'confusion_matrix_fold_{fold+1}.png'))
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title(f'Loss Curves - Fold {fold + 1}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title(f'Accuracy Curves - Fold {fold + 1}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, f'training_curves_fold_{fold+1}.png'))
    plt.close()

