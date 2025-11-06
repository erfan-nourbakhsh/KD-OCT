"""Ensemble prediction utilities."""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from data import FolderDataset, DataAugmentation
from models import build_student_model


def compute_ensemble_predictions(config, fold_models, class_names, device):
    """Compute ensemble predictions by averaging predictions from all fold models."""
    print("\n" + "="*60)
    print("Computing Ensemble Predictions")
    print("="*60)
    
    test_transform = DataAugmentation(config.image_size, is_training=False)
    test_dataset = FolderDataset(config.test_dir, test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        pin_memory=True
    )
    
    # Load all models
    models = []
    for state_dict in fold_models:
        model = build_student_model(config).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Ensemble Prediction'):
            images = images.to(device)
            
            # Average predictions across all models
            ensemble_logits = []
            for model in models:
                logits = model(images)
                ensemble_logits.append(F.softmax(logits, dim=1))
            
            # Average softmax outputs
            avg_probs = torch.stack(ensemble_logits).mean(dim=0)
            _, predicted = torch.max(avg_probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    accuracy = correct / len(all_labels)
    
    # Generate ensemble report
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\nEnsemble Classification Report:\n{report_text}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Student Model Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'student_ensemble_confusion_matrix.png'), dpi=300)
    plt.close()
    
    return accuracy

