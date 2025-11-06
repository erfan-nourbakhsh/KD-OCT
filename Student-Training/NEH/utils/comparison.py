"""Comparison utilities for teacher vs student performance."""

import os
import pandas as pd

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


def compare_with_teacher(config):
    """Compare student with teacher performance"""
    print("\n" + "="*60)
    print("TEACHER vs STUDENT COMPARISON")
    print("="*60)
    
    # Load student results
    student_results_path = os.path.join(config.output_dir, 'training_summary.csv')
    if not os.path.exists(student_results_path):
        print("Student results not found. Run training first.")
        return
    
    student_results = pd.read_csv(student_results_path)
    student_acc = student_results['mean_test_acc'].values[0]
    student_ensemble = student_results['ensemble_accuracy'].values[0]
    
    # Try to load teacher results
    teacher_results_path = os.path.join(config.teacher_model_dir, 'cv_results.csv')
    if os.path.exists(teacher_results_path):
        teacher_results = pd.read_csv(teacher_results_path)
        teacher_acc = teacher_results['test_accuracy'].mean()
        
        print(f"Teacher Model:")
        print(f"  Mean Test Accuracy: {teacher_acc:.4f}")
        
        print(f"\nStudent Model:")
        print(f"  Mean Test Accuracy: {student_acc:.4f}")
        print(f"  Ensemble Accuracy: {student_ensemble:.4f}")
        
        print(f"\nAccuracy Gap:")
        print(f"  Student vs Teacher (mean): {(teacher_acc - student_acc)*100:.2f}%")
        print(f"  Student Ensemble vs Teacher (mean): {(teacher_acc - student_ensemble)*100:.2f}%")
        
        # Load teacher and student models for parameter count
        if TIMM_AVAILABLE:
            from models.model_builder import build_student_model
            
            teacher_model = timm.create_model(
                config.teacher_backbone,
                pretrained=False,
                num_classes=config.num_classes,
            )
            teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
            
            student_model = build_student_model(config)
            student_params = sum(p.numel() for p in student_model.parameters()) / 1e6
            
            print(f"\nModel Size:")
            print(f"  Teacher Parameters: {teacher_params:.2f}M")
            print(f"  Student Parameters: {student_params:.2f}M")
            print(f"  Parameter Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
            print(f"  Compression Ratio: {teacher_params/student_params:.1f}x")
    else:
        print("Teacher results not found. Cannot compare.")

