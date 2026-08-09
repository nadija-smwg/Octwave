import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

def compute_classification_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive classification metrics."""
    metrics = {
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred)
    }
    return metrics

def plot_roc_curves(models_dict, X_val, y_val, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, pipeline in models_dict.items():
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        auc = roc_auc_score(y_val, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curves(models_dict, X_val, y_val, save_path):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, pipeline in models_dict.items():
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        plt.plot(rec, prec, label=f'{name} (PR-AUC = {pr_auc:.3f})')
        
    baseline = y_val.mean()
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_optimal_threshold(y_true, y_proba):
    """Find the optimal threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Ignore zero division warnings by adding a small epsilon
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1
