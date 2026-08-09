import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix

# Local imports
from src.preprocessing import split_data, build_full_pipeline, get_cv_strategy
from src.evaluate import (
    compute_classification_metrics, 
    plot_roc_curves, 
    plot_precision_recall_curves,
    find_optimal_threshold
)

def get_models(scale_pos_weight):
    """Return a dictionary of configured models."""
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42), # Using sample_weight during fit, handled by SMOTE if used, else not balanced here directly (XGB/LGBM preferred)
        'XGBoost': XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    }
    return models

def cross_validate_model(pipeline, X, y, cv, scoring):
    """Run cross-validation on a pipeline."""
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    return cv_results

def plot_confusion_matrices(models_dict, X_val, y_val, save_path):
    """Plot confusion matrices for all models."""
    n_models = len(models_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, (name, pipeline) in enumerate(models_dict.items(), 1):
        y_pred = pipeline.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        plt.subplot(rows, cols, i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_csv(train_path)
    X_train, X_val, y_train, y_val = split_data(df)
    
    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    models = get_models(scale_pos_weight)
    cv = get_cv_strategy()
    
    scoring = {
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }
    
    print("Training models and running cross-validation...")
    results_list = []
    fitted_pipelines = {}
    
    for name, model in models.items():
        print(f"Processing {name}...")
        pipeline = build_full_pipeline(model=model, use_smote=False) # Strategy A: Class Weights
        
        # Cross Validation
        cv_res = cross_validate_model(pipeline, X_train, y_train, cv, scoring)
        
        # Fit on full train, evaluate on val
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)
        
        # Store results
        res = {
            'Model': name,
            'CV F1 Mean': cv_res['test_f1'].mean(),
            'CV F1 Std': cv_res['test_f1'].std(),
            'Train F1 Mean': cv_res['train_f1'].mean(),
            'Val F1': val_metrics['F1'],
            'Val Precision': val_metrics['Precision'],
            'Val Recall': val_metrics['Recall'],
            'Val ROC-AUC': val_metrics['ROC-AUC']
        }
        results_list.append(res)
        
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('Val F1', ascending=False).reset_index(drop=True)
    
    # Output markdown table
    report_path = os.path.join(reports_dir, 'model_comparison.md')
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Results (Strategy A: Class Weights)\n\n")
        f.write(results_df.to_markdown(index=False))
        
    print(f"\nModel comparison saved to {report_path}")
    print(results_df)
    
    # Generate Plots
    print("\nGenerating evaluation plots...")
    plot_roc_curves(fitted_pipelines, X_val, y_val, os.path.join(figures_dir, 'roc_curves.png'))
    plot_precision_recall_curves(fitted_pipelines, X_val, y_val, os.path.join(figures_dir, 'pr_curves.png'))
    plot_confusion_matrices(fitted_pipelines, X_val, y_val, os.path.join(figures_dir, 'confusion_matrices.png'))
    
    # Threshold Tuning for the Best Model
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nPerforming Threshold Tuning for best model: {best_model_name}")
    
    best_pipeline = fitted_pipelines[best_model_name]
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    
    optimal_thresh, optimal_f1 = find_optimal_threshold(y_val, y_val_proba)
    
    print(f"Optimal Threshold: {optimal_thresh:.4f}")
    print(f"Optimal Val F1: {optimal_f1:.4f} (Base Val F1: {results_df.iloc[0]['Val F1']:.4f})")
    
    with open(report_path, 'a') as f:
        f.write(f"\n\n## Threshold Tuning ({best_model_name})\n")
        f.write(f"- Optimal Threshold: {optimal_thresh:.4f}\n")
        f.write(f"- Tuned Val F1: {optimal_f1:.4f}\n")

if __name__ == '__main__':
    main()
