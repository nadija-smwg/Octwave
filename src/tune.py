import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, f1_score

# Import pipelines and models
from src.preprocessing import split_data, build_full_pipeline, get_cv_strategy
from src.train import get_models
from src.evaluate import compute_classification_metrics

def get_param_grids(scale_pos_weight):
    """Return dictionary of parameter search spaces for top models."""
    grids = {
        'LightGBM': {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [-1, 3, 5, 7, 10],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__num_leaves': [15, 31, 50, 63],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_child_samples': [5, 10, 20, 30],
            'classifier__reg_alpha': [0, 0.01, 0.1, 1.0],
            'classifier__reg_lambda': [0, 0.01, 0.1, 1.0],
            'classifier__is_unbalance': [True],
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [3, 4, 5, 6, 7, 8],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_child_weight': [1, 3, 5, 7],
            'classifier__gamma': [0, 0.1, 0.2, 0.5],
            'classifier__reg_alpha': [0, 0.01, 0.1, 1.0],
            'classifier__reg_lambda': [0.5, 1.0, 2.0, 5.0],
            'classifier__scale_pos_weight': [scale_pos_weight * 0.5, scale_pos_weight, scale_pos_weight * 1.5],
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', 0.5],
        }
    }
    return grids

def tune_model(pipeline, param_grid, X_train, y_train, cv):
    """Run RandomizedSearchCV to find best hyperparameters."""
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=50,  # Reduced for speed in example, 80-100 optimal
        cv=cv,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        refit=True
    )
    
    search.fit(X_train, y_train)
    return search

def optimize_threshold(y_true, y_proba):
    """Find the optimal threshold to maximize F1 score."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_csv(train_path)
    X_train, X_val, y_train, y_val = split_data(df)
    
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # We will tune LightGBM and XGBoost
    models_to_tune = ['LightGBM', 'XGBoost']
    
    base_models = get_models(scale_pos_weight)
    param_grids = get_param_grids(scale_pos_weight)
    cv = get_cv_strategy()
    
    tuning_results = []
    
    for name in models_to_tune:
        print(f"\\nTuning {name}...")
        pipeline = build_full_pipeline(model=base_models[name], use_smote=False)
        
        # In XGBoost, if early stopping or other parameters cause issues, ensure grid matches
        search = tune_model(pipeline, param_grids[name], X_train, y_train, cv)
        
        best_model = search.best_estimator_
        
        # Evaluate on Validation
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        
        val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)
        
        # Optimize Threshold
        opt_thresh, opt_f1 = optimize_threshold(y_val, y_val_proba)
        
        # Train F1 vs CV F1
        cv_res = pd.DataFrame(search.cv_results_)
        best_idx = search.best_index_
        train_f1 = cv_res.loc[best_idx, 'mean_train_score']
        cv_f1 = cv_res.loc[best_idx, 'mean_test_score']
        
        res = {
            'Model': name,
            'Best Params': str(search.best_params_),
            'Train F1': train_f1,
            'CV F1': cv_f1,
            'Val F1 (Default 0.5)': val_metrics['F1'],
            'Val ROC-AUC': val_metrics['ROC-AUC'],
            'Optimal Threshold': opt_thresh,
            'Val F1 (Tuned)': opt_f1
        }
        tuning_results.append(res)
        
        print(f"Best CV F1: {cv_f1:.4f}")
        print(f"Validation F1 (Tuned): {opt_f1:.4f} at threshold {opt_thresh:.2f}")

    results_df = pd.DataFrame(tuning_results)
    
    report_path = os.path.join(reports_dir, 'tuning_results.md')
    with open(report_path, 'w') as f:
        f.write("# Hyperparameter Tuning Results\n\n")
        
        summary_df = results_df.drop(columns=['Best Params'])
        f.write("## Performance Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        
        f.write("\n\n## Best Parameters\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"### {row['Model']}\n")
            f.write(f"```python\n{row['Best Params']}\n```\n\n")
            
    print(f"\\nTuning complete. Results saved to {report_path}")

if __name__ == '__main__':
    main()
