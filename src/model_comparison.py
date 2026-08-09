import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import build_full_pipeline

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    results_dir = os.path.join(base_dir, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    # 1. Dataset Summary
    print("Generating dataset summary...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    dataset_summary = pd.DataFrame([{
        'Dataset': 'Processed Dataset',
        'Rows': len(df),
        'Features': X.shape[1],
        'Target': 'is_fraud',
        'Classes': 2,
        'Train Samples': len(X_train),
        'Test Samples': len(X_test)
    }])
    dataset_summary.to_csv(os.path.join(results_dir, 'dataset_summary.csv'), index=False)

    # 2. Define Models and Parameter Grids
    # Using 'classifier__' prefix because the model is named 'classifier' in the pipeline
    models_info = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'liblinear']
            },
            'search_type': 'grid'
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'param_grid': {
                'classifier__max_depth': [None, 5, 10, 15],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'search_type': 'grid'
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20]
            },
            'search_type': 'random',
            'n_iter': 4
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5]
            },
            'search_type': 'random',
            'n_iter': 4
        },
        'SVM': {
            'model': SVC(class_weight='balanced', probability=True, random_state=42),
            'param_grid': {
                'classifier__C': [0.1, 1],
                'classifier__kernel': ['linear', 'rbf']
            },
            'search_type': 'random',
            'n_iter': 3
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'param_grid': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance']
            },
            'search_type': 'grid'
        },
        'XGBoost': {
            'model': XGBClassifier(scale_pos_weight=97, eval_metric='logloss', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            },
            'search_type': 'random',
            'n_iter': 4
        }
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_list = []
    hyperparams_list = []
    
    best_pipelines = {}

    print("Starting model training and tuning...")
    for model_name, info in models_info.items():
        print(f"\nTraining {model_name}...")
        pipeline = build_full_pipeline(model=info['model'], use_smote=False)
        
        start_time = time.time()
        
        if info['search_type'] == 'grid':
            search = GridSearchCV(pipeline, info['param_grid'], cv=cv_strategy, scoring='f1', n_jobs=1)
        else:
            search = RandomizedSearchCV(pipeline, info['param_grid'], n_iter=info.get('n_iter', 5), 
                                        cv=cv_strategy, scoring='f1', random_state=42, n_jobs=1)
            
        search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        best_model = search.best_estimator_
        best_pipelines[model_name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = np.nan
        
        cv_score = search.best_score_
        best_params_str = str({k.replace('classifier__', ''): v for k, v in search.best_params_.items()})
        
        results_list.append({
            'Model': model_name,
            'Dataset': 'Processed',
            'Features': X_train.shape[1],
            'CV Score': cv_score,
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time': round(train_time, 2),
            'Best Parameters': best_params_str,
            'Score': f1 # Using F1 as primary metric for sorting
        })
        
        for k, v in search.best_params_.items():
            param_name = k.replace('classifier__', '')
            search_vals = str(info['param_grid'][k])
            hyperparams_list.append({
                'Model': model_name,
                'Parameter': param_name,
                'Search Values': search_vals,
                'Best Value': str(v)
            })

    # Sort results
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    results_df['Rank'] = results_df.index + 1
    
    # Save Model Comparison
    cols = ['Rank', 'Model', 'Dataset', 'Features', 'CV Score', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Training Time', 'Best Parameters']
    model_comparison_df = results_df[cols]
    model_comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    # Save Hyperparameters
    hp_df = pd.DataFrame(hyperparams_list)
    hp_df.to_csv(os.path.join(results_dir, 'hyperparameters.csv'), index=False)
    
    # Save Final Ranking
    ranking_df = results_df[['Rank', 'Model']].copy()
    ranking_df['Primary Metric'] = 'F1'
    ranking_df['Score'] = results_df['F1 Score']
    ranking_df['Selected'] = ['YES' if r == 1 else 'NO' for r in ranking_df['Rank']]
    ranking_df.to_csv(os.path.join(results_dir, 'model_ranking.csv'), index=False)
    
    # Plotting
    print("Generating plots...")
    sns.set_theme(style="whitegrid")
    
    # Test Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test Accuracy', y='Model', data=results_df, hue='Model', legend=False)
    plt.title('Test Accuracy Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.close()
    
    # F1 Score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='F1 Score', y='Model', data=results_df, hue='Model', legend=False)
    plt.title('F1 Score Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_comparison.png'))
    plt.close()
    
    # ROC-AUC
    plt.figure(figsize=(10, 6))
    sns.barplot(x='ROC-AUC', y='Model', data=results_df, hue='Model', legend=False)
    plt.title('ROC-AUC Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_auc_comparison.png'))
    plt.close()
    
    # Best Model Selection
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = best_pipelines[best_model_name]
    
    print(f"\nSaving Best Model: {best_model_name}")
    joblib.dump(best_pipeline, os.path.join(models_dir, 'best_model.pkl'))
    
    metadata = {
        'model_name': best_model_name,
        'f1_score': best_row['F1 Score'],
        'roc_auc': best_row['ROC-AUC'],
        'best_parameters': best_row['Best Parameters'],
        'training_time': best_row['Training Time']
    }
    with open(os.path.join(models_dir, 'best_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\n" + "="*50)
    print("MULTI-MODEL COMPARISON COMPLETE")
    print("="*50)
    print("\nMODEL RANKING (Top 3):")
    print(ranking_df.head(3).to_string(index=False))
    print("\nCONCLUSION:")
    print(f"* The best performing model was {best_model_name}.")
    print(f"* It achieved an F1-score of {best_row['F1 Score']:.4f} and a Test Accuracy of {best_row['Test Accuracy']:.4f}.")
    print(f"* Its optimal hyperparameters were: {best_row['Best Parameters']}")
    print(f"* It was selected over the others because it had the highest F1-score, which is critical for highly imbalanced fraud detection datasets.")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
