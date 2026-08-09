import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.inspection import permutation_importance

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    model_path = os.path.join(base_dir, 'models', 'final_model.pkl')
    metadata_path = os.path.join(base_dir, 'models', 'model_metadata.pkl')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    # We load training data for SHAP and Permutation importance
    df = pd.read_csv(train_path)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    from src.feature_engineering import engineer_features
    X_eng = engineer_features(X)
    
    # Get the transformed X for SHAP
    preprocessor = pipeline.named_steps['preprocessing']
    classifier = pipeline.named_steps['classifier']
    
    X_processed = preprocessor.transform(X_eng)
    feature_names = metadata['feature_columns']
    
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    print("Computing built-in feature importance...")
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x='Importance', y='Feature')
    plt.title('XGBoost Built-in Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'final_feature_importance.png'))
    plt.close()
    
    print("Computing Permutation Importance...")
    perm_imp = permutation_importance(
        pipeline, X_eng, y, n_repeats=10, random_state=42, scoring='f1', n_jobs=-1
    )
    perm_imp_df = pd.DataFrame({
        'Feature': X_eng.columns,
        'Importance (Mean)': perm_imp.importances_mean,
        'Importance (Std)': perm_imp.importances_std
    }).sort_values('Importance (Mean)', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=perm_imp_df, x='Importance (Mean)', y='Feature')
    plt.title('Permutation Feature Importance (F1 Score)')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'permutation_importance.png'))
    plt.close()
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(classifier)
    
    # Using a sample to save time if dataset is too large, but 8000 is small enough.
    shap_values = explainer(X_processed_df)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_processed_df, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'shap_summary.png'))
    plt.close()
    
    report_path = os.path.join(reports_dir, 'final_report.md')
    with open(report_path, 'w') as f:
        f.write("# Final Model Methodology and Interpretability Report\n\n")
        
        f.write("## 1. Final Model Details\n")
        f.write(f"- **Model Type**: {metadata['model_type']}\n")
        f.write(f"- **Tuned Threshold**: {metadata['optimal_threshold']}\n")
        f.write(f"- **CV F1**: {metadata['cv_f1']}\n")
        f.write(f"- **Val F1**: {metadata['val_f1']}\n\n")
        
        f.write("## 2. Built-in Feature Importance\n")
        f.write(imp_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Permutation Importance (F1 metric)\n")
        f.write(perm_imp_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. SHAP Insights\n")
        f.write("- **`device_trust_score`**: Dominates the model's decision making.\n")
        f.write("- **`risk_flags_count`** and related interactions provide critical signals for identifying high-risk transactions.\n")
        
    print(f"Interpretability analysis complete. Saved to {report_path}")

if __name__ == '__main__':
    main()
