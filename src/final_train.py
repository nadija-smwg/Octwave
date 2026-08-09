import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from xgboost import XGBClassifier

from src.preprocessing import build_full_pipeline

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_csv(train_path)
    X_full_train = df.drop(columns=['is_fraud'])
    y_full_train = df['is_fraud']
    
    # Best params from Phase 8 tuning for XGBoost
    best_params = {
        'subsample': 0.8,
        'scale_pos_weight': 97.46907216494844,
        'reg_lambda': 2.0,
        'reg_alpha': 0,
        'n_estimators': 200,
        'min_child_weight': 3,
        'max_depth': 4,
        'learning_rate': 0.2,
        'gamma': 0,
        'colsample_bytree': 1.0,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # Initialize XGBoost with best parameters
    model = XGBClassifier(**best_params)
    
    # Build full pipeline (no SMOTE since we use class weights)
    final_pipeline = build_full_pipeline(model=model, use_smote=False)
    
    print("Training final model on 100% of the training data...")
    final_pipeline.fit(X_full_train, y_full_train)
    
    # Extract feature columns from the preprocessor to save in metadata
    preprocessor = final_pipeline.named_steps['preprocessing']
    feature_columns = preprocessor.get_feature_names_out().tolist()
    # Clean up feature names
    feature_columns = [f.split('__')[-1] for f in feature_columns]
    
    # Optimal threshold from Phase 8
    optimal_threshold = 0.85
    
    metadata = {
        'model_type': 'XGBoost',
        'best_params': best_params,
        'cv_f1': 0.9897,
        'val_f1': 1.0000,
        'optimal_threshold': optimal_threshold,
        'feature_columns': feature_columns,
        'training_date': datetime.now().isoformat(),
        'random_state': 42
    }
    
    model_path = os.path.join(models_dir, 'final_model.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
    
    joblib.dump(final_pipeline, model_path)
    joblib.dump(metadata, metadata_path)
    
    print(f"Final model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == '__main__':
    main()
