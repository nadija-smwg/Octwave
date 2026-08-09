import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def get_column_lists():
    """Return lists of original numerical, categorical, and binary columns."""
    numerical_cols = [
        'amount', 'device_trust_score', 'velocity_last_24h', 
        'cardholder_age', 'transaction_hour'
    ]
    categorical_cols = ['merchant_category']
    binary_cols = ['foreign_transaction', 'location_mismatch']
    
    return numerical_cols, categorical_cols, binary_cols

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and validation sets stratifying by target."""
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_val, y_train, y_val

def get_cv_strategy(n_splits=5):
    """Return the cross-validation strategy."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def build_preprocessor():
    """Build the ColumnTransformer for the preprocessing pipeline."""
    from src.feature_engineering import get_feature_columns
    
    num_cols, cat_cols, bin_cols = get_column_lists()
    eng_cols = get_feature_columns()
    
    all_num_cols = num_cols + eng_cols['engineered_numeric']
    all_cat_cols = cat_cols + eng_cols['engineered_categorical']
    all_bin_cols = bin_cols + eng_cols['engineered_binary']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), all_cat_cols),
            ('bin', 'passthrough', all_bin_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def build_full_pipeline(model=None, use_smote=False):
    """Build the full sklearn Pipeline including feature engineering, preprocessing, SMOTE and modelling."""
    from src.feature_engineering import engineer_features
    
    fe_step = FunctionTransformer(engineer_features, validate=False)
    preprocessor = build_preprocessor()
    
    steps = [
        ('feature_engineering', fe_step),
        ('preprocessing', preprocessor)
    ]
    
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
        
    if model is not None:
        steps.append(('classifier', model))
        
    # We must use imblearn.pipeline.Pipeline to handle SMOTE correctly during fit/predict
    pipeline = ImbPipeline(steps)
    return pipeline

def verify_no_leakage(X_train, X_val):
    """Verify that there is no data leakage between train and validation sets."""
    # Check for ID overlap if ID column exists
    if 'transaction_id' in X_train.columns and 'transaction_id' in X_val.columns:
        overlap = set(X_train['transaction_id']).intersection(set(X_val['transaction_id']))
        assert len(overlap) == 0, f"Leakage detected: {len(overlap)} IDs overlap between train and val."
    else:
        # Check identical rows overlap
        # Pandas merge with indicator to find common rows
        merged = pd.merge(X_train, X_val, how='inner')
        assert len(merged) == 0, f"Leakage detected: {len(merged)} identical rows between train and val."
        
    print("Leakage verification passed: No overlapping rows between train and validation sets.")
    return True

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        
        # Split data
        X_train, X_val, y_train, y_val = split_data(df)
        print(f"Train set: {len(X_train)} samples, Fraud ratio: {y_train.mean():.4f}")
        print(f"Val set: {len(X_val)} samples, Fraud ratio: {y_val.mean():.4f}")
        
        # Verify leakage
        verify_no_leakage(X_train, X_val)
        
        # Build and test pipeline
        pipeline = build_full_pipeline(use_smote=True)
        X_train_transformed, y_train_resampled = pipeline.fit_resample(X_train, y_train)
        
        print("Pipeline with SMOTE built and tested successfully.")
        print(f"Transformed X_train shape: {X_train_transformed.shape}")
        print(f"Resampled y_train shape: {y_train_resampled.shape}, Fraud ratio: {y_train_resampled.mean():.4f}")
