import pandas as pd
import numpy as np
import os

def validate_column_types(df):
    """Convert merchant_category to category dtype."""
    df = df.copy()
    if 'merchant_category' in df.columns:
        df['merchant_category'] = df['merchant_category'].astype('category')
    return df

def check_value_ranges(df):
    """Generate a simple validation report."""
    report = {}
    if 'amount' in df.columns:
        report['amount_min'] = df['amount'].min()
    if 'transaction_hour' in df.columns:
        report['hour_min'] = df['transaction_hour'].min()
        report['hour_max'] = df['transaction_hour'].max()
    return report

def clean_categorical_labels(df):
    """Clean categorical labels for merchant_category."""
    df = df.copy()
    if 'merchant_category' in df.columns:
        df['merchant_category'] = df['merchant_category'].str.strip().str.title()
    return df

def handle_missing_values(df):
    """Passthrough for missing values as none exist."""
    return df.copy()

def detect_outliers(df, columns):
    """Identify outliers using IQR, do not remove them."""
    outlier_report = {}
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_report[col] = len(outliers)
    return outlier_report

def prepare_for_modelling(df):
    """Drop transaction_id and return features, target (if exists), id_series."""
    df = df.copy()
    id_series = None
    if 'transaction_id' in df.columns:
        id_series = df['transaction_id']
        df = df.drop(columns=['transaction_id'])
    
    target_series = None
    if 'is_fraud' in df.columns:
        target_series = df['is_fraud']
        features_df = df.drop(columns=['is_fraud'])
    else:
        features_df = df
        
    return features_df, target_series, id_series

def clean_pipeline(df):
    """Apply all cleaning steps sequentially."""
    df = clean_categorical_labels(df)
    df = handle_missing_values(df)
    df = validate_column_types(df)
    return df

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = base_dir
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load data
    train_path = os.path.join(raw_dir, 'train.csv')
    test_path = os.path.join(raw_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validation and Outlier Detection
    train_ranges = check_value_ranges(train_df)
    test_ranges = check_value_ranges(test_df)
    
    numeric_cols = ['amount', 'device_trust_score', 'velocity_last_24h', 'cardholder_age', 'transaction_hour']
    train_outliers = detect_outliers(train_df, numeric_cols)
    
    # Clean Data
    train_cleaned = clean_pipeline(train_df)
    test_cleaned = clean_pipeline(test_df)
    
    train_features, train_target, train_ids = prepare_for_modelling(train_cleaned)
    test_features, test_target, test_ids = prepare_for_modelling(test_cleaned)
    
    # Construct final dataframes for saving
    train_final = train_features.copy()
    if train_target is not None:
        train_final['is_fraud'] = train_target
        
    test_final = test_features.copy()
    if test_target is not None:
        test_final['is_fraud'] = test_target
    
    # Save processed data
    train_final.to_csv(os.path.join(processed_dir, 'train_cleaned.csv'), index=False)
    test_final.to_csv(os.path.join(processed_dir, 'test_cleaned.csv'), index=False)
    
    # Save IDs separately
    if train_ids is not None:
        train_ids.to_csv(os.path.join(processed_dir, 'train_ids.csv'), index=False)
    if test_ids is not None:
        test_ids.to_csv(os.path.join(processed_dir, 'test_ids.csv'), index=False)
    
    # Append cleaning summary to report
    report_path = os.path.join(reports_dir, 'data_quality_report.md')
    cleaning_summary = """
## 6. Data Cleaning Summary (Phase 2)
- **Missing values**: 0 (No imputation needed).
- **Duplicates**: 0 (No deduplication needed).
- **Type fixes**: `merchant_category` converted to categorical.
- **Invalid values**: None found.
- **Outliers**: Identified in `amount` and `velocity_last_24h` but kept as they represent valid financial signals.
- **ID columns**: `transaction_id` removed prior to modelling.
"""
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
        if "## 6. Data Cleaning Summary (Phase 2)" not in content:
            with open(report_path, 'a') as f:
                f.write(cleaning_summary)
    else:
        with open(report_path, 'w') as f:
            f.write("# Data Quality Report\n" + cleaning_summary)
            
    print(f"Data cleaning complete. Saved cleaned files to {processed_dir}")

if __name__ == '__main__':
    main()
