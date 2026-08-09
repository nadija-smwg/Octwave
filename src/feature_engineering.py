import pandas as pd
import numpy as np

def create_log_amount(df):
    """Log transform of amount to reduce right-skewness."""
    df = df.copy()
    if 'amount' in df.columns:
        df['log_amount'] = np.log1p(df['amount'])
    return df

def create_cyclical_hour(df):
    """Cyclical encoding of transaction_hour using sine and cosine."""
    df = df.copy()
    if 'transaction_hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
    return df

def create_interaction_features(df):
    """Create interaction features based on domain knowledge."""
    df = df.copy()
    
    if 'foreign_transaction' in df.columns and 'location_mismatch' in df.columns:
        df['risk_flags_count'] = df['foreign_transaction'] + df['location_mismatch']
        df['foreign_x_location_mismatch'] = df['foreign_transaction'] * df['location_mismatch']
        
    if 'velocity_last_24h' in df.columns and 'device_trust_score' in df.columns:
        df['high_velocity_low_trust'] = ((df['velocity_last_24h'] >= 4) & (df['device_trust_score'] <= 40)).astype(int)
        
    if 'transaction_hour' in df.columns:
        df['late_night_flag'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 4)).astype(int)
        
        if 'device_trust_score' in df.columns:
            df['late_night_x_low_trust'] = df['late_night_flag'] * (df['device_trust_score'] <= 40).astype(int)
            
    return df

def create_ratio_features(df):
    """Create ratio and mathematical combination features."""
    df = df.copy()
    if 'amount' in df.columns and 'velocity_last_24h' in df.columns:
        df['amount_per_velocity'] = df['amount'] / (df['velocity_last_24h'] + 1)
        df['velocity_x_amount'] = df['velocity_last_24h'] * df['amount']
    return df

def create_binned_features(df):
    """Create binned categorical features from numerical ones."""
    df = df.copy()
    if 'cardholder_age' in df.columns:
        df['age_group'] = pd.cut(df['cardholder_age'], bins=[17, 25, 35, 50, 70], labels=['young', 'adult', 'middle', 'senior'])
        
    if 'device_trust_score' in df.columns:
        df['device_trust_bin'] = pd.cut(df['device_trust_score'], bins=[24, 40, 60, 80, 100], labels=['very_low', 'low', 'medium', 'high'])
    return df

def engineer_features(df):
    """Apply all feature engineering steps."""
    df = create_log_amount(df)
    df = create_cyclical_hour(df)
    df = create_interaction_features(df)
    df = create_ratio_features(df)
    df = create_binned_features(df)
    return df

def get_feature_columns():
    """Return lists of engineered feature column names by type."""
    return {
        'engineered_numeric': [
            'log_amount', 'hour_sin', 'hour_cos', 'risk_flags_count', 
            'amount_per_velocity', 'velocity_x_amount'
        ],
        'engineered_binary': [
            'foreign_x_location_mismatch', 'high_velocity_low_trust', 
            'late_night_flag', 'late_night_x_low_trust'
        ],
        'engineered_categorical': [
            'age_group', 'device_trust_bin'
        ]
    }

if __name__ == '__main__':
    # Simple test
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        df_engineered = engineer_features(df)
        print("Engineered features created successfully. Shape:", df_engineered.shape)
        print("New columns:", set(df_engineered.columns) - set(df.columns))
