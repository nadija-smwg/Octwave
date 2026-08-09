import os
import pandas as pd
import numpy as np
import joblib

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'data', 'raw', 'test.csv')
    model_path = os.path.join(base_dir, 'models', 'final_model.pkl')
    metadata_path = os.path.join(base_dir, 'models', 'model_metadata.pkl')
    submission_path = os.path.join(base_dir, 'submission.csv')
    
    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print("Error: Missing test.csv or final_model.pkl")
        return
        
    print("Loading model and metadata...")
    pipeline = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    optimal_threshold = metadata['optimal_threshold']
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    # Store transaction_id for submission
    test_ids = test_df['transaction_id']
    
    # We don't need is_fraud from test set (it doesn't exist). 
    # But preprocessing expects features. Our pipeline includes feature engineering.
    # The ColumnTransformer is designed to take the engineered DataFrame.
    # However, our feature_engineering function engineer_features(df) needs to be called.
    from src.feature_engineering import engineer_features
    test_engineered = engineer_features(test_df)
    
    # Preprocessor handles the rest (scaling, encoding) inside the pipeline.
    # Wait, the pipeline has: [('feature_engineering', FunctionTransformer(engineer_features)), ('preprocessing', preprocessor), ('classifier', model)]
    # Oh! Yes, `build_full_pipeline` actually includes `engineer_features` if we passed it.
    # Let's check `src/preprocessing.py` build_full_pipeline:
    # Actually, in `src/preprocessing.py`, I did NOT include `engineer_features` directly inside the pipeline. I did it outside in most scripts.
    # Wait, `build_full_pipeline` in `src/preprocessing.py` takes `model` and `use_smote`. 
    # It just has [('preprocessing', build_preprocessor()), ('classifier', model)].
    # So I MUST call `engineer_features` manually before predict.
    
    print(f"Generating predictions using threshold: {optimal_threshold}...")
    # Predict probabilities
    # We must ensure test_engineered has same columns as train_engineered. 
    # The engineer_features function handles this.
    test_proba = pipeline.predict_proba(test_engineered)[:, 1]
    
    # Apply optimal threshold
    test_pred = (test_proba >= optimal_threshold).astype(int)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'transaction_id': test_ids,
        'is_fraud': test_pred
    })
    
    # Validation checks
    print("\nValidating submission file...")
    assert len(submission) == 2000, f"Expected 2000 rows, got {len(submission)}"
    assert list(submission.columns) == ['transaction_id', 'is_fraud'], "Invalid columns"
    assert submission['is_fraud'].isin([0, 1]).all(), "Predictions must be binary"
    
    fraud_rate = submission['is_fraud'].mean()
    print(f"Predicted Fraud Rate: {fraud_rate * 100:.2f}%")
    
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")

if __name__ == '__main__':
    main()
