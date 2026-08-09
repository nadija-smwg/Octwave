import os
import pandas as pd
import joblib

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'data', 'processed', 'test_cleaned.csv')
    test_ids_path = os.path.join(base_dir, 'data', 'processed', 'test_ids.csv')
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    output_dir = os.path.join(base_dir, 'output')
    output_file = os.path.join(output_dir, 'final_predictions.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(test_path) or not os.path.exists(model_path):
        print(f"Error: Missing files. Ensure {test_path} and {model_path} exist.")
        return
        
    print("Loading Gradient Boosting model...")
    pipeline = joblib.load(model_path)
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    test_ids = pd.read_csv(test_ids_path)['transaction_id']
    
    print("Generating predictions...")
    # The pipeline automatically handles feature engineering and preprocessing
    predictions = pipeline.predict(test_df)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'transaction_id': test_ids,
        'is_fraud': predictions
    })
    
    # Validation checks
    assert len(submission) == 2000, f"Expected 2000 rows, got {len(submission)}"
    
    fraud_rate = submission['is_fraud'].mean()
    print(f"Predicted Fraud Rate: {fraud_rate * 100:.2f}%")
    
    # Save the output
    submission.to_csv(output_file, index=False)
    print(f"\nFinal predictions successfully saved to: {output_file}")

if __name__ == '__main__':
    main()
