# Phase 9 — Final Model Selection, Interpretability & Deployment

## Objective
Select the final model, interpret its decisions, generate predictions for the competition test set, and save a production-ready pipeline.

---

## 9.1 Final Model Selection

### Selection Criteria (in order of priority)
1. **Validation F1-score** (primary metric, matches competition)
2. **CV F1-score stability** (low standard deviation across folds)
3. **No overfitting** (train F1 − CV F1 gap < 0.1)
4. **Recall** (catching fraud is more important than precision in most real scenarios, but F1 balances both)
5. **Model complexity** (prefer simpler model if metrics are comparable)

### Final Comparison Table
| Model | CV F1 (mean±std) | Val F1 | Val F1 (tuned threshold) | Train-CV Gap | Selected? |
|---|---|---|---|---|---|
| XGBoost (tuned) | TBD | TBD | TBD | TBD | TBD |
| LightGBM (tuned) | TBD | TBD | TBD | TBD | TBD |
| Random Forest (tuned) | TBD | TBD | TBD | TBD | TBD |

---

## 9.2 Retrain on Full Training Data

Once the final model and hyperparameters are selected:
```python
# Retrain on ALL of train.csv (8000 rows) for maximum learning
final_pipeline.fit(X_full_train, y_full_train)
```

This gives the model access to 100% of the labeled data before making competition predictions.

---

## 9.3 Model Interpretability

### 9.3.1 Feature Importance (Built-in)
```python
# For tree-based models
importances = final_model.named_steps['classifier'].feature_importances_
```
- Bar chart of feature importances
- Ranked list with percentage contributions

### 9.3.2 Permutation Importance
```python
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(
    final_pipeline, X_val, y_val,
    n_repeats=30, random_state=42, scoring='f1'
)
```
- More reliable than built-in importance for tree ensembles
- Shows actual impact of shuffling each feature on F1

### 9.3.3 SHAP Values
```python
import shap

# For tree models
explainer = shap.TreeExplainer(final_model.named_steps['classifier'])
shap_values = explainer.shap_values(X_val_processed)

# Visualizations
shap.summary_plot(shap_values, X_val_processed)    # Feature importance
shap.dependence_plot('device_trust_score', shap_values, X_val_processed)
shap.force_plot(explainer.expected_value, shap_values[0])  # Single prediction
```

### 9.3.4 Expected Feature Ranking (Hypothesis)
1. `device_trust_score` — Strongest single predictor (large separation between fraud/legit)
2. `transaction_hour` / `hour_sin/cos` — Strong temporal pattern
3. `foreign_transaction` — 10× higher fraud rate
4. `location_mismatch` — 9× higher fraud rate
5. `velocity_last_24h` — Moderate signal
6. `risk_flags_count` / interaction features — Compound signals
7. `amount` / `log_amount` — Weak standalone, may help in combinations
8. `cardholder_age` — Likely near-zero importance

---

## 9.4 Generate Competition Predictions

```python
# Load and prepare test data
test_df = pd.read_csv('data/raw/test.csv')
test_engineered = engineer_features(test_df)

# Predict using final pipeline
test_proba = final_pipeline.predict_proba(test_engineered)[:, 1]
test_pred = (test_proba >= optimal_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'transaction_id': test_df['transaction_id'],
    'is_fraud': test_pred
})
submission.to_csv('submission.csv', index=False)
```

### Submission Validation
- [ ] Exactly 2 000 rows
- [ ] Columns: `transaction_id`, `is_fraud`
- [ ] `transaction_id` matches `test.csv` IDs
- [ ] `is_fraud` is binary (0 or 1)
- [ ] Predicted fraud rate is reasonable (~1–5%)

---

## 9.5 Save Final Pipeline

```python
import joblib

# Save complete pipeline (preprocessor + model)
joblib.dump(final_pipeline, 'models/final_model.pkl')

# Save metadata
metadata = {
    'model_type': 'XGBoost',  # or whatever was selected
    'best_params': best_params,
    'cv_f1': cv_f1_mean,
    'val_f1': val_f1,
    'optimal_threshold': optimal_threshold,
    'feature_columns': feature_columns,
    'training_date': datetime.now().isoformat(),
    'random_state': 42
}
joblib.dump(metadata, 'models/model_metadata.pkl')
```

### Pipeline Usage (for new predictions)
```python
import joblib
pipeline = joblib.load('models/final_model.pkl')
metadata = joblib.load('models/model_metadata.pkl')

# Raw data → prediction
raw_data = pd.read_csv('new_transactions.csv')
engineered = engineer_features(raw_data)
predictions = pipeline.predict(engineered)
```

---

## 9.6 Model Strengths & Limitations

### Strengths
- Handles class imbalance explicitly
- Feature engineering captures domain-relevant fraud patterns
- Pipeline ensures no data leakage
- Threshold optimization maximizes F1

### Limitations
- Small dataset (8000 rows) limits model complexity
- Only 121 positive examples — model may not generalize to rare fraud patterns
- Simulated data may not capture real-world fraud complexity
- No temporal features (transaction date/sequence not available)
- Device trust score origin is unknown — possible circular reasoning

---

## 9.7 Code Structure for Phase 9

```
src/predict.py
    ├── load_pipeline(model_path) → pipeline
    ├── prepare_test_data(test_df) → engineered df
    ├── generate_predictions(pipeline, test_df, threshold) → predictions
    ├── create_submission(test_ids, predictions, output_path) → submission df
    └── validate_submission(submission_df) → validation report

src/interpret.py
    ├── get_feature_importance(pipeline) → importance df
    ├── compute_permutation_importance(pipeline, X, y) → importance df
    ├── compute_shap_values(pipeline, X) → shap_values
    ├── plot_shap_summary(shap_values, X, save_path)
    └── generate_interpretation_report(results, output_path)
```
