# Phase 9 — Final Model Selection, Interpretability & Deployment

## Objective
Select the final model, interpret its decisions, generate predictions for the competition test set, save a production-ready pipeline, and prepare winner-requirement documentation.

---

## Rules Compliance

| Rule | How We Comply |
|---|---|
| F1-score evaluation | Final model selected by validation F1, not training score |
| Up to 2 final submissions | Prepare Submission A (aggressive) + Submission B (conservative) |
| Max 10 submissions/day | Only submit after strong local validation |
| Reproducible code required | Full pipeline saved with joblib; all code documented in `src/` |
| Methodology overview required | `reports/final_report.md` covers all methodology for winner requirements |
| No external data | Pipeline only uses provided dataset features |
| Open source only | All libraries are OSI-approved (sklearn, xgboost, lightgbm, shap) |

---

## 9.1 Final Model Selection

### Selection Criteria (in order of priority)
1. **Validation F1-score** — primary metric matching competition rules
2. **CV F1-score stability** (low standard deviation across folds)
3. **No overfitting** (train F1 − CV F1 gap < 0.1)
4. **Recall** (catching fraud is important, but F1 balances both precision and recall)
5. **Model complexity** (prefer simpler model if metrics are comparable — better generalization)

### Final Comparison Table
| Model | CV F1 (mean±std) | Val F1 | Val F1 (tuned threshold) | Train-CV Gap | Selected? |
|---|---|---|---|---|---|
| XGBoost (tuned) | TBD | TBD | TBD | TBD | TBD |
| LightGBM (tuned) | TBD | TBD | TBD | TBD | TBD |
| Random Forest (tuned) | TBD | TBD | TBD | TBD | TBD |

---

## 9.2 Dual Submission Strategy (Rules: Up to 2 Final Submissions)

The competition allows selecting **up to 2 submissions** for the Private Leaderboard.

### Submission A — Aggressive (Best F1)
- Best single model with optimal threshold tuning
- Highest validation F1
- Risk: may overfit to Public LB patterns

### Submission B — Conservative (Best Generalization)
- Ensemble of top 2–3 models (soft voting or stacking)
- OR: Model with strong CV F1 and lowest CV standard deviation
- Safety pick: more likely to generalize to Private LB

### Submission Tracking Log
| # | Date | Model | Threshold | Local Val F1 | Public LB F1 | Notes |
|---|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD | First submission |
| ... | | | | | | Max 10/day |

---

## 9.3 Retrain on Full Training Data

Once the final model and hyperparameters are selected:
```python
# Retrain on ALL of train.csv (8000 rows) for maximum learning
final_pipeline.fit(X_full_train, y_full_train)
```

This gives the model access to 100% of the labeled data before making competition predictions.

---

## 9.4 Model Interpretability

### 9.4.1 Feature Importance (Built-in)
```python
importances = final_model.named_steps['classifier'].feature_importances_
```
- Bar chart of feature importances
- Ranked list with percentage contributions

### 9.4.2 Permutation Importance
```python
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(
    final_pipeline, X_val, y_val,
    n_repeats=30, random_state=42, scoring='f1'  # Use F1 per competition rules
)
```

### 9.4.3 SHAP Values
```python
import shap
explainer = shap.TreeExplainer(final_model.named_steps['classifier'])
shap_values = explainer.shap_values(X_val_processed)

shap.summary_plot(shap_values, X_val_processed)
shap.dependence_plot('device_trust_score', shap_values, X_val_processed)
```

### 9.4.4 Expected Feature Ranking (Hypothesis)
1. `device_trust_score` — Strongest single predictor
2. `transaction_hour` / `hour_sin/cos` — Strong temporal pattern
3. `foreign_transaction` — 10× higher fraud rate
4. `location_mismatch` — 9× higher fraud rate
5. `velocity_last_24h` — Moderate signal
6. Interaction features — Compound signals
7. `amount` / `log_amount` — Weak standalone
8. `cardholder_age` — Likely near-zero importance

---

## 9.5 Generate Competition Predictions

```python
test_df = pd.read_csv('data/raw/test.csv')
test_engineered = engineer_features(test_df)

# Predict using final pipeline
test_proba = final_pipeline.predict_proba(test_engineered)[:, 1]
test_pred = (test_proba >= optimal_threshold).astype(int)

submission = pd.DataFrame({
    'transaction_id': test_df['transaction_id'],
    'is_fraud': test_pred
})
submission.to_csv('submission.csv', index=False)
```

### Submission Validation Checklist (Before Uploading to Kaggle)
- [ ] Exactly 2 000 rows
- [ ] Columns: `transaction_id`, `is_fraud` (matches sample_submission format)
- [ ] `transaction_id` values match `test.csv` IDs exactly
- [ ] `is_fraud` is binary (0 or 1 only)
- [ ] Predicted fraud rate is reasonable (~1–5%, not 0% or 50%)
- [ ] File is CSV format with no extra columns or index
- [ ] Daily submission count < 10 before uploading

---

## 9.6 Save Pipeline & Metadata

```python
import joblib

# Save complete pipeline (preprocessor + model)
joblib.dump(final_pipeline, 'models/final_model.pkl')

# Save metadata
metadata = {
    'model_type': 'XGBoost',
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

---

## 9.7 Winner Requirements Documentation (Rules Section 5)

If we place in the top rankings, we must submit:

### Required: Reproducible Source Code
- [ ] All code in `src/` is clean, commented, and runs end-to-end
- [ ] `requirements.txt` lists all dependencies with versions
- [ ] `README.md` has step-by-step reproduction instructions
- [ ] Running the pipeline from scratch produces the same submission file

### Required: Methodology Overview
- [ ] `reports/final_report.md` contains:
  - Data preprocessing steps
  - Feature engineering rationale
  - Model selection process
  - Hyperparameter tuning approach
  - Final model description
  - Feature importance analysis

---

## 9.8 Model Strengths & Limitations

### Strengths
- Handles class imbalance explicitly via class weights + threshold tuning
- Feature engineering captures domain-relevant fraud patterns
- Pipeline ensures no data leakage
- Threshold optimization maximizes F1 (competition metric)
- Dual submission strategy hedges against Public/Private LB divergence

### Limitations
- Small dataset (8000 rows) limits model complexity
- Only 121 positive examples — model may not generalize to rare fraud patterns
- Simulated data may not capture real-world fraud complexity
- No temporal features (transaction date/sequence not available)
- Device trust score origin is unknown — possible circular reasoning

---

## 9.9 Code Structure

```
src/predict.py
    ├── load_pipeline(model_path) → pipeline
    ├── prepare_test_data(test_df) → engineered df
    ├── generate_predictions(pipeline, test_df, threshold) → predictions
    ├── create_submission(test_ids, predictions, output_path) → submission df
    └── validate_submission(submission_df, sample_submission_df) → validation report

src/interpret.py
    ├── get_feature_importance(pipeline) → importance df
    ├── compute_permutation_importance(pipeline, X, y) → importance df
    ├── compute_shap_values(pipeline, X) → shap_values
    ├── plot_shap_summary(shap_values, X, save_path)
    └── generate_interpretation_report(results, output_path)
```
