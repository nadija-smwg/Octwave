# Phase 8 — Hyperparameter Tuning

## Objective
Tune the top 2–3 models from Phase 7 to maximize F1-score using cross-validated search.

---

## 8.1 Tuning Strategy

- **Method**: `RandomizedSearchCV` (efficient for continuous hyperparameters)
- **CV**: `StratifiedKFold(5)` — same as model comparison
- **Scoring**: `f1`
- **n_iter**: 50–100 per model (balance quality vs compute time)
- **Random state**: 42 for reproducibility

---

## 8.2 Hyperparameter Search Spaces

### XGBoost
```python
xgb_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 3, 5, 7],
    'classifier__gamma': [0, 0.1, 0.2, 0.5],
    'classifier__reg_alpha': [0, 0.01, 0.1, 1.0],
    'classifier__reg_lambda': [0.5, 1.0, 2.0, 5.0],
    'classifier__scale_pos_weight': [30, 50, 65, 80, 100],
}
```

### LightGBM
```python
lgbm_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [-1, 3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__num_leaves': [15, 31, 50, 63],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_samples': [5, 10, 20, 30],
    'classifier__reg_alpha': [0, 0.01, 0.1, 1.0],
    'classifier__reg_lambda': [0, 0.01, 0.1, 1.0],
    'classifier__is_unbalance': [True],
}
```

### Random Forest
```python
rf_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7],
    'classifier__class_weight': ['balanced', 'balanced_subsample'],
}
```

### Gradient Boosting
```python
gb_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', 0.5],
}
```

---

## 8.3 Tuning Process

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=80,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    refit=True  # Refit best model on full training data
)

search.fit(X_train, y_train)
```

### Post-Tuning Steps
1. Extract `search.best_params_` and `search.best_score_`
2. Evaluate best model on validation set
3. Check for overfitting: compare train F1 vs CV F1
4. Record top-5 parameter combinations for analysis

---

## 8.4 Threshold Optimization (Post-Tuning)

After tuning model hyperparameters, optimize the classification threshold:

```python
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# Search thresholds from 0.1 to 0.9
thresholds = np.arange(0.05, 0.95, 0.01)
f1_scores = [f1_score(y_val, (y_val_proba >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

- Default threshold (0.5) may not be optimal for imbalanced data
- Lower threshold → more fraud detected (higher recall, lower precision)
- Plot Precision-Recall vs Threshold curve

---

## 8.5 SMOTE Variant Comparison (Optional)

If class weights alone don't perform well, test SMOTE variants:

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

smote_variants = {
    'SMOTE_0.3': SMOTE(sampling_strategy=0.3, random_state=42),
    'SMOTE_0.5': SMOTE(sampling_strategy=0.5, random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
}
```

---

## 8.6 Expected Outputs

- Best hyperparameters for top 2–3 models
- Tuned CV F1 scores
- Validation F1 scores (pre and post threshold tuning)
- Overfitting analysis (train vs CV gap)
- `reports/tuning_results.md`

---

## 8.7 Code Structure for Phase 8

```
src/tune.py
    ├── get_param_grids() → dict of {model_name: param_dict}
    ├── tune_model(pipeline, param_grid, X_train, y_train, cv) → SearchCV result
    ├── optimize_threshold(y_true, y_proba) → optimal_threshold
    ├── compare_tuned_models(results_dict) → comparison DataFrame
    └── save_tuning_results(results, output_path)
```
