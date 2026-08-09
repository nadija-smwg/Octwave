# 🧑‍💻 Member 2 — Modelling Track (Phases 5–10)

## Scope
You own **pipeline architecture, feature selection, model training, tuning, final model, and project documentation**.  
You depend on Member 1's feature engineering outputs.

---

## Timeline & Dependencies

```
Member 1 (Data)                         Member 2 (Modelling)
─────────────────                       ─────────────────────
Phase 1: Data Profiling                 Setup project structure (parallel)
Phase 2: Data Cleaning                  Phase 5: Pipeline architecture (parallel)
Phase 3: EDA                            Phase 6: Feature selection code (parallel)
Phase 4: Feature Eng.  ── HANDOFF ──►   Phase 7: Model Training
                                        Phase 8: Hyperparameter Tuning
                                        Phase 9: Final Model & Interpretability
                                        Phase 10: Documentation & Deliverables
```

**Phases 5–6 can start immediately (parallel with Member 1).**  
**Phases 7–10 require Member 1's engineered data.**

---

## PARALLEL WORK (Start Immediately)

### Setup — Project Structure
- [ ] Create directory structure:
```
mkdir data/raw data/processed notebooks src models reports reports/figures
```
- [ ] Copy original CSVs to `data/raw/`
- [ ] Create `src/__init__.py`
- [ ] Create `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
joblib>=1.3.0
statsmodels>=0.14.0
```
- [ ] Install dependencies: `pip install -r requirements.txt`

---

## Phase 5 — Data Splitting & Leak-Proof Pipeline (Can Start Now)

### Your Tasks
- [ ] Create `src/preprocessing.py` with:

```python
# --- Column definitions (will be confirmed by Member 1) ---
NUMERICAL_COLS = [
    'log_amount', 'device_trust_score', 'velocity_last_24h',
    'cardholder_age', 'amount_per_velocity', 'hour_sin', 'hour_cos'
]
BINARY_COLS = [
    'foreign_transaction', 'location_mismatch',
    'high_velocity_low_trust', 'late_night_flag',
    'late_night_x_low_trust', 'foreign_x_location'
]
CATEGORICAL_COLS = ['merchant_category']
ORDINAL_COLS = ['risk_flags_count']  # 0, 1, 2 — treat as numeric


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/validation split."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    # Expected: 6400 train / 1600 val
    # ~97 fraud in train, ~24 fraud in val


def get_cv_strategy(n_splits=5, random_state=42):
    """Stratified K-Fold for cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_preprocessor():
    """ColumnTransformer: scale numericals, encode categoricals, pass binaries."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'), CATEGORICAL_COLS),
            ('bin', 'passthrough', BINARY_COLS),
            ('ord', 'passthrough', ORDINAL_COLS),
        ],
        remainder='drop'
    )


def build_pipeline(model, use_smote=False):
    """Build complete preprocessing + model pipeline."""
    preprocessor = build_preprocessor()

    if use_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
            ('classifier', model)
        ])
    else:
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
```

### Leakage Prevention Rules
| Rule | How |
|---|---|
| Scaling | `StandardScaler` inside Pipeline → fit on train only |
| Encoding | `OneHotEncoder` inside ColumnTransformer → fit on train only |
| SMOTE | Inside `imblearn.Pipeline` → applied only during `.fit()` |
| Feature engineering | Stateless transforms in Member 1's code → no leakage |
| CV | `StratifiedKFold` → maintains class ratio per fold |
| Threshold tuning | On validation set only, never on test |

---

## Phase 6 — Feature Selection (Can Start Scaffolding Now)

### Your Tasks
- [ ] Create `src/feature_selection.py` with:

```python
def compute_correlation_with_target(X, y):
    """Pearson correlation of each feature with target."""
    import pandas as pd
    correlations = pd.DataFrame(X).corrwith(pd.Series(y))
    return correlations.abs().sort_values(ascending=False)


def compute_mutual_information(X, y, random_state=42):
    """Mutual information scores (captures non-linear relationships)."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def compute_vif(X):
    """Variance Inflation Factor for multicollinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False)


def compute_tree_importance(X, y, random_state=42):
    """Random Forest feature importance."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=random_state
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def select_features(X, y):
    """Run all methods and return consensus feature ranking."""
    corr = compute_correlation_with_target(X, y)
    mi = compute_mutual_information(X, y)
    tree_imp = compute_tree_importance(X, y)
    vif = compute_vif(X)
    # Combine rankings, flag features low across all methods
    return corr, mi, tree_imp, vif
```

### Expected Outcomes
| Feature | Likely Verdict |
|---|---|
| `device_trust_score` | ✅ Keep — top signal |
| `hour_sin` / `hour_cos` | ✅ Keep — strong temporal signal |
| `foreign_transaction` | ✅ Keep — 10× fraud rate |
| `location_mismatch` | ✅ Keep — 9× fraud rate |
| `velocity_last_24h` | ✅ Keep — moderate signal |
| `risk_flags_count` | ✅ Keep — likely redundant with components but adds interpretability |
| `log_amount` | ✅ Keep — weak but may help in combinations |
| `late_night_flag` | ✅ Keep — direct fraud-hour signal |
| `cardholder_age` | ⚠️ Evaluate — near-zero correlation, check MI |
| `foreign_x_location` | ⚠️ Evaluate — may overlap with `risk_flags_count` |

---

## AFTER HANDOFF FROM MEMBER 1 (Requires Engineered Data)

---

## Phase 7 — Model Training & Comparison

### Your Tasks
- [ ] Create `src/train.py`:

```python
RANDOM_STATE = 42

def get_models():
    """Return dict of model name → configured model instance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    # scale_pos_weight ≈ 65 (n_neg / n_pos)
    return {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, scale_pos_weight=65,
            eval_metric='logloss', random_state=RANDOM_STATE
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, is_unbalance=True,
            random_state=RANDOM_STATE, verbose=-1
        ),
        'SVM (RBF)': SVC(
            class_weight='balanced', probability=True, random_state=RANDOM_STATE
        ),
    }
```

- [ ] Create `src/evaluate.py`:

```python
def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """Compute all classification metrics."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        accuracy_score, roc_auc_score, average_precision_score,
        confusion_matrix
    )
    metrics = {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    return metrics


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximises F1 score."""
    import numpy as np
    from sklearn.metrics import f1_score
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]
```

- [ ] Create `notebooks/03_model_training.ipynb` that:
  1. Loads engineered data from `data/processed/`
  2. Calls `engineer_features()` from Member 1's code
  3. Splits data (stratified)
  4. Runs feature selection
  5. Trains all 6 models with cross-validation
  6. Evaluates on validation set
  7. Produces comparison table

### Model Comparison Table (to fill in)
| Model | CV F1 (mean±std) | Val F1 | Val Precision | Val Recall | Val ROC-AUC | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| LightGBM | — | — | — | — | — | — |
| SVM (RBF) | — | — | — | — | — | — |

### Visualizations to Generate
- [ ] Model comparison bar chart (F1 scores)
- [ ] ROC curves (all models on same plot)
- [ ] Precision-Recall curves (all models)
- [ ] Confusion matrices (grid of 6)

---

## Phase 8 — Hyperparameter Tuning

### Your Tasks
- [ ] Create `src/tune.py`
- [ ] Select top 2–3 models from Phase 7
- [ ] Run `RandomizedSearchCV` with `scoring='f1'`, `cv=StratifiedKFold(5)`

### Search Spaces (XGBoost Example)
```python
xgb_params = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 3, 5, 7],
    'classifier__scale_pos_weight': [30, 50, 65, 80, 100],
}
```

- [ ] After tuning: optimize classification threshold on validation set
- [ ] Compare SMOTE vs class-weight approaches for top model
- [ ] Save tuning results to `reports/model_results.md`

---

## Phase 9 — Final Model, Interpretability & Predictions

### Your Tasks

#### 9.1 Final Model Selection
- [ ] Select model with best **validation F1** (not training F1)
- [ ] Retrain on **full train.csv** (all 8000 rows) with best hyperparameters
- [ ] Apply optimal threshold

#### 9.2 Interpretability
- [ ] Create `src/interpret.py`
- [ ] Feature importance (built-in) → bar chart
- [ ] Permutation importance → bar chart
- [ ] SHAP summary plot
- [ ] SHAP dependence plots for top 3 features
- [ ] Save all to `reports/figures/`

#### 9.3 Generate Submission
```python
# Load test, apply Member 1's feature engineering, predict
test_df = pd.read_csv('data/raw/test.csv')
test_engineered = engineer_features(test_df)
test_proba = final_pipeline.predict_proba(test_engineered)[:, 1]
test_pred = (test_proba >= optimal_threshold).astype(int)

submission = pd.DataFrame({
    'transaction_id': test_df['transaction_id'],
    'is_fraud': test_pred
})
submission.to_csv('submission.csv', index=False)
```

- [ ] Validate: 2000 rows, 2 columns, binary values, reasonable fraud rate

#### 9.4 Save Pipeline
```python
import joblib
joblib.dump(final_pipeline, 'models/final_model.pkl')
joblib.dump({
    'model_type': '...',
    'best_params': best_params,
    'optimal_threshold': optimal_threshold,
    'cv_f1': cv_f1,
    'val_f1': val_f1,
    'feature_columns': feature_columns,
}, 'models/model_metadata.pkl')
```

---

## Phase 10 — Documentation & Deliverables

### Your Tasks
- [ ] Write `README.md` (project overview, setup, reproduction steps, results)
- [ ] Write `reports/model_results.md` (model comparison table, tuning results)
- [ ] Write `reports/final_report.md`:
  1. Executive Summary
  2. Data Cleaning Summary (from Member 1's report)
  3. Key EDA Findings (from Member 1's plots)
  4. Feature Engineering (from Member 1's code)
  5. Models Tested & Comparison Table
  6. Final Model: why selected, hyperparameters, feature importance
  7. SHAP Analysis Highlights
  8. Limitations & Future Work
- [ ] Create `src/predict.py` (load pipeline, predict from raw CSV)

---

## 🤝 What You Need from Member 1

| Item | File | When Needed |
|---|---|---|
| Feature engineering function | `src/feature_engineering.py` | Before Phase 7 |
| Cleaning function | `src/data_cleaning.py` | Before Phase 7 |
| Engineered train/test CSVs | `data/processed/*.csv` | Before Phase 7 |
| Column name lists | In `feature_engineering.py` | Before Phase 7 |
| EDA plots | `reports/figures/` | Before Phase 10 |
| Data quality report | `reports/data_quality_report.md` | Before Phase 10 |

---

## Deliverables Checklist (Member 2 Owns)

- [ ] `src/preprocessing.py` — Pipeline/ColumnTransformer
- [ ] `src/feature_selection.py` — All selection methods
- [ ] `src/train.py` — Model definitions & training
- [ ] `src/evaluate.py` — Metrics & threshold tuning
- [ ] `src/tune.py` — Hyperparameter search
- [ ] `src/predict.py` — Prediction from raw data
- [ ] `src/interpret.py` — SHAP, permutation importance
- [ ] `notebooks/03_model_training.ipynb` — Full modelling notebook
- [ ] `models/final_model.pkl` — Saved pipeline
- [ ] `models/model_metadata.pkl` — Metadata
- [ ] `reports/model_results.md` — Comparison table
- [ ] `reports/final_report.md` — Complete report
- [ ] `requirements.txt` — Dependencies
- [ ] `README.md` — Project documentation
- [ ] `submission.csv` — Competition submission
