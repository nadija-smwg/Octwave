# Phase 5 — Data Splitting & Leakage Prevention

## Objective
Split data correctly before any fitting operations. Build a leak-proof sklearn Pipeline that encapsulates all preprocessing.

---

## 5.1 Train / Validation / Test Strategy

### Competition Context
- **Training set**: `train.csv` (8 000 rows) — we control this
- **Competition test set**: `test.csv` (2 000 rows) — no labels, for submission only
- **We need our own internal validation split** from `train.csv`

### Split Strategy
```python
from sklearn.model_selection import train_test_split

# Internal split of train.csv
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,          # 6400 train / 1600 validation
    random_state=42,
    stratify=y              # CRITICAL: preserve 1.5% fraud ratio in both splits
)
```

| Split | Rows | Expected Fraud | Purpose |
|---|---|---|---|
| Train | 6 400 | ~97 | Model fitting, CV |
| Validation | 1 600 | ~24 | Model selection, threshold tuning |
| Competition Test | 2 000 | Unknown | Final submission only |

### Why Stratified Split
With only 121 fraud cases, a random split could give the validation set as few as 10–15 fraud cases. Stratified ensures proportional representation.

---

## 5.2 Cross-Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **Stratified**: Maintains 1.5% fraud ratio in each fold
- **5-fold**: Each fold has ~1280 samples with ~19 fraud cases — enough for meaningful metrics
- **Scoring**: `f1` (primary), also track `roc_auc`, `precision`, `recall`

---

## 5.3 Leakage Prevention Checklist

| Potential Leakage Source | Status | Prevention |
|---|---|---|
| Imputation fitted on full data | ✅ Prevented | No missing values; imputer in pipeline fitted on train only |
| Scaling fitted on full data | ✅ Prevented | StandardScaler/MinMaxScaler inside Pipeline |
| Encoding fitted on full data | ✅ Prevented | OneHotEncoder inside ColumnTransformer |
| Feature selection using all data | ✅ Prevented | Selection done within cross-validation |
| Target leakage from features | ✅ None | No feature directly encodes the target |
| SMOTE on full training data | ✅ Prevented | SMOTE inside `imblearn.Pipeline`, applied per fold |
| Threshold tuning on test data | ✅ Prevented | Tuned on validation set only |

### Target Leakage Analysis
| Feature | Could it leak the target? | Assessment |
|---|---|---|
| `foreign_transaction` | No | Inherent transaction property, not derived from outcome |
| `location_mismatch` | No | Inherent transaction property |
| `device_trust_score` | Possibly | Could theoretically be post-hoc if updated after fraud detection. **Assumption**: It is a pre-transaction score. |
| `velocity_last_24h` | No | Count of prior transactions |
| All other features | No | Pre-transaction properties |

---

## 5.4 Pipeline Architecture

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Step 1: Feature engineering (stateless transforms) applied BEFORE pipeline
# (These are stateless — no leakage risk)

# Step 2: ColumnTransformer for encoding/scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['merchant_category']),
        ('bin', 'passthrough', binary_columns),
    ],
    remainder='drop'
)

# Step 3: Full pipeline with optional SMOTE
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),     # Only applied during fit
    ('classifier', model)
])
```

---

## 5.5 Code Structure for Phase 5

```
src/preprocessing.py
    ├── split_data(df, test_size=0.2, random_state=42) → X_train, X_val, y_train, y_val
    ├── get_cv_strategy(n_splits=5) → StratifiedKFold
    ├── build_preprocessor(numerical_cols, categorical_cols, binary_cols) → ColumnTransformer
    ├── build_pipeline(preprocessor, model, use_smote=False) → Pipeline
    └── verify_no_leakage(X_train, X_val) → validation checks
```
