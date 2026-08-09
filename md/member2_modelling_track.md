# 🧑‍💻 Member 2 — Modelling Track (Phases 5–10)

## Scope
You own **pipeline architecture, feature selection, model training, tuning, final model, and project documentation**.  
You depend on Member 1's feature engineering outputs.

---

## Competition Rules Relevant to You

| Rule | Impact |
|---|---|
| **F1-score metric** (Section 3) | ALL model selection, CV, and tuning uses `scoring='f1'` |
| **Max 10 submissions/day** (Section 3) | Validate locally first; don't waste submissions |
| **Up to 2 final submissions** (Section 3) | Prepare Submission A (aggressive) + Submission B (conservative) |
| **No external data** (Section 4) | No pre-trained models, no external feature sources |
| **Winner Requirements** (Section 5) | Must submit: reproducible code + methodology overview |
| **Open source only** (Section 6c) | All libraries must be OSI-approved |
| **No hand-labeling** (Section 4b) | Test predictions come from model only |
| **Reproducibility** | `random_state=42`; pinned dependency versions; README with steps |

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
                                        Phase 9: Final Model & Submission
                                        Phase 10: Documentation (Winner Req.)
```

---

## PARALLEL WORK (Start Immediately)

### Setup — Project Structure
- [ ] Create directory structure:
```
mkdir data/raw data/processed notebooks src models reports reports/figures
```
- [ ] Copy original CSVs to `data/raw/`
- [ ] Create `src/__init__.py`
- [ ] Create `requirements.txt` (OSI-approved libraries only — Rules Section 6c):
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
- [ ] Install: `pip install -r requirements.txt`
- [ ] Create `submission_log.md` (to track daily submission count — max 10/day)

---

## Phase 5 — Data Splitting & Leak-Proof Pipeline

### Tasks
- [ ] Create `src/preprocessing.py` with pipeline builders
- [ ] Implement stratified split (preserve 1.5% fraud ratio)
- [ ] Build ColumnTransformer (fit on train only)
- [ ] Support both class-weight and SMOTE pipelines

### Key Code

```python
RANDOM_STATE = 42

# Column definitions (agreed with Member 1)
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
ORDINAL_COLS = ['risk_flags_count']

def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/validation split."""
    return train_test_split(X, y, test_size=test_size,
                           random_state=random_state, stratify=y)

def get_cv_strategy(n_splits=5, random_state=RANDOM_STATE):
    """Stratified K-Fold for cross-validation."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def build_preprocessor():
    """ColumnTransformer: scale numericals, encode categoricals, pass binaries."""
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(drop='first', sparse_output=False,
                              handle_unknown='ignore'), CATEGORICAL_COLS),
        ('bin', 'passthrough', BINARY_COLS),
        ('ord', 'passthrough', ORDINAL_COLS),
    ], remainder='drop')

def build_pipeline(model, use_smote=False):
    """Build complete preprocessing + model pipeline."""
    preprocessor = build_preprocessor()
    if use_smote:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        return ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)),
            ('classifier', model)
        ])
    else:
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
```

---

## Phase 6 — Feature Selection

### Tasks
- [ ] Create `src/feature_selection.py` with:
  - `compute_correlation_with_target(X, y)`
  - `compute_mutual_information(X, y)`
  - `compute_tree_importance(X, y)`
  - `compute_vif(X)`
  - `select_features(X, y)` → consensus ranking

---

## AFTER HANDOFF FROM MEMBER 1

---

## Phase 7 — Model Training & Comparison

### Tasks
- [ ] Create `src/train.py` with model definitions
- [ ] Create `src/evaluate.py` with metrics and threshold optimization
- [ ] Create `notebooks/03_model_training.ipynb`

### Models (all using open-source libraries — Rules Section 6c)

```python
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, scale_pos_weight=65, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, is_unbalance=True, random_state=42, verbose=-1),
    'SVM (RBF)': SVC(class_weight='balanced', probability=True, random_state=42),
}
```

### Evaluation Protocol
- **Primary metric**: `f1` (competition metric — Rules Section 3)
- CV: `StratifiedKFold(5, shuffle=True, random_state=42)`
- Also track: precision, recall, roc_auc (but select by F1)
- Generate: model comparison table, ROC curves, PR curves, confusion matrices

### Threshold Optimization
```python
def find_optimal_threshold(y_true, y_proba):
    """Find threshold maximizing F1-score (competition metric)."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1s)], max(f1s)
```

---

## Phase 8 — Hyperparameter Tuning

### Tasks
- [ ] Create `src/tune.py`
- [ ] Select top 2–3 models by **CV F1** (not accuracy)
- [ ] Run `RandomizedSearchCV(scoring='f1', cv=StratifiedKFold(5))`
- [ ] Optimize threshold after tuning

---

## Phase 9 — Final Model, Interpretability & Submissions

### Tasks

#### 9.1 Final Model Selection
- [ ] Select by **validation F1** (not training F1)
- [ ] Retrain on **full train.csv** (all 8000 rows) with best hyperparameters
- [ ] Apply optimal threshold

#### 9.2 Interpretability
- [ ] Feature importance (built-in) → bar chart
- [ ] Permutation importance (with `scoring='f1'`) → bar chart
- [ ] SHAP summary plot + dependence plots for top 3 features

#### 9.3 Dual Submission Strategy (Rules: 2 final selections)
- **Submission A (Aggressive)**: Best single model with optimal threshold
- **Submission B (Conservative)**: Ensemble or regularized model — safety pick
- Both saved as `submission.csv` and `submission_conservative.csv`

#### 9.4 Submission Validation (Before Uploading)
- [ ] Exactly 2 000 rows
- [ ] Columns: `transaction_id`, `is_fraud`
- [ ] IDs match `test.csv`
- [ ] `is_fraud` is binary (0 or 1)
- [ ] Predicted fraud rate is reasonable (~1–5%)
- [ ] **Check daily submission count < 10** (Rules Section 3)

#### 9.5 Submission Tracking
Update `submission_log.md` with every Kaggle submission:
```
| # | Date | Model | Threshold | Local Val F1 | Public LB F1 | Notes |
```

#### 9.6 Save Pipeline
```python
joblib.dump(final_pipeline, 'models/final_model.pkl')
joblib.dump(metadata, 'models/model_metadata.pkl')
```

---

## Phase 10 — Documentation & Deliverables (Winner Requirements)

### Tasks

#### README.md
- [ ] Project overview, setup, reproduction steps, results
- [ ] **Must enable full reproduction** (Rules Section 5)

#### reports/final_report.md (Winner Requirement — Rules Section 5)
> "Top-performing participants will be required to submit: Complete, reproducible source code and environment details. A brief overview of model methodology, data preprocessing, and training steps."

- [ ] Executive Summary
- [ ] Data Preprocessing Steps
- [ ] Feature Engineering (from Member 1)
- [ ] Model Development & Comparison
- [ ] Final Model: why selected, hyperparameters, feature importance, SHAP
- [ ] Submission Strategy
- [ ] Limitations

#### src/predict.py
- [ ] Load saved pipeline, predict from raw CSV, generate submission

---

## 🤝 What You Need from Member 1

| Item | When Needed |
|---|---|
| `src/feature_engineering.py` → `engineer_features()` | Before Phase 7 |
| `src/data_cleaning.py` → `prepare_for_modelling()` | Before Phase 7 |
| `data/processed/*.csv` (engineered) | Before Phase 7 |
| Column name lists (agreed above) | Before Phase 7 |
| `reports/figures/` (EDA plots) | Before Phase 10 |
| `reports/data_quality_report.md` | Before Phase 10 |

---

## Deliverables Checklist (Member 2 Owns)

- [ ] `src/preprocessing.py`
- [ ] `src/feature_selection.py`
- [ ] `src/train.py`
- [ ] `src/evaluate.py`
- [ ] `src/tune.py`
- [ ] `src/predict.py`
- [ ] `src/interpret.py`
- [ ] `notebooks/03_model_training.ipynb`
- [ ] `models/final_model.pkl`
- [ ] `models/model_metadata.pkl`
- [ ] `submission.csv` (Submission A — aggressive)
- [ ] `submission_conservative.csv` (Submission B — conservative)
- [ ] `submission_log.md` (track all uploads)
- [ ] `reports/model_results.md`
- [ ] `reports/final_report.md` (Winner Requirement)
- [ ] `requirements.txt` (OSI-approved libraries only)
- [ ] `README.md` (reproducible setup instructions)
