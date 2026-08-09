# Phase 7 — Model Training & Comparison

## Objective
Train multiple appropriate models, compare performance using cross-validation and the validation set, and identify the best candidates for hyperparameter tuning.

---

## Rules Compliance

| Rule | How We Comply |
|---|---|
| F1-score evaluation | All CV and validation uses `scoring='f1'` as primary metric |
| No external data | All models trained only on provided `train.csv` |
| Open source only | Using sklearn, xgboost, lightgbm — all OSI-approved |
| Reproducible code required | `random_state=42` everywhere; all code in `src/` modules |
| No hand-labeling test data | Test set never used during training or validation |
| Max 10 submissions/day | Validate locally first; only submit when confident |

---

## 7.1 Model Selection Rationale

### Dataset Characteristics
| Property | Value | Implication |
|---|---|---|
| Training size | ~6 400 (after 80/20 split) | Medium — all models viable |
| Features | ~15–18 (after engineering) | Low-dimensional — no dimensionality issues |
| Target | Binary (1.5% positive) | Severe imbalance — need class weights / SMOTE |
| Feature types | Mixed (numerical + categorical + binary) | Tree models handle natively; linear models need encoding |
| Evaluation metric | **F1-score** (competition rules) | Must optimize for precision-recall balance, not accuracy |

### Selected Models

| Model | Why Selected | Imbalance Handling |
|---|---|---|
| **Logistic Regression** | Baseline; interpretable; fast; works well with few features | `class_weight='balanced'` |
| **Random Forest** | Strong ensemble; handles mixed features; robust | `class_weight='balanced'` |
| **Gradient Boosting** | Top performer for tabular data; handles imbalance | `sample_weight` via class weights |
| **XGBoost** | Industry standard for competitions; built-in `scale_pos_weight` | `scale_pos_weight ≈ 65` |
| **LightGBM** | Fast, handles categorical features natively | `is_unbalance=True` |
| **SVM (RBF kernel)** | Good with few features; captures non-linear boundaries | `class_weight='balanced'` |

### Models NOT Selected (and why)
| Model | Reason for Exclusion |
|---|---|
| KNN | Sensitive to imbalanced data; 1.5% positive rate makes it impractical |
| Naive Bayes | Feature independence assumption violated (interaction features) |
| Decision Tree (alone) | Prone to overfitting; Random Forest is strictly better |
| Neural Networks | Dataset too small (8000 rows); overfitting risk; harder to reproduce |
| CatBoost | Redundant with XGBoost/LightGBM for this dataset size |

---

## 7.2 Training Protocol

### Step 1: Cross-Validation on Training Set
```python
from sklearn.model_selection import cross_validate

scoring = {
    'f1': 'f1',              # PRIMARY — matches competition metric
    'precision': 'precision',
    'recall': 'recall',
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy'   # Secondary only — do NOT select models by accuracy
}

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring=scoring,
    return_train_score=True
)
```

### Step 2: Fit on Full Training Set → Evaluate on Validation Set
```python
pipeline.fit(X_train, y_train)
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)[:, 1]
```

### Step 3: Compute Comprehensive Metrics
```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
```

---

## 7.3 Class Imbalance Strategies

### Strategy A: Class Weights (Primary)
```python
# Logistic Regression, SVM, Random Forest
model = LogisticRegression(class_weight='balanced')

# XGBoost
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])  # ~65
model = XGBClassifier(scale_pos_weight=scale_pos_weight)
```

### Strategy B: SMOTE (on training data only)
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
    ('classifier', model)
])
```

### Strategy C: Threshold Tuning (Post-Training)
```python
# Optimize classification threshold on VALIDATION set (not test)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

### Plan
1. Train each model with **class weights first** (simpler, no resampling)
2. Compare with **SMOTE variants** for the top 2–3 models
3. Apply **threshold tuning** to the final selected model
4. **Only submit to Kaggle when local val F1 is strong** (preserve daily submission quota)

---

## 7.4 Model Comparison Table

| Model | CV F1 (mean±std) | Val F1 | Val Precision | Val Recall | Val ROC-AUC | Train Time |
|---|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | <1s |
| Random Forest | TBD | TBD | TBD | TBD | TBD | ~2s |
| Gradient Boosting | TBD | TBD | TBD | TBD | TBD | ~3s |
| XGBoost | TBD | TBD | TBD | TBD | TBD | ~2s |
| LightGBM | TBD | TBD | TBD | TBD | TBD | ~1s |
| SVM (RBF) | TBD | TBD | TBD | TBD | TBD | ~5s |

### Selection Criteria for Tuning (prioritised)
1. **Primary**: Highest **CV F1** (mean) — matches competition metric
2. **Secondary**: Low gap between train and CV F1 (no overfitting)
3. **Tertiary**: Validation F1 confirms CV ranking
4. Select **top 2–3 models** for hyperparameter tuning

---

## 7.5 Confusion Matrix Analysis

For each model:
```
              Predicted 0   Predicted 1
Actual 0         TN            FP
Actual 1         FN            TP
```

For **F1-score optimization**:
- F1 = 2 × (Precision × Recall) / (Precision + Recall)
- Need **balance between catching fraud (Recall) and being correct when flagging (Precision)**
- Too many FP → low Precision → low F1
- Too many FN → low Recall → low F1

---

## 7.6 Code Structure

```
src/train.py
    ├── get_models(scale_pos_weight) → dict of {name: pipeline}
    ├── cross_validate_model(pipeline, X, y, cv, scoring) → results dict
    ├── train_and_evaluate(pipeline, X_train, y_train, X_val, y_val) → metrics
    ├── compare_models(results_dict) → comparison DataFrame
    ├── plot_model_comparison(comparison_df, save_path)
    └── plot_confusion_matrices(models, X_val, y_val, save_path)

src/evaluate.py
    ├── compute_classification_metrics(y_true, y_pred, y_proba) → metrics dict
    ├── plot_roc_curves(models, X_val, y_val, save_path)
    ├── plot_precision_recall_curves(models, X_val, y_val, save_path)
    └── find_optimal_threshold(y_true, y_proba) → optimal_threshold
```
