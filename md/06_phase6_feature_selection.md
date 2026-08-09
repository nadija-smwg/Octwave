# Phase 6 — Feature Selection

## Objective
Evaluate feature usefulness and remove noise features that could hurt model performance or add unnecessary complexity.

---

## 6.1 Correlation-Based Filtering

### Feature-Target Correlations (already computed)
| Feature | |Correlation with is_fraud| | Action |
|---|---|---|
| `foreign_transaction` | 0.179 | Keep — strong signal |
| `location_mismatch` | 0.168 | Keep — strong signal |
| `device_trust_score` | 0.138 | Keep — strong signal |
| `transaction_hour` | 0.135 | Keep — strong signal (replaced by cyclical) |
| `velocity_last_24h` | 0.110 | Keep — moderate signal |
| `amount` | 0.034 | Keep — weak but potentially useful in interactions |
| `cardholder_age` | 0.000 | **Candidate for removal** — no linear signal |

### Inter-Feature Correlations
- Check correlation between `foreign_transaction` and `location_mismatch` — both are binary risk flags, may overlap
- Check if engineered features are redundant with source features
- Threshold for concern: |r| > 0.8

---

## 6.2 Variance Filtering

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
```

- All features have reasonable variance
- Binary features `foreign_transaction` (10% positive) and `location_mismatch` (8.5% positive) have variance ~0.09 and ~0.08 — well above any practical threshold
- **No features expected to be removed**

---

## 6.3 Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
```

- Captures non-linear relationships (unlike Pearson)
- `cardholder_age` may show low MI even though Pearson = 0
- Engineered features like `late_night_flag` may show higher MI than raw `transaction_hour`
- Rank all features by MI and visualise as bar chart

---

## 6.4 Feature Importance from Tree Models

```python
# Train a quick RandomForest and extract importances
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_processed, y_train)
importances = rf.feature_importances_
```

- Tree-based importance captures complex interactions
- Compare with correlation and MI rankings
- Features consistently ranked low across all methods are candidates for removal

---

## 6.5 Multicollinearity Analysis (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

| Feature | Expected VIF | Concern? |
|---|---|---|
| `amount` / `log_amount` | Low | No — distinct from other features |
| `hour_sin` / `hour_cos` | ~1.0 | No — orthogonal by construction |
| `device_trust_score` | Low | No |
| `velocity_last_24h` | Low | No |
| `foreign_transaction` | Low-Moderate | Possible overlap with `risk_flags_count` |
| `location_mismatch` | Low-Moderate | Possible overlap with `risk_flags_count` |
| `risk_flags_count` | Moderate | Derived from foreign + location — expected |

### Decision Rule
- VIF > 10: Investigate for removal
- VIF > 5: Flag but keep if feature importance is high
- VIF < 5: Keep

---

## 6.6 Recursive Feature Elimination (optional)

```python
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=LogisticRegression(class_weight='balanced'),
              step=1, cv=StratifiedKFold(5), scoring='f1')
```

- Run only if the feature set exceeds ~20 features
- With ~15-18 features total, RFE may be overkill
- **Decision**: Run RFECV if initial models show signs of overfitting

---

## 6.7 Feature Selection Decision Matrix

| Feature | Correlation | MI | Tree Importance | VIF | Final Decision |
|---|---|---|---|---|---|
| `foreign_transaction` | ✅ High | TBD | TBD | TBD | **Keep** |
| `location_mismatch` | ✅ High | TBD | TBD | TBD | **Keep** |
| `device_trust_score` | ✅ High | TBD | TBD | TBD | **Keep** |
| `hour_sin/cos` | ✅ High | TBD | TBD | TBD | **Keep** |
| `velocity_last_24h` | ✅ Moderate | TBD | TBD | TBD | **Keep** |
| `amount` / `log_amount` | ⚠️ Weak | TBD | TBD | TBD | **Keep** (interactions) |
| `cardholder_age` | ❌ None | TBD | TBD | TBD | **Evaluate** |
| `risk_flags_count` | TBD | TBD | TBD | TBD | **Evaluate** (redundancy) |
| Engineered features | TBD | TBD | TBD | TBD | **Evaluate** |

---

## 6.8 Code Structure for Phase 6

```
src/feature_selection.py
    ├── compute_correlation_with_target(X, y) → correlation series
    ├── compute_mutual_information(X, y) → MI scores
    ├── compute_tree_importance(X, y) → importance dict
    ├── compute_vif(X) → VIF table
    ├── run_rfecv(X, y, estimator, cv) → selected features
    ├── select_features(X, y, method='all') → selected feature list
    └── plot_feature_importance_comparison(results, save_path)
```
