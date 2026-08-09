# Phase 3 — Exploratory Data Analysis (EDA)

## Objective
Generate comprehensive visualizations and statistical analyses to understand data patterns, feature-target relationships, and guide feature engineering and model selection.

---

## 3.1 Target Variable Distribution

### Visualizations
- [ ] **Bar chart** of class counts (0 vs 1) with percentage labels
- [ ] **Pie chart** showing the 98.5% / 1.5% split
- [ ] Annotate the imbalance ratio (~65:1)

### Key Insight
With only 121 fraud cases out of 8 000, a naive classifier predicting all-0 achieves 98.5% accuracy but 0% recall. This confirms F1-score is the right metric.

---

## 3.2 Numerical Feature Distributions

### Visualizations (for each numerical feature)
- [ ] **Histogram with KDE** overlay — `amount`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `transaction_hour`
- [ ] **Box plots** — all numerical features side by side (normalized)
- [ ] **Violin plots** — numerical features split by `is_fraud`

### Features to plot:
| Feature | Expected Distribution | Notes |
|---|---|---|
| `amount` | Right-skewed (mean 175, median 121) | Long tail; log transform candidate |
| `transaction_hour` | Roughly uniform 0–23 | Cyclical |
| `device_trust_score` | Roughly uniform 25–99 | |
| `velocity_last_24h` | Right-skewed, discrete (0–9) | Most values 0–4 |
| `cardholder_age` | Roughly uniform 18–69 | |

---

## 3.3 Categorical Feature Distributions

### Visualizations
- [ ] **Bar chart** of `merchant_category` counts
- [ ] **Stacked/grouped bar chart** of `merchant_category` × `is_fraud`
- [ ] **Bar charts** for `foreign_transaction` and `location_mismatch` distributions
- [ ] **Fraud rate bar chart** by each categorical feature

### Key Categories:
- `merchant_category`: 5 balanced categories (~1 550–1 670 each)
- `foreign_transaction`: ~90% domestic, ~10% foreign
- `location_mismatch`: ~91.5% match, ~8.5% mismatch

---

## 3.4 Correlation Analysis

### Visualizations
- [ ] **Heatmap** of Pearson correlations (all numerical features + target)
- [ ] **Bar chart** of correlations with `is_fraud` sorted by absolute value
- [ ] **Point-biserial correlation** for binary features vs target

### Expected Strong Correlations with `is_fraud`:
| Feature | Correlation | Strength |
|---|---|---|
| `foreign_transaction` | +0.179 | Moderate |
| `location_mismatch` | +0.168 | Moderate |
| `device_trust_score` | −0.138 | Moderate |
| `transaction_hour` | −0.135 | Moderate |
| `velocity_last_24h` | +0.110 | Weak-Moderate |
| `amount` | +0.034 | Very Weak |
| `cardholder_age` | +0.000 | None |

---

## 3.5 Feature-Target Relationships (Deep Dive)

### Visualizations for each feature vs `is_fraud`:
- [ ] **`amount`**: Overlapping histograms (fraud vs legit), box plot by class
- [ ] **`transaction_hour`**: Fraud rate by hour bar chart — expect spike at 0–3 AM
- [ ] **`device_trust_score`**: KDE by class — fraud clusters at low scores (25–40)
- [ ] **`velocity_last_24h`**: Fraud rate by velocity — expect monotonic increase
- [ ] **`cardholder_age`**: Box plot by class — expect no difference
- [ ] **`foreign_transaction`**: Fraud rate comparison (0.8% vs 8.1%)
- [ ] **`location_mismatch`**: Fraud rate comparison (0.9% vs 8.2%)
- [ ] **`merchant_category`**: Fraud rate by category bar chart

### Key Fraud Profile:
```
Typical fraudulent transaction:
├── Hour: 0–3 AM (median=2 vs non-fraud median=12)
├── Device trust: Low (median=32 vs non-fraud median=62)
├── Foreign: Yes (8.1% fraud rate vs 0.8%)
├── Location mismatch: Yes (8.2% fraud rate vs 0.9%)
├── Velocity: Higher (mean=3.3 vs 2.0)
├── Amount: Slightly higher but very noisy
└── Age: No pattern
```

---

## 3.6 Outlier Analysis

### Visualizations
- [ ] **Box plots** of each numerical feature, coloured by `is_fraud`
- [ ] **Scatter plot**: `amount` vs `device_trust_score` coloured by `is_fraud`
- [ ] **IQR-based outlier table** with counts

### Decision
All outliers are valid domain values — do not remove.

---

## 3.7 Class Imbalance Analysis

### Visualizations
- [ ] Class balance bar chart with exact counts
- [ ] **Comparison table** of metrics if using accuracy vs F1
- [ ] Document planned imbalance handling strategies:
  - Class weights in models
  - SMOTE on training data only
  - Threshold tuning
  - Stratified sampling for cross-validation

---

## 3.8 Feature Interaction Analysis

### Visualizations
- [ ] **Scatter matrix** (pairplot) of top features coloured by `is_fraud`
- [ ] **Heatmap** showing fraud rate for `foreign_transaction` × `location_mismatch` combinations
- [ ] **2D density plots** for key feature pairs

### Key Interactions to Investigate:
| Pair | Hypothesis |
|---|---|
| `foreign_transaction` × `location_mismatch` | Both = 1 → very high fraud rate |
| `transaction_hour` × `device_trust_score` | Late night + low trust → fraud |
| `velocity_last_24h` × `device_trust_score` | High velocity + low trust → fraud |
| `amount` × `foreign_transaction` | High amount + foreign → fraud? |

---

## 3.9 Redundancy & Multicollinearity Check

### Analysis
- [ ] **VIF (Variance Inflation Factor)** for all numerical features
- [ ] Check if `foreign_transaction` and `location_mismatch` are highly correlated (both are fraud flags)
- [ ] Identify any features that can be safely removed without information loss

### Expected Result
No strong multicollinearity — features represent distinct concepts.

---

## 3.10 Code Structure for Phase 3

```
src/eda.py
    ├── plot_target_distribution(df, save_dir)
    ├── plot_numerical_distributions(df, save_dir)
    ├── plot_categorical_distributions(df, save_dir)
    ├── plot_correlation_analysis(df, save_dir)
    ├── plot_feature_target_relationships(df, save_dir)
    ├── plot_outlier_analysis(df, save_dir)
    ├── plot_interaction_analysis(df, save_dir)
    ├── compute_vif(df) → VIF table
    └── generate_eda_summary(df) → summary dict
```

### Saved Artifacts
- All plots saved to `reports/figures/` directory
- EDA summary in `reports/eda_summary.md`
