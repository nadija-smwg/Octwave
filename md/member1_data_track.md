# 🧑‍💻 Member 1 — Data Track (Phases 1–4)

## Scope
You own **data understanding, cleaning, EDA, and feature engineering**.  
Member 2 (Modelling Track) depends on your outputs.

---

## Timeline & Dependencies

```
Member 1 (Data)                         Member 2 (Modelling)
─────────────────                       ─────────────────────
Phase 1: Data Profiling  ───────────►   (waits / sets up project structure)
Phase 2: Data Cleaning   ───────────►   Phase 5: Pipeline architecture (parallel)
Phase 3: EDA             ───────────►   Phase 6: Feature selection helpers (parallel)
Phase 4: Feature Eng.    ──── HANDOFF ──► Phase 7-10: Training, Tuning, Final
```

**Your deliverables feed directly into Member 2's pipeline.**

---

## Phase 1 — Dataset Understanding & Profiling

### Your Tasks
- [ ] Create `src/data_profiling.py` with these functions:
  - `load_datasets(data_dir)` → returns dict `{'train': df, 'test': df, 'submission': df}`
  - `profile_dataframe(df, name)` → prints shape, dtypes, head, describe, nulls, unique counts
  - `check_data_quality(train_df, test_df)` → returns quality report dict
  - `compare_distributions(train_df, test_df)` → checks for distribution shift
  - `generate_quality_report(report_dict, output_path)` → writes `reports/data_quality_report.md`

- [ ] Create `notebooks/01_data_analysis.ipynb` with:
  - Load all 3 CSVs
  - Display profiling tables for each
  - Confirm: no missing, no duplicates, all types correct
  - Verify train/test have same columns (except `is_fraud`)
  - Verify test IDs match sample_submission IDs
  - Distribution comparison (KS test or overlapping histograms)

- [ ] Write `reports/data_quality_report.md`

### Key Facts (Already Discovered)
```
Train: 8000 rows × 10 cols (9 features + is_fraud)
Test:  2000 rows × 9 cols  (no target)
Missing: 0 everywhere
Duplicates: 0
Target: is_fraud (binary) — 7879 legit, 121 fraud (1.51%)
Categories: Food, Clothing, Travel, Electronics, Grocery (all clean)
```

### Output Files
```
src/data_profiling.py
notebooks/01_data_analysis.ipynb
reports/data_quality_report.md
data/raw/train.csv          (copy originals here)
data/raw/test.csv
data/raw/sample_submission.csv
```

---

## Phase 2 — Data Cleaning

### Your Tasks
- [ ] Create `src/data_cleaning.py` with these functions:
  - `validate_column_types(df)` → checks/fixes dtypes, returns warnings
  - `check_value_ranges(df)` → validates all values are in expected ranges
  - `clean_categorical_labels(df)` → `.str.strip().str.title()` safety step
  - `detect_outliers(df, columns)` → IQR-based outlier report (detect, don't remove)
  - `prepare_for_modelling(df)` → separates `(X, y, ids)`, drops `transaction_id`
  - `clean_pipeline(df)` → runs all above steps, returns cleaned df

- [ ] Document every decision in code comments:
  - Why outliers are KEPT (valid domain values)
  - Why `transaction_id` is dropped (identifier only)
  - Why `merchant_category` is converted to category dtype
  - Why no rows or columns are removed beyond `transaction_id`

- [ ] Save cleaned datasets

### Cleaning Decisions (Pre-Determined)
| Step | Action | Rows Affected | Reason |
|---|---|---|---|
| Missing values | None needed | 0 | No nulls |
| Duplicates | None needed | 0 | No dupes |
| Type fix | `merchant_category` → category | All rows | Proper dtype |
| Outliers | Keep all | 0 removed | Valid financial values |
| Drop column | `transaction_id` | All rows | ID only, stored separately |

### Output Files
```
src/data_cleaning.py
data/processed/train_cleaned.csv
data/processed/test_cleaned.csv
```

---

## Phase 3 — Exploratory Data Analysis (EDA)

### Your Tasks
- [ ] Create `src/eda.py` with plotting functions (use `matplotlib` + `seaborn`)
- [ ] Create `notebooks/02_eda.ipynb` with all visualizations
- [ ] Save all plots to `reports/figures/`

### Required Visualizations

#### 3.1 Target Distribution
- [ ] Bar chart: class 0 vs 1 counts with percentage labels
- [ ] Save as `reports/figures/target_distribution.png`

#### 3.2 Numerical Feature Distributions
- [ ] Histograms with KDE for: `amount`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `transaction_hour`
- [ ] Box plots of all numerical features (normalized scale)
- [ ] Save as `reports/figures/feature_distributions.png`

#### 3.3 Categorical Feature Distributions
- [ ] Bar chart of `merchant_category` counts
- [ ] Bar charts for `foreign_transaction`, `location_mismatch`
- [ ] Fraud rate by each categorical feature
- [ ] Save as `reports/figures/categorical_distributions.png`

#### 3.4 Correlation Analysis
- [ ] Heatmap of all numerical features + target
- [ ] Bar chart of |correlation with is_fraud| sorted
- [ ] Save as `reports/figures/correlation_heatmap.png`

#### 3.5 Feature-Target Deep Dive
- [ ] `transaction_hour` → fraud rate by hour (bar chart) — expect spike at 0–3 AM
- [ ] `device_trust_score` → KDE overlay by class — fraud clusters at 25–40
- [ ] `velocity_last_24h` → fraud rate by velocity
- [ ] `amount` → overlapping histograms by class
- [ ] `foreign_transaction` × `location_mismatch` → heatmap of fraud rates
- [ ] Save as `reports/figures/fraud_by_hour.png`, `fraud_by_device_trust.png`, etc.

#### 3.6 Outlier Visualization
- [ ] Box plots of each feature coloured by `is_fraud`
- [ ] Scatter: `amount` vs `device_trust_score` coloured by `is_fraud`

#### 3.7 Feature Interactions
- [ ] Pairplot of top 5 features coloured by `is_fraud`
- [ ] Fraud rate heatmap: `foreign_transaction` × `location_mismatch`

### Output Files
```
src/eda.py
notebooks/02_eda.ipynb
reports/figures/*.png (all plots)
```

---

## Phase 4 — Feature Engineering

### Your Tasks
- [ ] Create `src/feature_engineering.py` with:
  - `engineer_features(df)` → applies ALL feature engineering, returns enhanced df
  - Individual functions for each feature group
  - `get_feature_columns()` → returns lists of column names by type

### Features to Create

| # | Feature | Code | Justification |
|---|---|---|---|
| 1 | `log_amount` | `np.log1p(amount)` | Reduce right-skew of amount |
| 2 | `hour_sin` | `np.sin(2π × hour / 24)` | Cyclical encoding — 23h close to 0h |
| 3 | `hour_cos` | `np.cos(2π × hour / 24)` | Cyclical encoding — complement |
| 4 | `risk_flags_count` | `foreign + location_mismatch` | Additive risk tier (0, 1, 2) |
| 5 | `foreign_x_location` | `foreign × location_mismatch` | Both flags = highest risk |
| 6 | `high_velocity_low_trust` | `(velocity ≥ 4) & (trust ≤ 40)` | Compound fraud indicator |
| 7 | `late_night_flag` | `(hour ≥ 22) \| (hour ≤ 4)` | Peak fraud hours |
| 8 | `late_night_x_low_trust` | `late_night × (trust ≤ 40)` | Compound night + device risk |
| 9 | `amount_per_velocity` | `amount / (velocity + 1)` | Spending rate |
| 10 | `device_trust_bin` | `pd.cut(trust, [24,40,60,80,100])` | Interpretable trust tiers |

### Important Rules
- All transforms must be **stateless** (no fitting on data = no leakage)
- The `engineer_features()` function must work on BOTH train and test DataFrames
- Do NOT include scaling or encoding here — that goes in Member 2's Pipeline
- Keep `merchant_category` as string — encoding is in Member 2's ColumnTransformer

### Output Files
```
src/feature_engineering.py
data/processed/train_engineered.csv
data/processed/test_engineered.csv
```

---

## 🤝 HANDOFF to Member 2

When you finish Phase 4, Member 2 needs these from you:

1. **`src/feature_engineering.py`** — the `engineer_features(df)` function
2. **`src/data_cleaning.py`** — the `clean_pipeline(df)` and `prepare_for_modelling(df)` functions
3. **`data/processed/train_engineered.csv`** and **`data/processed/test_engineered.csv`**
4. **Feature column lists**:
   - Numerical columns to scale: `['log_amount', 'device_trust_score', 'velocity_last_24h', 'cardholder_age', 'amount_per_velocity', 'hour_sin', 'hour_cos']`
   - Binary columns (passthrough): `['foreign_transaction', 'location_mismatch', 'high_velocity_low_trust', 'late_night_flag', 'late_night_x_low_trust', 'foreign_x_location']`
   - Categorical columns to encode: `['merchant_category']`
   - Ordinal/integer columns: `['risk_flags_count', 'device_trust_bin']`
5. **EDA insights** that affect modelling:
   - Severe imbalance (65:1) — need class weights / SMOTE
   - `cardholder_age` has zero signal — candidate for removal
   - `device_trust_score` and `transaction_hour` are the strongest features
   - `foreign_transaction` × `location_mismatch` interaction is critical

---

## Checklist Before Handoff

- [ ] All `src/` modules have docstrings and comments
- [ ] `data/processed/` has cleaned and engineered CSVs
- [ ] `reports/figures/` has all EDA plots
- [ ] `reports/data_quality_report.md` is complete
- [ ] `notebooks/01_data_analysis.ipynb` runs end-to-end
- [ ] `notebooks/02_eda.ipynb` runs end-to-end
- [ ] `engineer_features()` works on both train.csv and test.csv
- [ ] No data leakage — all transforms are stateless
