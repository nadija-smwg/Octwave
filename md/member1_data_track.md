# 🧑‍💻 Member 1 — Data Track (Phases 1–4)

## Scope
You own **data understanding, cleaning, EDA, and feature engineering**.  
Member 2 (Modelling Track) depends on your outputs.

---

## Competition Rules Relevant to You

| Rule | Impact |
|---|---|
| **No external data** (Section 4) | Only use `train.csv`, `test.csv`, `sample_submission.csv` — no external fraud datasets, no pre-built features |
| **No hand-labeling** (Section 4b) | Do not manually inspect test data to guess labels |
| **F1-score metric** (Section 3) | Focus EDA on understanding precision/recall trade-offs, not just accuracy |
| **Reproducible code** (Section 5) | All code must have `random_state=42`, clear comments, and run end-to-end |
| **Open source only** (Section 6c) | Only use OSI-approved libraries (pandas, numpy, matplotlib, seaborn, sklearn) |

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

---

## Phase 1 — Dataset Understanding & Profiling

### Tasks
- [ ] Create `src/data_profiling.py` with:
  - `load_datasets(data_dir)` → returns dict `{'train': df, 'test': df, 'submission': df}`
  - `profile_dataframe(df, name)` → prints shape, dtypes, head, describe, nulls, unique counts
  - `check_data_quality(train_df, test_df)` → returns quality report dict
  - `compare_distributions(train_df, test_df)` → checks for distribution shift
  - `generate_quality_report(report_dict, output_path)` → writes `reports/data_quality_report.md`

- [ ] Create `notebooks/01_data_analysis.ipynb` with:
  - Load all 3 CSVs **from `data/raw/` only** (no external sources — Rules Section 4)
  - Display profiling tables for each
  - Confirm: no missing, no duplicates, all types correct
  - Verify train/test have same columns (except `is_fraud`)
  - Verify test IDs match sample_submission IDs
  - Distribution comparison (KS test or overlapping histograms)

- [ ] Write `reports/data_quality_report.md`

### Key Facts
```
Train: 8000 rows × 10 cols (9 features + is_fraud)
Test:  2000 rows × 9 cols  (no target)
Missing: 0 everywhere
Duplicates: 0
Target: is_fraud (binary) — 7879 legit, 121 fraud (1.51%)
Metric: F1-score (Rules Section 3)
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

### Tasks
- [ ] Create `src/data_cleaning.py` with:
  - `validate_column_types(df)` → checks/fixes dtypes
  - `check_value_ranges(df)` → validates all values in expected ranges
  - `clean_categorical_labels(df)` → `.str.strip().str.title()` safety step
  - `detect_outliers(df, columns)` → IQR-based outlier report (detect, don't remove)
  - `prepare_for_modelling(df)` → separates `(X, y, ids)`, drops `transaction_id`
  - `clean_pipeline(df)` → runs all steps, returns cleaned df

- [ ] Document every decision in code comments

### Cleaning Decisions
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

### Tasks
- [ ] Create `src/eda.py` with plotting functions
- [ ] Create `notebooks/02_eda.ipynb` with all visualizations
- [ ] Save all plots to `reports/figures/`

### Required Visualizations

#### 3.1 Target Distribution
- [ ] Bar chart: class 0 vs 1 counts with percentage labels
- [ ] Highlight: **65:1 imbalance ratio** — this is why competition uses F1, not accuracy

#### 3.2 Numerical Feature Distributions
- [ ] Histograms with KDE for: `amount`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`, `transaction_hour`
- [ ] Box plots of all numerical features

#### 3.3 Categorical Feature Distributions
- [ ] Bar chart of `merchant_category` counts
- [ ] Fraud rate by each categorical feature
- [ ] `foreign_transaction` and `location_mismatch` distributions

#### 3.4 Correlation Analysis
- [ ] Heatmap of all numerical features + target
- [ ] Bar chart of |correlation with is_fraud| sorted

#### 3.5 Feature-Target Deep Dive
- [ ] `transaction_hour` → fraud rate by hour (expect spike at 0–3 AM)
- [ ] `device_trust_score` → KDE overlay by class (fraud clusters at 25–40)
- [ ] `velocity_last_24h` → fraud rate by velocity
- [ ] `amount` → overlapping histograms by class
- [ ] `foreign_transaction` × `location_mismatch` → fraud rate heatmap

#### 3.6 Outlier & Interaction Visualization
- [ ] Box plots coloured by `is_fraud`
- [ ] Scatter: `amount` vs `device_trust_score` coloured by fraud
- [ ] Pairplot of top 5 features

### Output Files
```
src/eda.py
notebooks/02_eda.ipynb
reports/figures/*.png
```

---

## Phase 4 — Feature Engineering

### Tasks
- [ ] Create `src/feature_engineering.py` with:
  - `engineer_features(df)` → applies ALL feature engineering, returns enhanced df
  - `get_feature_columns()` → returns lists of column names by type

### Features to Create

| # | Feature | Code | Justification |
|---|---|---|---|
| 1 | `log_amount` | `np.log1p(amount)` | Reduce right-skew |
| 2 | `hour_sin` | `np.sin(2π × hour / 24)` | Cyclical encoding |
| 3 | `hour_cos` | `np.cos(2π × hour / 24)` | Cyclical complement |
| 4 | `risk_flags_count` | `foreign + location_mismatch` | Additive risk tier |
| 5 | `foreign_x_location` | `foreign × location_mismatch` | Both flags = highest risk |
| 6 | `high_velocity_low_trust` | `(velocity ≥ 4) & (trust ≤ 40)` | Compound indicator |
| 7 | `late_night_flag` | `(hour ≥ 22) \| (hour ≤ 4)` | Peak fraud hours |
| 8 | `late_night_x_low_trust` | `late_night × (trust ≤ 40)` | Compound night risk |
| 9 | `amount_per_velocity` | `amount / (velocity + 1)` | Spending rate |
| 10 | `device_trust_bin` | `pd.cut(trust, [24,40,60,80,100])` | Trust tiers |

### Rules Compliance for Feature Engineering
- ⚠️ **No external data**: All features derived from provided columns only
- ✅ **All transforms are stateless**: No fitting on data → no leakage
- ✅ **`engineer_features()` works on both train and test DataFrames**
- ⚠️ Do NOT include scaling or encoding here — that goes in Member 2's Pipeline

### Output Files
```
src/feature_engineering.py
data/processed/train_engineered.csv
data/processed/test_engineered.csv
```

---

## 🤝 HANDOFF to Member 2

When you finish Phase 4, Member 2 needs:

1. **`src/feature_engineering.py`** — the `engineer_features(df)` function
2. **`src/data_cleaning.py`** — the `clean_pipeline(df)` and `prepare_for_modelling(df)` functions
3. **`data/processed/train_engineered.csv`** and **`data/processed/test_engineered.csv`**
4. **Feature column lists** (exact names, agreed with Member 2):
   - Numerical: `['log_amount', 'device_trust_score', 'velocity_last_24h', 'cardholder_age', 'amount_per_velocity', 'hour_sin', 'hour_cos']`
   - Binary: `['foreign_transaction', 'location_mismatch', 'high_velocity_low_trust', 'late_night_flag', 'late_night_x_low_trust', 'foreign_x_location']`
   - Categorical: `['merchant_category']`
   - Ordinal: `['risk_flags_count', 'device_trust_bin']`
5. **EDA insights**: Imbalance ratio, top features, fraud profile

---

## Checklist Before Handoff

- [ ] All `src/` modules have docstrings, comments, and `random_state=42`
- [ ] `data/processed/` has cleaned and engineered CSVs
- [ ] `reports/figures/` has all EDA plots
- [ ] `reports/data_quality_report.md` is complete
- [ ] Notebooks run end-to-end
- [ ] `engineer_features()` works on both train and test
- [ ] No external data used anywhere
- [ ] No data leakage — all transforms are stateless
