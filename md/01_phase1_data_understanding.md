# Phase 1 — Dataset Understanding & Data Quality Report

## Objective
Load, inspect, and profile all CSV datasets. Generate a formal data-quality report and confirm the ML task definition.

---

## Rules Compliance Check

| Rule | How We Comply |
|---|---|
| No external data | Only using `train.csv`, `test.csv`, `sample_submission.csv` |
| No hand-labeling test data | Test set used only for final prediction, never inspected for labels |
| Evaluation = F1-score | All analysis focuses on F1-relevant insights (class imbalance, precision/recall trade-offs) |

---

## 1.1 Load & Initial Inspection

### Tasks
- [ ] Load `train.csv`, `test.csv`, and `sample_submission.csv` using pandas
- [ ] Display `.shape`, `.columns`, `.dtypes`, `.head()`, `.describe()` for each file
- [ ] Verify `transaction_id` uniqueness in both train and test
- [ ] Confirm test IDs match sample_submission IDs
- [ ] Confirm train and test have no overlapping `transaction_id` values

### Expected Outputs
| Dataset | Rows | Feature Columns | Target |
|---|---|---|---|
| train.csv | 8 000 | 9 | `is_fraud` |
| test.csv | 2 000 | 9 | — (to predict) |
| sample_submission.csv | 2 000 | 2 | template |

---

## 1.2 Feature Profiling

| Feature | Type | Missing | Unique | Range | Notes |
|---|---|---|---|---|---|
| `transaction_id` | int | 0 | 8000/2000 | 1–10000 | ID only — drop before modelling |
| `amount` | float | 0 | 7216 | 0.00–1390.24 (train) | Right-skewed; consider log transform |
| `transaction_hour` | int | 0 | 24 | 0–23 | Cyclical; consider sin/cos encoding |
| `merchant_category` | str | 0 | 5 | 5 categories | Balanced distribution; one-hot or target encode |
| `foreign_transaction` | binary | 0 | 2 | 0/1 | Strong fraud signal (corr +0.179) |
| `location_mismatch` | binary | 0 | 2 | 0/1 | Strong fraud signal (corr +0.168) |
| `device_trust_score` | int | 0 | 75 | 25–99 | Strong negative fraud signal (corr −0.138) |
| `velocity_last_24h` | int | 0 | 10 | 0–9 | Moderate fraud signal (corr +0.110) |
| `cardholder_age` | int | 0 | 52 | 18–69 | Near-zero correlation with fraud |

---

## 1.3 Target Variable Analysis

- **Target**: `is_fraud` (binary: 0/1)
- **Task**: Binary Classification
- **Evaluation Metric**: F1-score (competition-specified in rules Section 3)
- **Class Distribution**:
  - Legitimate (0): 7 879 samples (98.49%)
  - Fraudulent (1): 121 samples (1.51%)
  - **Imbalance ratio: ~65:1**

---

## 1.4 Dataset Relationship Analysis

- `train.csv` and `test.csv` share identical feature schema (test lacks `is_fraud`)
- `sample_submission.csv` maps test `transaction_id` → predicted `is_fraud`
- **No merge needed** — standard train/test competition split
- Verify distribution consistency between train and test (detect distribution shift)

---

## 1.5 Data Quality Report

### Checks to perform:
- [ ] Missing value counts per column (train & test)
- [ ] Duplicate row detection
- [ ] Duplicate ID detection
- [ ] Data type consistency between train and test
- [ ] Value range consistency between train and test
- [ ] Categorical label consistency between train and test
- [ ] Statistical distribution comparison (KS test or visual)

### Output
- `reports/data_quality_report.md`

---

## 1.6 Code Structure

```
src/data_profiling.py
    ├── load_datasets(data_dir) → dict of DataFrames
    ├── profile_dataframe(df, name) → prints/returns profiling info
    ├── check_data_quality(train_df, test_df) → quality report dict
    ├── compare_distributions(train_df, test_df) → distribution comparison
    └── generate_quality_report(report_dict, output_path) → writes .md
```

---

## 1.7 Key Decisions & Assumptions

| Decision | Rationale |
|---|---|
| Target = `is_fraud` | Explicitly stated in competition description and rules |
| Task = Binary Classification | Target is binary (0/1) |
| Primary metric = F1-score | Competition rules Section 3: "scored using the F1-score metric" |
| `transaction_id` = ID column (not a feature) | Sequential identifier with no predictive value |
| No external data used | Competition rules Section 4: "strictly prohibited" |
| No dataset merging required | Standard train/test split format |
