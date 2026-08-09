# Phase 2 — Data Cleaning

## Objective
Systematically clean the training and test datasets while documenting every decision. Preserve all useful information.

---

## 2.1 Missing Values

### Finding
**No missing values** in either train or test datasets — all 10 columns are fully populated.

### Action
- No imputation required
- Document this as a positive finding in the quality report
- Still include imputation steps in the sklearn Pipeline for production robustness (handles unseen data)

---

## 2.2 Duplicate Detection

### Finding
- **0 duplicate rows** in train
- **0 duplicate `transaction_id`** values in train
- Same expected for test

### Action
- No deduplication needed
- Add assertion checks in code to catch duplicates if data changes

---

## 2.3 Data Type Validation

| Column | Current Type | Expected Type | Status | Action |
|---|---|---|---|---|
| `transaction_id` | int64 | int | ✅ Correct | — |
| `amount` | float64 | float | ✅ Correct | — |
| `transaction_hour` | int64 | int (0–23) | ✅ Correct | — |
| `merchant_category` | object/str | categorical | ⚠️ Convert | Convert to `category` dtype |
| `foreign_transaction` | int64 | bool/binary | ✅ Acceptable | Keep as int for modelling |
| `location_mismatch` | int64 | bool/binary | ✅ Acceptable | Keep as int for modelling |
| `device_trust_score` | int64 | int | ✅ Correct | — |
| `velocity_last_24h` | int64 | int | ✅ Correct | — |
| `cardholder_age` | int64 | int | ✅ Correct | — |
| `is_fraud` | int64 | binary | ✅ Correct | — |

---

## 2.4 Invalid / Impossible Value Checks

| Column | Validation Rule | Status |
|---|---|---|
| `amount` | ≥ 0 | ✅ Min = 0.00 |
| `transaction_hour` | 0 – 23 | ✅ Range correct |
| `merchant_category` | 5 known categories | ✅ Clean |
| `foreign_transaction` | 0 or 1 only | ✅ Clean |
| `location_mismatch` | 0 or 1 only | ✅ Clean |
| `device_trust_score` | Reasonable positive range | ✅ 25–99 |
| `velocity_last_24h` | Non-negative integer | ✅ 0–9 |
| `cardholder_age` | Reasonable adult age | ✅ 18–69 |

---

## 2.5 Categorical Label Consistency

### `merchant_category` — 5 labels in both train and test:
- `Food`, `Clothing`, `Travel`, `Electronics`, `Grocery`
- **No spelling variations, no mixed case, no trailing spaces**
- **Same 5 labels in both train and test** ✅

### Action
- No label cleaning required
- Add `.str.strip().str.title()` as a safety step in pipeline

---

## 2.6 Outlier Investigation

| Feature | IQR Method Outliers? | Investigation |
|---|---|---|
| `amount` | Yes — right tail (high transactions) | **Keep**: High-value transactions are valid and may be fraud signals |
| `device_trust_score` | No significant outliers | Range 25–99 is well-bounded |
| `velocity_last_24h` | Mild — values 7–9 are rare | **Keep**: High velocity is a strong fraud signal |
| `cardholder_age` | No | Range 18–69 is reasonable |
| `transaction_hour` | No | Bounded 0–23 |

### Decision
**Do NOT remove any outliers.** All values are plausible for financial transactions. High/extreme values may carry fraud signal. Removing them would lose information and potentially bias the model.

---

## 2.7 Column Handling Decisions

| Column | Decision | Justification |
|---|---|---|
| `transaction_id` | **Drop before modelling** | Sequential identifier; no predictive value; needed only for submission mapping |
| All other columns | **Keep** | All carry potential predictive signal |

---

## 2.8 Cleaning Summary

| Step | Rows Removed | Columns Removed | Transformations |
|---|---|---|---|
| Missing values | 0 | 0 | None needed |
| Duplicates | 0 | 0 | None needed |
| Type fixes | 0 | 0 | `merchant_category` → category |
| Invalid values | 0 | 0 | None found |
| Outliers | 0 | 0 | Kept (valid domain values) |
| ID columns | 0 | 1 (`transaction_id`) | Stored separately for submission |

> **Net result**: Clean dataset with **8 000 rows × 9 columns** (8 features + 1 target).  
> The data is remarkably clean — this is a simulated/synthetic dataset.

---

## 2.9 Code Structure for Phase 2

```
src/data_cleaning.py
    ├── validate_column_types(df) → warnings/fixes
    ├── check_value_ranges(df) → validation report
    ├── clean_categorical_labels(df) → cleaned df
    ├── handle_missing_values(df) → cleaned df (passthrough here)
    ├── detect_outliers(df, columns) → outlier report
    ├── prepare_for_modelling(df) → (features_df, target_series, id_series)
    └── clean_pipeline(df) → fully cleaned df
```

---

## 2.10 Saved Artifacts

- `data/processed/train_cleaned.csv`
- `data/processed/test_cleaned.csv`
- Cleaning decisions documented in `reports/data_quality_report.md`
