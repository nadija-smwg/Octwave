# OctWave 3.0 — Credit Card Fraud Detection: Competition Overview

## 1. Competition Goal

Build a **binary classification** model to detect fraudulent credit card transactions.  
**Evaluation metric**: **F1-score** (chosen because of severe class imbalance).

---

## 2. Dataset Summary

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `train.csv` | 8 000 | 10 (9 features + target) | Model training & validation |
| `test.csv` | 2 000 | 9 (features only) | Unseen predictions for submission |
| `sample_submission.csv` | 2 000 | 2 (`transaction_id`, `is_fraud`) | Submission format template |

---

## 3. Feature Catalogue

| Feature | Type | Range / Values | Description |
|---|---|---|---|
| `transaction_id` | int | 1 – 10 000 (unique) | Row identifier — **drop before modelling** |
| `amount` | float | 0.00 – 1 471.04 | Transaction monetary value |
| `transaction_hour` | int | 0 – 23 | Hour of day (0 = midnight) |
| `merchant_category` | str | Food, Clothing, Travel, Electronics, Grocery | Merchant type (5 categories) |
| `foreign_transaction` | int (binary) | 0 / 1 | Whether the transaction occurred abroad |
| `location_mismatch` | int (binary) | 0 / 1 | Location vs. cardholder expectation mismatch |
| `device_trust_score` | int | 25 – 99 | Trust score of the device used |
| `velocity_last_24h` | int | 0 – 9 | Number of transactions in last 24 h |
| `cardholder_age` | int | 18 – 69 | Age of the cardholder |
| **`is_fraud`** | int (binary) | 0 / 1 | **Target variable** — 0 = legit, 1 = fraud |

---

## 4. Data Quality Snapshot

| Check | Result |
|---|---|
| Missing values | **None** — all columns fully populated |
| Duplicate rows | **0** |
| Duplicate transaction IDs | **0** |
| Data-type issues | None — all types consistent with expectations |
| Invalid / impossible values | None detected |
| Inconsistent categoricals | Clean — exactly 5 well-formatted categories |

---

## 5. Target Distribution — Critical Imbalance

| Class | Count | Percentage |
|---|---|---|
| 0 (Legitimate) | 7 879 | **98.49 %** |
| 1 (Fraud) | 121 | **1.51 %** |

> **Imbalance ratio ≈ 65 : 1.**  
> Accuracy is meaningless here. The competition uses **F1-score**.  
> Must apply class-weight balancing, SMOTE, or threshold tuning.

---

## 6. Key Patterns Discovered in Training Data

### 6.1 Strongest Fraud Signals (Pearson correlation with `is_fraud`)

| Feature | Correlation | Direction |
|---|---|---|
| `foreign_transaction` | +0.179 | Foreign transactions ≈ **10× higher fraud rate** (8.1 % vs 0.8 %) |
| `location_mismatch` | +0.168 | Mismatch ≈ **9× higher fraud rate** (8.2 % vs 0.9 %) |
| `device_trust_score` | −0.138 | Fraud devices have **much lower trust** (mean 38 vs 62) |
| `transaction_hour` | −0.135 | Fraud peaks at **midnight–3 AM** (median 2 vs 12) |
| `velocity_last_24h` | +0.110 | Fraudsters make **more transactions** (mean 3.3 vs 2.0) |
| `amount` | +0.034 | Weak — fraud mean slightly higher but very noisy |
| `cardholder_age` | +0.000 | Essentially **no signal** |

### 6.2 Fraud by Merchant Category

| Category | Fraud Rate |
|---|---|
| Grocery | 2.14 % |
| Food | 1.86 % |
| Travel | 1.52 % |
| Clothing | 1.10 % |
| Electronics | 0.96 % |

Differences exist but are modest — all categories present in both classes.

### 6.3 Fraud Profile Summary

A typical fraudulent transaction is characterised by:
- **Late-night hour** (0 – 3 AM)
- **Foreign origin** and/or **location mismatch**
- **Low device trust score** (median ≈ 32)
- **Higher transaction velocity** in last 24 h
- Amount and age provide minimal signal

---

## 7. Submission Format

```csv
transaction_id,is_fraud
9945,0
2602,0
343,0
...
```

Two columns: `transaction_id` (int) and `is_fraud` (0 or 1).  
Must contain exactly 2 000 rows matching `test.csv` IDs.
