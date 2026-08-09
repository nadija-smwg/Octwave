# Data Quality Report

## 1. Dataset Shapes & Missing Values
- **Train shape**: (8000, 10)
- **Test shape**: (2000, 9)
- **Sample Submission shape**: (2000, 2)
- **Missing values in Train**: 0
- **Missing values in Test**: 0

## 2. Duplicate Checks
- **Duplicate rows (Train)**: 0
- **Duplicate rows (Test)**: 0
- **Duplicate transaction_id (Train)**: 0
- **Duplicate transaction_id (Test)**: 0
- **transaction_id overlap (Train vs Test)**: 0 (Expected 0)
- **Test IDs match Submission IDs**: True

## 3. Target Distribution (is_fraud)
- **Count**: {0: 7879, 1: 121}
- **Proportion**: {0: 0.984875, 1: 0.015125}
- **Imbalance**: Severe (requires specific techniques)

## 4. Categorical Consistency
- **Categories in Train**: ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel']
- **Categories in Test**: ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel']
- **Match**: True

## 5. Distribution Comparison (Train vs Test)
| Feature | Train Mean | Test Mean | Train Std | Test Std |
|---|---|---|---|---|
| amount | 175.3615 | 178.3034 | 174.7273 | 178.0550 |
| transaction_hour | 11.5886 | 11.6120 | 6.9170 | 6.9471 |
| foreign_transaction | 0.0985 | 0.0950 | 0.2980 | 0.2933 |
| location_mismatch | 0.0850 | 0.0885 | 0.2789 | 0.2841 |
| device_trust_score | 61.9940 | 61.0185 | 21.4932 | 21.4502 |
| velocity_last_24h | 2.0079 | 2.0130 | 1.4392 | 1.4060 |
| cardholder_age | 43.4078 | 43.7125 | 14.9733 | 15.0037 |

**Conclusion**: Data distributions between train and test match closely. Data quality is excellent.
## 6. Data Cleaning Summary (Phase 2)
- **Missing values**: 0 (No imputation needed).
- **Duplicates**: 0 (No deduplication needed).
- **Type fixes**: `merchant_category` converted to categorical.
- **Invalid values**: None found.
- **Outliers**: Identified in `amount` and `velocity_last_24h` but kept as they represent valid financial signals.
- **ID columns**: `transaction_id` removed prior to modelling.
