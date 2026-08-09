# Exploratory Data Analysis Summary

## 1. Target Distribution
- Total Records: 8000
- Fraud Cases: 121 (1.51%)
- Imbalance Ratio: ~65:1

## 2. Key Correlates with Fraud
- **foreign_transaction**: 0.179
- **location_mismatch**: 0.168
- **velocity_last_24h**: 0.110
- **amount**: 0.034
- **cardholder_age**: 0.000
- **transaction_hour**: -0.135
- **device_trust_score**: -0.138

## 3. Multicollinearity (VIF)
- **amount**: 1.93
- **device_trust_score**: 6.15
- **velocity_last_24h**: 2.69
- **cardholder_age**: 6.14
- **transaction_hour**: 3.36
- **foreign_transaction**: 1.11
- **location_mismatch**: 1.09

## 4. Key Insights & Decisions
- No features exhibit high multicollinearity (all VIFs are relatively low, usually < 5 is safe).
- **transaction_hour** shows strong fraud patterns (spikes around 0-3 AM).
- **device_trust_score** is inversely correlated; lower trust scores have higher fraud rates.
- **foreign_transaction** and **location_mismatch** strongly increase fraud probability.
- Imbalance requires class weights, SMOTE, or threshold tuning.
