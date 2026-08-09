# Phase 4 — Feature Engineering

## Objective
Create meaningful, domain-relevant features that capture fraud signals beyond raw feature values. Every engineered feature must have a clear justification.

---

## 4.1 Numerical Transformations

### 4.1.1 Log Transform of `amount`
```python
log_amount = np.log1p(amount)
```
- **Why**: `amount` is right-skewed (mean=175, median=121, max=1390). Log transform reduces the influence of extreme values and may help linear models.
- **Expected impact**: Moderate — `amount` has weak correlation with fraud (+0.034), but the transform may reveal non-linear patterns.

### 4.1.2 Scaled `device_trust_score`
```python
device_trust_normalized = (device_trust_score - 25) / (99 - 25)  # Scale to [0, 1]
```
- **Why**: Normalizing to [0,1] aids models sensitive to feature scales (Logistic Regression, SVM, KNN). The raw range 25–99 is arbitrary.
- **Note**: This will be done inside the sklearn Pipeline (fit on train only).

---

## 4.2 Cyclical Encoding of `transaction_hour`

```python
hour_sin = np.sin(2 * np.pi * transaction_hour / 24)
hour_cos = np.cos(2 * np.pi * transaction_hour / 24)
```

- **Why**: Hours are cyclical — 23:00 is close to 00:00, but linear encoding treats them as maximally distant. Sin/cos encoding preserves the circular relationship.
- **Expected impact**: High — fraud heavily concentrates at 0–3 AM. Cyclical encoding lets models learn that hours 23 and 0 are adjacent.

---

## 4.3 Interaction Features

### 4.3.1 `risk_flags_count`
```python
risk_flags_count = foreign_transaction + location_mismatch
```
- **Why**: Both are strong individual fraud signals. Their sum (0, 1, or 2) creates a simple risk-tier feature. A transaction with both flags should be flagged as highest risk.
- **Expected impact**: High — fraud rate likely spikes dramatically when both flags = 1.

### 4.3.2 `foreign_x_location_mismatch`
```python
foreign_x_location_mismatch = foreign_transaction * location_mismatch
```
- **Why**: Captures the specific interaction where BOTH flags are present simultaneously. This is a pure interaction term.
- **Expected impact**: High — dual-flag transactions are likely the highest-fraud segment.

### 4.3.3 `high_velocity_low_trust`
```python
high_velocity_low_trust = (velocity_last_24h >= 4) & (device_trust_score <= 40)
```
- **Why**: High transaction velocity on an untrusted device is a classic fraud pattern. This creates a binary flag for this specific combination.
- **Expected impact**: Moderate-High — captures a compound fraud indicator.

### 4.3.4 `late_night_flag`
```python
late_night_flag = (transaction_hour >= 22) | (transaction_hour <= 4)
```
- **Why**: Fraud concentrates at late-night hours (median fraud hour = 2). A binary flag simplifies what the model needs to learn.
- **Expected impact**: High — directly captures the strongest temporal fraud signal.

### 4.3.5 `late_night_x_low_trust`
```python
late_night_x_low_trust = late_night_flag * (device_trust_score <= 40)
```
- **Why**: Late-night + untrusted device is a high-risk combination.
- **Expected impact**: Moderate-High.

---

## 4.4 Ratio / Mathematical Combinations

### 4.4.1 `amount_per_velocity`
```python
amount_per_velocity = amount / (velocity_last_24h + 1)
```
- **Why**: High spending per transaction while making many transactions could indicate automated fraud. This normalizes amount by activity level.
- **Expected impact**: Moderate.

### 4.4.2 `velocity_x_amount` (total daily spend proxy)
```python
velocity_x_amount = velocity_last_24h * amount
```
- **Why**: Proxy for total spending in the last 24h. High total spend could be a fraud indicator.
- **Expected impact**: Weak-Moderate — depends on whether total spend matters more than per-transaction amount.

---

## 4.5 Binning

### 4.5.1 `age_group`
```python
age_group = pd.cut(cardholder_age, bins=[17, 25, 35, 50, 70], labels=['young', 'adult', 'middle', 'senior'])
```
- **Why**: While `cardholder_age` has near-zero correlation with fraud, age groups might reveal non-linear patterns (e.g., very young cardholders might be targeted differently).
- **Expected impact**: Low — included as exploratory. Will be evaluated in feature selection.

### 4.5.2 `device_trust_bin`
```python
device_trust_bin = pd.cut(device_trust_score, bins=[24, 40, 60, 80, 100], labels=['very_low', 'low', 'medium', 'high'])
```
- **Why**: Creates interpretable trust tiers. Tree models can learn these boundaries, but binning helps linear models capture non-linearity.
- **Expected impact**: Moderate.

---

## 4.6 Categorical Encoding

### `merchant_category` — One-Hot Encoding
```python
# Inside ColumnTransformer:
OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
```
- **Why**: 5 categories is small enough for one-hot encoding without dimensionality issues. `drop='first'` avoids the dummy variable trap for linear models.
- **Note**: Fit only on training data. `handle_unknown='ignore'` handles unseen categories in test/production.

### Alternative: Target Encoding (for tree models)
```python
# merchant_fraud_rate = merchant_category.map(train_fraud_rates)
```
- **Why**: Encodes the category's fraud rate directly. More information-dense than one-hot.
- **Risk**: Data leakage if not done with cross-validation within training fold.
- **Decision**: Use one-hot in primary pipeline; test target encoding as a variant.

---

## 4.7 Feature Engineering Summary

| Feature | Type | Source | Justification | Expected Impact |
|---|---|---|---|---|
| `log_amount` | Numeric | `amount` | Reduce skewness | Moderate |
| `hour_sin` | Numeric | `transaction_hour` | Cyclical encoding | High |
| `hour_cos` | Numeric | `transaction_hour` | Cyclical encoding | High |
| `risk_flags_count` | Numeric (0–2) | `foreign_transaction` + `location_mismatch` | Additive risk indicator | High |
| `foreign_x_location` | Binary | `foreign_transaction` × `location_mismatch` | Interaction | High |
| `high_velocity_low_trust` | Binary | `velocity_last_24h`, `device_trust_score` | Compound indicator | Moderate-High |
| `late_night_flag` | Binary | `transaction_hour` | Temporal fraud signal | High |
| `late_night_x_low_trust` | Binary | `late_night_flag`, `device_trust_score` | Compound indicator | Moderate-High |
| `amount_per_velocity` | Numeric | `amount` / (`velocity` + 1) | Spending rate | Moderate |
| `merchant_category_*` | Binary × 4 | `merchant_category` | One-hot encoding | Low-Moderate |

---

## 4.8 Features NOT Created (and why)

| Idea | Why Rejected |
|---|---|
| Polynomial features (all pairs) | Too many features for 8000 samples; overfitting risk |
| PCA components | Destroys interpretability; few features to begin with |
| `age × amount` | Both have near-zero fraud correlation; interaction unlikely meaningful |
| Frequency encoding of `transaction_id` | IDs are unique; no frequency signal |
| Text features | No text columns in dataset |
| Time-series features | No temporal ordering; `transaction_hour` is point-in-time |

---

## 4.9 Data Leakage Prevention

| Step | Leakage-Safe? | How |
|---|---|---|
| Log transform | ✅ | Stateless transform — no fitting needed |
| Cyclical encoding | ✅ | Stateless mathematical transform |
| Interaction features | ✅ | Computed from feature values only, not statistics |
| Binning | ✅ | Fixed bin edges (domain-based, not data-derived) |
| One-hot encoding | ✅ | Fit on train only via ColumnTransformer |
| Scaling | ✅ | Fit on train only via Pipeline |
| Target encoding | ⚠️ | Must use cross-validated within-fold encoding |

---

## 4.10 Code Structure for Phase 4

```
src/feature_engineering.py
    ├── create_log_amount(df) → df with log_amount
    ├── create_cyclical_hour(df) → df with hour_sin, hour_cos
    ├── create_interaction_features(df) → df with interaction columns
    ├── create_ratio_features(df) → df with ratio columns
    ├── create_binned_features(df) → df with binned columns
    ├── engineer_features(df) → df with all engineered features
    └── get_feature_columns() → list of feature column names

src/preprocessing.py
    ├── build_preprocessor() → ColumnTransformer
    ├── build_full_pipeline(model) → Pipeline
    └── get_column_lists() → (numerical_cols, categorical_cols, binary_cols)
```
