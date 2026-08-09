# 📋 Team Coordination — Parallel Workstream Overview

## Two Members, One Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TIME →                                          │
│                                                                        │
│  MEMBER 1 (Data Track)            MEMBER 2 (Modelling Track)           │
│  ─────────────────────            ──────────────────────────            │
│                                                                        │
│  ┌─────────────────────┐          ┌─────────────────────┐              │
│  │ Phase 1: Profiling  │          │ Setup: Project dirs  │  PARALLEL   │
│  │ ~1 hour             │          │ requirements.txt     │             │
│  └────────┬────────────┘          └──────────┬──────────┘              │
│           │                                  │                         │
│  ┌────────▼────────────┐          ┌──────────▼──────────┐              │
│  │ Phase 2: Cleaning   │          │ Phase 5: Pipeline   │  PARALLEL   │
│  │ ~1 hour             │          │ architecture        │             │
│  └────────┬────────────┘          └──────────┬──────────┘              │
│           │                                  │                         │
│  ┌────────▼────────────┐          ┌──────────▼──────────┐              │
│  │ Phase 3: EDA        │          │ Phase 6: Feature    │  PARALLEL   │
│  │ ~2 hours            │          │ selection scaffold  │             │
│  └────────┬────────────┘          └──────────┬──────────┘              │
│           │                                  │                         │
│  ┌────────▼────────────┐                     │                         │
│  │ Phase 4: Feature    │                     │                         │
│  │ Engineering         │                     │                         │
│  │ ~1.5 hours          │                     │                         │
│  └────────┬────────────┘                     │                         │
│           │                                  │                         │
│           ╚══════════ HANDOFF ═══════════════╝                         │
│                          │                                             │
│                 ┌────────▼────────────┐                                │
│                 │ Phase 7: Training   │  MEMBER 2 ONLY                │
│                 │ ~2 hours            │                                │
│                 └────────┬────────────┘                                │
│                          │                                             │
│                 ┌────────▼────────────┐                                │
│                 │ Phase 8: Tuning     │  MEMBER 2 ONLY                │
│                 │ ~1.5 hours          │                                │
│                 └────────┬────────────┘                                │
│                          │                                             │
│                 ┌────────▼────────────┐                                │
│                 │ Phase 9: Final      │  MEMBER 2 ONLY                │
│                 │ ~1.5 hours          │  (Member 1 can help           │
│                 └────────┬────────────┘   with report writing)        │
│                          │                                             │
│                 ┌────────▼────────────┐                                │
│                 │ Phase 10: Docs      │  BOTH (split the docs)        │
│                 │ ~1 hour             │                                │
│                 └────────────────────┘                                │
│                                                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## File Ownership

| File | Owner | Touches |
|---|---|---|
| `src/data_profiling.py` | **Member 1** | M1 only |
| `src/data_cleaning.py` | **Member 1** | M1 creates, M2 imports |
| `src/eda.py` | **Member 1** | M1 only |
| `src/feature_engineering.py` | **Member 1** | M1 creates, M2 imports |
| `src/preprocessing.py` | **Member 2** | M2 only |
| `src/feature_selection.py` | **Member 2** | M2 only |
| `src/train.py` | **Member 2** | M2 only |
| `src/evaluate.py` | **Member 2** | M2 only |
| `src/tune.py` | **Member 2** | M2 only |
| `src/predict.py` | **Member 2** | M2 only |
| `src/interpret.py` | **Member 2** | M2 only |
| `notebooks/01_data_analysis.ipynb` | **Member 1** | M1 only |
| `notebooks/02_eda.ipynb` | **Member 1** | M1 only |
| `notebooks/03_model_training.ipynb` | **Member 2** | M2 only |
| `reports/data_quality_report.md` | **Member 1** | M1 creates |
| `reports/model_results.md` | **Member 2** | M2 creates |
| `reports/final_report.md` | **Member 2** | Both contribute |
| `README.md` | **Member 2** | Both review |
| `requirements.txt` | **Member 2** | M2 creates |
| `submission.csv` | **Member 2** | M2 creates |

---

## Critical Coordination Points

### 1. Feature Column Names (Agree Before Starting)
Both members must use the **exact same column names**:

```python
# Engineered features (Member 1 creates, Member 2 consumes)
NUMERICAL_COLS = [
    'log_amount', 'device_trust_score', 'velocity_last_24h',
    'cardholder_age', 'amount_per_velocity', 'hour_sin', 'hour_cos'
]
BINARY_COLS = [
    'foreign_transaction', 'location_mismatch',
    'high_velocity_low_trust', 'late_night_flag',
    'late_night_x_low_trust', 'foreign_x_location'
]
CATEGORICAL_COLS = ['merchant_category']
ORDINAL_COLS = ['risk_flags_count']
TARGET = 'is_fraud'
ID_COL = 'transaction_id'
```

### 2. Function Interface Contract
Member 1's `engineer_features(df)` must:
- Accept a raw DataFrame (from CSV with all original columns)
- Return a DataFrame with all original columns PLUS engineered columns
- Work identically on train and test data
- NOT drop `transaction_id` or `is_fraud` (Member 2's code handles that)

### 3. Handoff Checklist
Before Member 2 starts Phase 7, confirm:
- [ ] `src/feature_engineering.py` exists and `engineer_features()` works
- [ ] `src/data_cleaning.py` exists and `prepare_for_modelling()` works
- [ ] Running `engineer_features(pd.read_csv('data/raw/test.csv'))` doesn't error
- [ ] Column names match the agreed list above
- [ ] No target leakage in any feature

---

## Estimated Timeline

| Phase | Duration | Parallel? | Who |
|---|---|---|---|
| Setup + Phase 1 | 1 hour | ✅ Yes | M1 + M2 |
| Phase 2 + Phase 5 | 1 hour | ✅ Yes | M1 + M2 |
| Phase 3 + Phase 6 | 2 hours | ✅ Yes | M1 + M2 |
| Phase 4 | 1.5 hours | M1 only | M1 |
| **Handoff** | 15 min | — | Both |
| Phase 7 | 2 hours | M2 only | M2 (M1 reviews EDA) |
| Phase 8 | 1.5 hours | M2 only | M2 |
| Phase 9 | 1.5 hours | M2 only | M2 |
| Phase 10 | 1 hour | ✅ Both | M1: quality report, M2: README + final report |
| **Total** | **~8 hours** | | Saved ~3 hours vs sequential |

---

## Git Branch Strategy (if using version control)

```
main
├── feature/member1-data-track     (Phases 1-4)
├── feature/member2-model-track    (Phases 5-6, then 7-10)
└── merge after handoff verification
```
