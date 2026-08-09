# 📋 Team Coordination — Parallel Workstream Overview

## Competition Rules Summary (Affecting Teamwork)

| Rule | Detail | Impact |
|---|---|---|
| **Team Size** | Up to 4 members (we have 2) | ✅ Compliant |
| **Single Kaggle Account** | Each member = 1 account, 1 team | Both members must be on same official Kaggle team |
| **Official Team Name** | Must match OC-assigned name | Set team name before first submission |
| **No Private Code Sharing** | Outside of official team | All sharing happens within the Kaggle team or shared repo |
| **Max 10 Submissions/Day** | Per team | Coordinate who submits — don't both submit independently |
| **2 Final Submissions** | For Private Leaderboard | Agree on: Submission A (aggressive) + B (conservative) |
| **Winner Requirements** | Reproducible code + methodology | Both members contribute to final documentation |
| **F1-Score Metric** | Primary evaluation | Every model decision uses F1, not accuracy |

---

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
│                 │ Phase 9: Final +    │  MEMBER 2                     │
│                 │ Dual Submissions    │  (Member 1 helps with docs)   │
│                 └────────┬────────────┘                                │
│                          │                                             │
│                 ┌────────▼────────────┐                                │
│                 │ Phase 10: Docs      │  BOTH MEMBERS                 │
│                 │ (Winner Req.)       │  Split the documentation      │
│                 └────────────────────┘                                │
│                                                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## File Ownership

| File | Owner | Notes |
|---|---|---|
| `src/data_profiling.py` | **Member 1** | — |
| `src/data_cleaning.py` | **Member 1** | M2 imports `prepare_for_modelling()` |
| `src/eda.py` | **Member 1** | — |
| `src/feature_engineering.py` | **Member 1** | M2 imports `engineer_features()` |
| `src/preprocessing.py` | **Member 2** | — |
| `src/feature_selection.py` | **Member 2** | — |
| `src/train.py` | **Member 2** | — |
| `src/evaluate.py` | **Member 2** | — |
| `src/tune.py` | **Member 2** | — |
| `src/predict.py` | **Member 2** | — |
| `src/interpret.py` | **Member 2** | — |
| `notebooks/01_data_analysis.ipynb` | **Member 1** | — |
| `notebooks/02_eda.ipynb` | **Member 1** | — |
| `notebooks/03_model_training.ipynb` | **Member 2** | — |
| `reports/data_quality_report.md` | **Member 1** | — |
| `reports/model_results.md` | **Member 2** | — |
| `reports/final_report.md` | **Both** | Winner Requirement (Rules Section 5) |
| `README.md` | **Member 2** | Both review; must be reproducible |
| `requirements.txt` | **Member 2** | OSI-approved libs only (Rules Section 6c) |
| `submission.csv` | **Member 2** | Submission A (aggressive) |
| `submission_conservative.csv` | **Member 2** | Submission B (conservative) |
| `submission_log.md` | **Both** | Track all Kaggle uploads (max 10/day) |

---

## Critical Coordination Points

### 1. Kaggle Team Setup (Do First!)
- [ ] Both members join the **same Kaggle team**
- [ ] Set team name to **OC-assigned name** (Rules Section 2)
- [ ] Confirm both members have Kaggle accounts (Rules Section 1)

### 2. Submission Coordination (Rules: Max 10/Day)
- **Designate one person** to upload submissions to Kaggle
- **Always validate locally** before using a submission slot
- **Log every submission** in `submission_log.md`
- **Communicate before submitting** — don't duplicate

### 3. Feature Column Names (Agree Before Starting)
```python
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

### 4. Function Interface Contract
Member 1's `engineer_features(df)` must:
- Accept a raw DataFrame (from CSV with all original columns)
- Return a DataFrame with all original columns PLUS engineered columns
- Work identically on train and test data
- NOT drop `transaction_id` or `is_fraud`
- NOT use any external data

### 5. Handoff Checklist
Before Member 2 starts Phase 7:
- [ ] `src/feature_engineering.py` exists and `engineer_features()` works
- [ ] `src/data_cleaning.py` exists and `prepare_for_modelling()` works
- [ ] Running `engineer_features(pd.read_csv('data/raw/test.csv'))` doesn't error
- [ ] Column names match the agreed list above
- [ ] No target leakage in any feature
- [ ] No external data used

---

## Estimated Timeline

| Phase | Duration | Parallel? | Who |
|---|---|---|---|
| Kaggle team setup | 15 min | ✅ Both | Both |
| Setup + Phase 1 | 1 hour | ✅ Yes | M1 + M2 |
| Phase 2 + Phase 5 | 1 hour | ✅ Yes | M1 + M2 |
| Phase 3 + Phase 6 | 2 hours | ✅ Yes | M1 + M2 |
| Phase 4 | 1.5 hours | M1 only | M1 |
| **Handoff** | 15 min | — | Both |
| Phase 7 | 2 hours | M2 only | M2 (M1 reviews) |
| Phase 8 | 1.5 hours | M2 only | M2 |
| Phase 9 (+ submissions) | 1.5 hours | M2 only | M2 |
| Phase 10 (docs) | 1 hour | ✅ Both | M1: quality report, M2: README + final report |
| **Total** | **~8 hours** | | Saved ~3 hours vs sequential |

---

## Git Branch Strategy

```
main
├── feature/member1-data-track     (Phases 1–4)
├── feature/member2-model-track    (Phases 5–6, then 7–10)
└── merge after handoff verification
```

> ⚠️ **Rules reminder**: Do not share code privately outside the team. All sharing must be within the official Kaggle team or on public forums (Rules Section 4, 5.d).
