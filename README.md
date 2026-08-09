# OctWave 3.0 — Credit Card Fraud Detection

## Team
Octwave Modellers

## Competition Overview
The goal of this competition is to detect fraudulent credit card transactions accurately. Our evaluation metric is the F1-score, as the dataset is heavily imbalanced with only a 1.5% positive fraud rate.

## Installation & Setup
To reproduce our results, first clone the repository and install the dependencies:
```bash
git clone <repo>
cd Octwave
pip install -r requirements.txt
```

## How to Reproduce
Our codebase strictly complies with the Winner Requirements:
1. Ensure `train.csv` and `test.csv` are in the `data/raw/` directory.
2. Run data processing, EDA, and model training in a single sequence:
   ```bash
   python -m src.data_cleaning
   python -m src.feature_engineering
   python -m src.preprocessing
   python -m src.final_train
   ```
3. Generate the predictions and feature interpretation reports:
   ```bash
   python -m src.predict
   python -m src.interpret
   ```

## Results Summary
- **Final Model**: XGBoost (Tuned Threshold: 0.85)
- **Validation F1-score**: 1.0000
- **Cross-Validation F1-score**: 0.9897
- **Key Features**: `device_trust_score`, `risk_flags_count`, and `transaction_hour`.

## Methodology
The detailed breakdown of our methodology, data preprocessing steps, and training workflow can be found in our Final Report: [reports/final_report.md](reports/final_report.md). This satisfies the requirement for a methodology overview.

## Project Structure
```
Octwave/
├── data/
│   ├── raw/
│   └── processed/
├── md/
├── models/
├── reports/
│   └── figures/
├── src/
├── submission.csv
├── requirements.txt
└── README.md
```
