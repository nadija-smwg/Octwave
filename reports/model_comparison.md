# Model Comparison Results (Strategy A: Class Weights)

| Model               |   CV F1 Mean |   CV F1 Std |   Train F1 Mean |   Val F1 |   Val Precision |   Val Recall |   Val ROC-AUC |
|:--------------------|-------------:|------------:|----------------:|---------:|----------------:|-------------:|--------------:|
| LightGBM            |     0.977452 |   0.0332724 |        1        | 1        |        1        |     1        |      1        |
| Gradient Boosting   |     0.936806 |   0.0407334 |        1        | 1        |        1        |     1        |      1        |
| XGBoost             |     0.952074 |   0.0221253 |        1        | 0.938776 |        0.92     |     0.958333 |      0.999921 |
| Random Forest       |     0.813137 |   0.0534158 |        1        | 0.909091 |        1        |     0.833333 |      0.999484 |
| SVM (RBF)           |     0.814295 |   0.0390605 |        0.903743 | 0.857143 |        0.84     |     0.875    |      0.999022 |
| Logistic Regression |     0.745217 |   0.0547922 |        0.764785 | 0.695652 |        0.533333 |     1        |      0.998784 |

## Threshold Tuning (LightGBM)
- Optimal Threshold: 0.9943
- Tuned Val F1: 1.0000
