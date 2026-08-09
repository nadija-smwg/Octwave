# Hyperparameter Tuning Results

## Performance Summary

| Model    |   Train F1 |    CV F1 |   Val F1 (Default 0.5) |   Val ROC-AUC |   Optimal Threshold |   Val F1 (Tuned) |
|:---------|-----------:|---------:|-----------------------:|--------------:|--------------------:|-----------------:|
| LightGBM |          1 | 0.989466 |                   0.96 |      0.999974 |                0.8  |         0.979592 |
| XGBoost  |          1 | 0.989717 |                   0.96 |      1        |                0.85 |         1        |

## Best Parameters

### LightGBM
```python
{'classifier__subsample': 0.9, 'classifier__reg_lambda': 1.0, 'classifier__reg_alpha': 1.0, 'classifier__num_leaves': 15, 'classifier__n_estimators': 500, 'classifier__min_child_samples': 10, 'classifier__max_depth': 3, 'classifier__learning_rate': 0.1, 'classifier__is_unbalance': True, 'classifier__colsample_bytree': 0.9}
```

### XGBoost
```python
{'classifier__subsample': 0.8, 'classifier__scale_pos_weight': 97.46907216494844, 'classifier__reg_lambda': 2.0, 'classifier__reg_alpha': 0, 'classifier__n_estimators': 200, 'classifier__min_child_weight': 3, 'classifier__max_depth': 4, 'classifier__learning_rate': 0.2, 'classifier__gamma': 0, 'classifier__colsample_bytree': 1.0}
```

