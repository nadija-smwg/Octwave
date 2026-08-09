# Octwave Credit Card Fraud Detection: Comprehensive Pipeline Report

This report provides a complete, end-to-end breakdown of the machine learning pipeline used in the Octwave Credit Card Fraud Detection project. It is designed to help you understand not just **what** the code does, but **why** these specific data science decisions were made.

---

## 1. The Core Challenge: Highly Imbalanced Data
The central problem of this project is detecting fraudulent transactions in a dataset where only **1.5% of transactions are fraud**. This is known as a **class imbalance**. 
Because fraud is so rare, traditional accuracy metrics fail (a model that always predicts "Not Fraud" would be 98.5% accurate but completely useless). To solve this, the project optimizes for the **F1-score**, which balances Precision (how many predicted frauds were actually fraud) and Recall (how many actual frauds were caught).

---

## 2. Feature Engineering (`src/feature_engineering.py`)
Before passing data to a model, raw data must be transformed into "features" that highlight patterns for the algorithm. The project uses advanced feature engineering techniques based on domain knowledge:

*   **Log Transformation (`log_amount`)**: Financial amounts are often highly skewed (mostly small purchases, a few massive ones). Taking the logarithm (`np.log1p`) squashes this distribution, making it easier for models to process.
*   **Cyclical Encoding (`hour_sin`, `hour_cos`)**: Time is cyclical. Hour 23 is right next to Hour 0. If you just use the number 23 and 0, the model thinks they are far apart. Using sine and cosine waves mathematically connects 23:00 and 00:00.
*   **Interaction Features**: Fraud isn't usually just one bad signal, it's a combination. The code mathematically combines signals to give the model a "shortcut":
    *   `risk_flags_count`: Adding up red flags (`foreign_transaction` + `location_mismatch`).
    *   `high_velocity_low_trust`: Flagging users who make many transactions rapidly (`velocity_last_24h >= 4`) while having a low device trust score.
    *   `late_night_x_low_trust`: Combining time of day with device trust.
*   **Binning (`age_group`, `device_trust_bin`)**: Grouping continuous numbers into categories (e.g., ages 17-25 as "young"). This helps models find patterns in groups rather than treating a 24-year-old and 25-year-old as completely different.

---

## 3. Preprocessing Pipeline (`src/preprocessing.py`)
Machine learning models require data in a specific format (numbers, scaled appropriately). The project builds a `ColumnTransformer` to handle this automatically:
*   **Scaling (`StandardScaler`)**: Numerical features (like `amount` and `velocity`) operate on completely different scales. Standardizing them forces them to have a mean of 0 and standard deviation of 1, preventing large numbers from dominating the model.
*   **Categorical Encoding (`OneHotEncoder`)**: Models cannot read text (like a `merchant_category` of "Groceries"). One-Hot Encoding converts these categories into binary (0 or 1) columns.
*   **Data Leakage Prevention**: The code explicitly checks (`verify_no_leakage`) to ensure that no data from the training set accidentally leaked into the validation set. This is a critical best practice to ensure the model's evaluation is honest.

---

## 4. Feature Selection (`src/feature_selection.py`)
Once hundreds of features are created, you need to find the best ones. Too many useless features confuse the model and cause "overfitting". The project defines several scientific ways to evaluate feature quality:
*   **Mutual Information & Correlation**: Measures mathematically how much knowing a feature tells you about the target (`is_fraud`).
*   **Random Forest Importance**: Trains a quick tree model to see which features it relies on the most.
*   **VIF (Variance Inflation Factor)**: Checks for "multicollinearity"—when two features are basically telling the model the exact same thing.
*   **RFECV (Recursive Feature Elimination)**: Systematically drops the weakest features one by one, using cross-validation to find the perfect subset of features.

---

## 5. Model Training and Handling Imbalance (`src/final_train.py`)
The project uses **XGBoost (Extreme Gradient Boosting)**, one of the most powerful algorithms for tabular data. It builds decision trees sequentially, where each new tree tries to fix the errors of the previous ones.

**How does it handle the 1.5% fraud rate?**
Instead of artificially creating fake fraud data (a technique called SMOTE, which is in the codebase but disabled for the final run), the final model uses `scale_pos_weight = 97.46`. 
*   This tells XGBoost: *"Every time you misclassify a fraudulent transaction, penalize yourself 97 times harder than if you misclassify a normal transaction."* This forces the model to care about the minority class.

The model uses fine-tuned hyperparameters (like `learning_rate=0.2` and `max_depth=4`) to prevent overfitting. Once trained, it is saved using `joblib` so it can be loaded later without retraining.

---

## 6. Inference and Threshold Tuning (`src/predict.py`)
When making final predictions on new, unseen data (`test.csv`), the script loads the saved model and runs the same feature engineering steps.

**The Thresholding Secret:**
By default, machine learning models predict fraud if the probability is > 50% (0.50). However, because fraud is so rare, the model might only be 30% confident, but 30% is still highly suspicious!
The project uses an **optimal threshold of 0.85**. Wait, 0.85? Yes! Because of the extreme `scale_pos_weight` used during training, the model's raw probabilities get heavily skewed upwards. Tuning the final probability threshold (finding the exact cutoff that maximizes the F1-score) is a classic advanced data science technique to squeeze out maximum performance.

Finally, it outputs the results into `submission.csv` to be scored.
