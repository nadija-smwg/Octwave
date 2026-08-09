import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_correlation_with_target(X, y):
    """Compute Pearson correlation of each feature with the target."""
    df = X.copy()
    df['target'] = y
    corr = df.corr()['target'].drop('target').sort_values(ascending=False)
    return corr

def compute_mutual_information(X, y):
    """Compute mutual information scores for features."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_series

def compute_tree_importance(X, y):
    """Compute feature importances using a Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importances

def compute_vif(X):
    """Compute Variance Inflation Factor for features."""
    # VIF requires numeric data and cannot handle NaNs/Infs
    X_num = X.select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_num.columns
    # Add a small constant to prevent division by zero in VIF calculation if variables are highly correlated
    vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(len(X_num.columns))]
    return vif_data.sort_values('VIF', ascending=False)

def run_rfecv(X, y, estimator=None, cv=None):
    """Run Recursive Feature Elimination with Cross-Validation."""
    if estimator is None:
        estimator = LogisticRegression(class_weight='balanced', max_iter=1000)
    if cv is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='f1')
    rfecv.fit(X, y)
    
    selected_features = X.columns[rfecv.support_].tolist()
    return selected_features, rfecv

def plot_feature_importance_comparison(results, save_path):
    """Plot comparison of different feature importance metrics."""
    # results is a dict with 'Correlation', 'Mutual Information', 'Tree Importance'
    df_results = pd.DataFrame(results)
    
    # Normalize the scores for comparison (0 to 1)
    for col in df_results.columns:
        if df_results[col].dtype == np.float64:
             df_results[col] = (df_results[col] - df_results[col].min()) / (df_results[col].max() - df_results[col].min())
    
    plt.figure(figsize=(15, 10))
    
    # We transpose for plotting
    df_plot = df_results.reset_index().melt(id_vars='index', var_name='Metric', value_name='Normalized Score')
    df_plot.rename(columns={'index': 'Feature'}, inplace=True)
    
    sns.barplot(data=df_plot, x='Normalized Score', y='Feature', hue='Metric')
    plt.title('Normalized Feature Importances Comparison')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_cleaned.csv')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    if os.path.exists(train_path):
        # We need the preprocessed dataframe to run feature selection properly
        import sys
        sys.path.append(base_dir)
        from src.preprocessing import split_data, build_preprocessor, build_full_pipeline
        
        df = pd.read_csv(train_path)
        X_train, _, y_train, _ = split_data(df)
        
        print("Transforming training data for feature selection analysis...")
        preprocessor = build_preprocessor()
        from src.feature_engineering import engineer_features
        
        X_train_eng = engineer_features(X_train)
        X_train_processed = preprocessor.fit_transform(X_train_eng)
        
        # Get feature names after ColumnTransformer
        feature_names = preprocessor.get_feature_names_out()
        # Clean feature names (remove prefix like num__, cat__)
        clean_feature_names = [f.split('__')[-1] for f in feature_names]
        
        X_train_df = pd.DataFrame(X_train_processed, columns=clean_feature_names, index=X_train.index)
        
        print("Computing correlations...")
        corr = compute_correlation_with_target(X_train_df, y_train)
        
        print("Computing Mutual Information...")
        mi = compute_mutual_information(X_train_df, y_train)
        
        print("Computing Tree Importance...")
        tree_imp = compute_tree_importance(X_train_df, y_train)
        
        print("Computing VIF...")
        # For VIF, we should avoid dummy variable trap and perfectly correlated features
        vif = compute_vif(X_train_df)
        
        # Compile results
        results = pd.DataFrame({
            'Correlation (Abs)': corr.abs(),
            'Mutual Information': mi,
            'Tree Importance': tree_imp
        }).fillna(0)
        
        print("Plotting results...")
        plot_feature_importance_comparison(results, os.path.join(figures_dir, 'feature_importance_comparison.png'))
        
        print("Running RFECV...")
        selected_features, rfecv = run_rfecv(X_train_df, y_train)
        
        # Generate summary report
        report_path = os.path.join(reports_dir, 'feature_selection_summary.md')
        
        with open(report_path, 'w') as f:
            f.write("# Feature Selection Summary\n\n")
            f.write("## 1. Feature Importance Metrics\n\n")
            f.write(results.sort_values('Tree Importance', ascending=False).to_markdown())
            f.write("\n\n## 2. VIF Analysis\n\n")
            f.write(vif.to_markdown())
            f.write("\n\n## 3. RFECV Results\n\n")
            f.write(f"- Optimal number of features: {rfecv.n_features_}\n")
            f.write(f"- Selected features: {', '.join(selected_features)}\n")
            
        print(f"Feature selection complete. Summary saved to {report_path}")
