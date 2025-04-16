# fine_tune_gb.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def fine_tune_gradient_boosting(X_train, y_train, X_test, y_test, feature_names=None, n_iter=20, cv=3):
    """
    Fine-tune a GradientBoosting model using RandomizedSearchCV
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target values
    X_test : DataFrame or array
        Test features
    y_test : Series or array
        Test target values
    feature_names : list, optional
        List of feature names to use
    n_iter : int
        Number of parameter settings to sample in RandomizedSearchCV
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    tuple
        (best model, performance metrics dictionary)
    """
    print("Starting GradientBoosting model fine-tuning...")
    
    # Define the parameter grid for GradientBoosting
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Define a custom scorer that minimizes MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create the RandomizedSearchCV object
    gb_random = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=mae_scorer,
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the random search
    gb_random.fit(X_train, y_train)
    
    # Print the best parameters and score
    print(f"Best parameters found: {gb_random.best_params_}")
    print(f"Best MAE score: {-gb_random.best_score_:.2f}")
    
    # Train a new model with the best parameters
    best_gb = GradientBoostingRegressor(**gb_random.best_params_, random_state=42)
    best_gb.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_tuned = best_gb.predict(X_test)
    y_pred_tuned = np.maximum(0, y_pred_tuned)  # Ensure non-negative
    
    # Calculate metrics
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mape_tuned = np.mean(np.abs((y_test - y_pred_tuned) / (y_test + 1))) * 100
    
    print(f"Tuned GradientBoosting performance: MAE = {mae_tuned:.2f}, RMSE = {rmse_tuned:.2f}, MAPE = {mape_tuned:.2f}%")
    
    # Compare with default GradientBoosting
    default_gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    default_gb.fit(X_train, y_train)
    y_pred_default = default_gb.predict(X_test)
    y_pred_default = np.maximum(0, y_pred_default)
    
    mae_default = mean_absolute_error(y_test, y_pred_default)
    rmse_default = np.sqrt(mean_squared_error(y_test, y_pred_default))
    mape_default = np.mean(np.abs((y_test - y_pred_default) / (y_test + 1))) * 100
    
    print(f"Default GradientBoosting performance: MAE = {mae_default:.2f}, RMSE = {rmse_default:.2f}, MAPE = {mape_default:.2f}%")
    
    # Get feature importance
    if feature_names is not None:
        # 使用传入的特征名称
        feature_cols = feature_names
    elif hasattr(X_train, 'columns'):
        # 如果X_train是DataFrame
        feature_cols = X_train.columns.tolist()
    else:
        # 如果X_train是numpy数组，创建通用特征名称
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # 确保X_train是数组格式
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_gb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save the model
    joblib.dump(best_gb, 'best_gradient_boosting_model.pkl')
    print("Best model saved as 'best_gradient_boosting_model.pkl'")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance - Tuned GradientBoosting Model')
    plt.tight_layout()
    plt.savefig('gb_feature_importance.png')
    print("Feature importance visualization saved as 'gb_feature_importance.png'")
    

    # Return the best model and metrics
    metrics = {
        'mae': mae_tuned,
        'rmse': rmse_tuned,
        'mape': mape_tuned,
        'feature_importance': feature_importance,
        'best_params': gb_random.best_params_
    }
    
    return best_gb, metrics
