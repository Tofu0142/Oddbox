import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
# For Prophet model
from src.train_model import add_time_features
try:
    from prophet import Prophet
except ImportError:
    print("Prophet not installed. To use Prophet model, install it with: pip install prophet")


import warnings
warnings.filterwarnings('ignore')
# Import the data processing function

def feature_engineering(df, forecast_periods=4):
        # Ensure data is sorted by week
    df = df.sort_values(['box_type', 'week'])
    
    # Add box_type as categorical features using one-hot encoding
    box_type_dummies = pd.get_dummies(df['box_type'], prefix='box')
    df = pd.concat([df, box_type_dummies], axis=1)
    
    # First split the data into train and test sets - STRATIFIED BY BOX TYPE
    # This ensures each box type has both training and test data
    box_types = df['box_type'].unique()
    train_data_list = []
    test_data_list = []
    
    for box_type in box_types:
        box_data = df[df['box_type'] == box_type].copy()
        train_size = int(len(box_data) * 0.8)
        train_data_list.append(box_data.iloc[:train_size])
        test_data_list.append(box_data.iloc[train_size:])
    
    train_data = pd.concat(train_data_list)
    test_data = pd.concat(test_data_list)
    
    # Now add time-dependent features to each set separately
    train_data_with_features = add_time_features(train_data, 'box_orders', is_train=True)
    test_data_with_features = add_time_features(test_data, 'box_orders', is_train=False, train_data=train_data)
    
    # Select features
    feature_cols = [col for col in train_data_with_features.columns 
                   if col not in ['week', 'box_type', 'box_orders']]
    
    # Feature selection based on correlation
    print("\nPerforming feature selection based on correlation...")
    # Calculate correlation matrix for numerical features
    numeric_features = train_data_with_features[feature_cols].select_dtypes(include=['number'])
    corr_matrix = numeric_features.corr()
    
    
    # Filter highly correlated features
    def filter_correlated_features(corr_matrix, threshold=0.8):
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return to_drop
    
    high_corr_features = filter_correlated_features(corr_matrix, threshold=0.8)
    
    if high_corr_features:
        print(f"Removing {len(high_corr_features)} highly correlated features: {high_corr_features}")
        # Remove highly correlated features from feature_cols
        feature_cols = [col for col in feature_cols if col not in high_corr_features]
    else:
        print("No highly correlated features found.")
    
    print(f"Selected {len(feature_cols)} features after correlation filtering")
    
    # Handle NaN values in both train and test sets
    print(f"Checking for NaN values in features...")
    
    # For training data
    train_null_counts = train_data_with_features[feature_cols].isnull().sum()
    if train_null_counts.sum() > 0:
        print(f"NaN values in training features:")
        print(train_null_counts[train_null_counts > 0])
        
        # Drop columns with too many NaNs
        for col in feature_cols.copy():
            if train_data_with_features[col].isnull().sum() > len(train_data_with_features) * 0.3:
                print(f"Dropping feature {col} due to too many NaN values")
                feature_cols.remove(col)
        
        # For remaining columns, fill NaNs with median
        for col in feature_cols:
            if train_data_with_features[col].isnull().sum() > 0:
                median_val = train_data_with_features[col].median()
                train_data_with_features[col] = train_data_with_features[col].fillna(median_val)
                print(f"Filled NaNs in {col} with median value: {median_val}")
    
    # For test data
    test_null_counts = test_data_with_features[feature_cols].isnull().sum()
    if test_null_counts.sum() > 0:
        print(f"NaN values in test features:")
        print(test_null_counts[test_null_counts > 0])
        
        # Fill NaNs with median from training data
        for col in feature_cols:
            if test_data_with_features[col].isnull().sum() > 0:
                train_median = train_data_with_features[col].median()
                test_data_with_features[col] = test_data_with_features[col].fillna(train_median)
                print(f"Filled NaNs in test {col} with training median: {train_median}")
    
    # Prepare training data for ML models
    X_train = train_data_with_features[feature_cols]
    y_train = train_data_with_features['box_orders']
    X_test = test_data_with_features[feature_cols]
    y_test = test_data_with_features['box_orders']
    
    #
    if isinstance(X_train, np.ndarray):
        
        feature_names = feature_cols
    else:
        
        feature_names = X_train.columns.tolist()
    
   
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    
    return X_train, X_test, y_train, y_test, feature_names, test_data_with_features, train_data_with_features


    


def train_global_model(df, forecast_periods=4):
    """
    Train multiple models (including time series models) for all box types and compare their performance
    
    Parameters:
    -----------
    df : DataFrame
        The processed data for all box types
    forecast_periods : int
        Number of periods to forecast ahead
        
    Returns:
    --------
    tuple
        (best model, results dictionary, feature importance)
    """
    print("\nTraining global models for all box types...")
    
    X_train, X_test, y_train, y_test, feature_names, test_data_with_features, train_data_with_features = feature_engineering(df, forecast_periods)
        # Final check for NaNs
    assert X_train.isnull().sum().sum() == 0, "Training features still contain NaN values"
    assert X_test.isnull().sum().sum() == 0, "Test features still contain NaN values"
    
    # Standardize features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Define ML models to compare
    ml_models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=42),
        'Linear': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    # Train ML models and evaluate
    model_results = {}
    
    for name, model in ml_models.items():
        print(f"\nTraining {name} model...")
        
        try:
            # Train model
            if name in ['Linear',  'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Ensure predictions are non-negative
            y_pred = np.maximum(0, y_pred)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
            
            print(f"{name} model performance: MAE = {mae:.2f}, RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")
            
            # Store results
            model_results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'y_pred': y_pred,
                'model_type': 'ML'
            }
            
            # Get feature importance for tree-based models
            if name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                model_results[name]['feature_importance'] = feature_importance
                
                print("\nTop 10 most important features:")
                print(feature_importance.head(10))
                
        except Exception as e:
            print(f"Error training {name} model: {e}")
    
    # Now train time series models
    print("\nTraining time series models...")
    
    # For time series models, we'll use the total demand across all box types
    # Aggregate data by week
    weekly_total = df.groupby('week')['box_orders'].sum().reset_index()
    weekly_total = weekly_total.sort_values('week')
    
    # Split into train and test
    ts_train_size = int(len(weekly_total) * 0.8)
    ts_train = weekly_total.iloc[:ts_train_size]
    ts_test = weekly_total.iloc[ts_train_size:]
    
    # 1. ARIMA model
    try:
        print("\nTraining ARIMA model...")
        
        # Convert to time series
        train_ts = pd.Series(ts_train['box_orders'].values, index=ts_train['week'])
        test_ts = pd.Series(ts_test['box_orders'].values, index=ts_test['week'])
        
        # Fit ARIMA model
        arima_model = ARIMA(train_ts, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        
        # Forecast
        arima_forecast = arima_fit.forecast(steps=len(test_ts))
        
        # Ensure non-negative forecasts
        arima_forecast = np.maximum(0, arima_forecast)
        
        # Calculate metrics
        arima_mae = mean_absolute_error(test_ts, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_forecast))
        arima_mape = np.mean(np.abs((test_ts - arima_forecast) / (test_ts + 1))) * 100
        
        print(f"ARIMA model performance: MAE = {arima_mae:.2f}, RMSE = {arima_rmse:.2f}, MAPE = {arima_mape:.2f}%")
        
        # Store results
        model_results['ARIMA'] = {
            'model': arima_fit,
            'mae': arima_mae,
            'rmse': arima_rmse,
            'mape': arima_mape,
            'y_pred': arima_forecast,
            'model_type': 'TS'
        }
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
    

    # 3. Prophet model
    try:
        print("\nTraining global Prophet model...")
        
        # Prepare data for Prophet
        prophet_train = ts_train.rename(columns={'week': 'ds', 'box_orders': 'y'})
        prophet_test = ts_test.rename(columns={'week': 'ds', 'box_orders': 'y'})
        
        # Check if we have enough data
        if len(prophet_train) >= 10:  # Prophet needs some minimum amount of data
            # Initialize and fit Prophet model
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Add holiday effects if available
            if 'holiday_week' in df.columns:
                # Create holiday dataframe
                holidays = df[df['holiday_week'] == 1][['week']].rename(columns={'week': 'ds'})
                holidays['holiday'] = 'holiday'
                prophet_model.add_country_holidays(country_name='UK')
            
            prophet_model.fit(prophet_train)
            
            # Create future dataframe for prediction
            future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='W')
            forecast = prophet_model.predict(future)
            
            # Extract predictions for test period
            prophet_forecast = forecast.iloc[-len(prophet_test):]['yhat'].values
            
            # Ensure non-negative forecasts
            prophet_forecast = np.maximum(0, prophet_forecast)
            
            # Calculate metrics
            prophet_mae = mean_absolute_error(prophet_test['y'], prophet_forecast)
            prophet_rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_forecast))
            prophet_mape = np.mean(np.abs((prophet_test['y'] - prophet_forecast) / (prophet_test['y'] + 1))) * 100
            
            print(f"Prophet model performance: MAE = {prophet_mae:.2f}, RMSE = {prophet_rmse:.2f}, MAPE = {prophet_mape:.2f}%")
            
            # Store results
            model_results['Prophet'] = {
                'model': prophet_model,
                'mae': prophet_mae,
                'rmse': prophet_rmse,
                'mape': prophet_mape,
                'y_pred': prophet_forecast,
                'model_type': 'TS',
                'forecast': forecast
            }
        else:
            print("Not enough data for Prophet model")
    except Exception as e:
        print(f"Error training global Prophet model: {e}")
    
    # Find best model
    if model_results:
        best_model_name = min(model_results, key=lambda x: model_results[x]['mae'])
        best_model = model_results[best_model_name]['model']
        best_model_type = model_results[best_model_name]['model_type']
        
        print(f"\nBest model: {best_model_name} (Type: {best_model_type})")
        print(f"MAE: {model_results[best_model_name]['mae']:.2f}")
        print(f"RMSE: {model_results[best_model_name]['rmse']:.2f}")
        print(f"MAPE: {model_results[best_model_name]['mape']:.2f}%")
    else:
        print("No successful models")
        return None, None, None
    
    # Generate forecasts for each box type
    forecasts = {}
    
    # Calculate the historical proportion of each box type
    box_proportions = {}
    total_orders = df.groupby('box_type')['box_orders'].sum()
    total_all_orders = total_orders.sum()
    box_types = total_orders.index
    for box_type in box_types:
        box_proportions[box_type] = total_orders[box_type] / total_all_orders
        print(f"Historical proportion for {box_type}: {box_proportions[box_type]:.2%}")
    
    # Generate forecasts
    if best_model_type == 'ML':
        # For ML models, forecast each box type separately
        for box_type in box_types:
            print(f"\nGenerating forecast for {box_type}...")
            
            # Get the last data point for this box type
            last_data = test_data_with_features[test_data_with_features['box_type'] == box_type].tail(1).copy()
            
            if len(last_data) == 0:
                print(f"No test data available for {box_type}. Using last training data point.")
                last_data = train_data_with_features[train_data_with_features['box_type'] == box_type].tail(1).copy()
            
            if len(last_data) == 0:
                print(f"No data available for {box_type}. Using proportion-based forecast.")
                # If we still don't have data, use the proportion method
                if best_model_type == 'TS':
                    # For time series models, use the total forecast and apply proportion
                    if best_model_name == 'Prophet':
                        # For Prophet, we need to create a future dataframe
                        future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='W')
                        forecast = prophet_model.predict(future)
                        total_forecast = forecast.tail(forecast_periods)['yhat'].values
                    elif best_model_name == 'ARIMA':
                        # For ARIMA, use forecast method
                        total_forecast = arima_fit.forecast(steps=forecast_periods)
                    
                    else:
                        # Default case
                        total_forecast = np.array([total_orders.mean()] * forecast_periods)
                    
                    # Apply proportion
                    box_forecast = total_forecast * box_proportions[box_type]
                    forecasts[box_type] = list(np.maximum(0, box_forecast))
                continue
            
            # Initialize forecast list
            future_forecast = []
            
            # Make forecasts for each future period
            future_data = last_data.copy()
            
            for i in range(forecast_periods):
                # Update date-related features
                future_date = future_data['week'].iloc[0] + pd.Timedelta(weeks=i+1)
                future_data['week'] = future_date
                future_data['month'] = future_date.month
                future_data['day_of_year'] = future_date.dayofyear
                future_data['week_of_year'] = future_date.isocalendar()[1]
                future_data['quarter'] = (future_date.month - 1) // 3 + 1
                
                
                # Update lag features with previous predictions
                if i > 0:
                    for lag in range(1, 5):
                        lag_col = f'lag_{lag}'
                        if lag_col in future_data.columns:
                            if lag == 1:
                                future_data[lag_col] = future_forecast[-1]
                            elif i >= lag:
                                future_data[lag_col] = future_forecast[i-lag]
                    
                    # Update rolling features
                    if 'rolling_mean_4' in future_data.columns and i >= 3:
                        recent_values = future_forecast[i-4:i]
                        future_data['rolling_mean_4'] = np.mean(recent_values)
                        future_data['rolling_std_4'] = np.std(recent_values)
                        future_data['rolling_min_4'] = np.min(recent_values)
                        future_data['rolling_max_4'] = np.max(recent_values)
                
                # Make prediction
                X_future = future_data[feature_names]
                
                # Scale if needed
                if best_model_name in ['Linear', 'KNN']:
                    X_future_scaled = scaler.transform(X_future)
                    prediction = best_model.predict(X_future_scaled)[0]
                else:
                    prediction = best_model.predict(X_future)[0]
                    
                prediction = max(0, prediction)  # Ensure non-negative
                future_forecast.append(prediction)
            
            # Store forecast
            forecasts[box_type] = future_forecast
    else:
        # For time series models, use the total forecast and apply proportion
        if best_model_name == 'Prophet':
            # For Prophet, we need to create a future dataframe
            future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='W')
            forecast = prophet_model.predict(future)
            total_forecast = forecast.tail(forecast_periods)['yhat'].values
        elif best_model_name == 'ARIMA':
            # For ARIMA, use forecast method
            total_forecast = arima_fit.forecast(steps=forecast_periods)
        
        # Apply proportion to each box type
        for box_type in box_types:
            box_forecast = total_forecast * box_proportions[box_type]
            forecasts[box_type] = list(np.maximum(0, box_forecast))
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Model comparison
        plt.subplot(3, 1, 1)
        
        # Prepare data for plotting
        model_names = list(model_results.keys())
        mae_values = [model_results[model]['mae'] for model in model_names]
        rmse_values = [model_results[model]['rmse'] for model in model_names]
        mape_values = [model_results[model]['mape'] for model in model_names]
        
        # Create DataFrame for easy plotting
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'MAE': mae_values,
            'RMSE': rmse_values,
            'MAPE (%)': mape_values,
            'Type': [model_results[model]['model_type'] for model in model_names]
        })
        
        # Sort by MAE for better visualization
        metrics_df = metrics_df.sort_values('MAE')
        
        # Plot metrics
        ax = plt.subplot(2, 1, 1)
        bar_width = 0.25
        index = np.arange(len(metrics_df))
        
        # Color bars by model type
        colors = {'ML': 'skyblue', 'TS': 'lightgreen'}
        bar_colors = [colors[t] for t in metrics_df['Type']]
        
        plt.bar(index, metrics_df['MAE'], bar_width, label='MAE', color=bar_colors)
        plt.bar(index + bar_width, metrics_df['RMSE'], bar_width, label='RMSE', color=bar_colors, alpha=0.7)
        plt.bar(index + 2*bar_width, metrics_df['MAPE (%)'], bar_width, label='MAPE (%)', color=bar_colors, alpha=0.5)
        
        plt.xlabel('Model')
        plt.ylabel('Error Metric')
        plt.title('Global Model Performance Comparison')
        plt.xticks(index + bar_width, metrics_df['Model'], rotation=45)
        plt.legend()
        
        # Add model type legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors['ML'], label='Machine Learning'),
                          Patch(facecolor=colors['TS'], label='Time Series')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Plot 2: Test predictions
        plt.subplot(2, 1, 2)
        
        if best_model_type == 'ML':
            # For ML models, show predictions by box type
            box_types_in_test = test_data_with_features['box_type'].unique()
            
            for box_type in box_types_in_test:
                box_test = test_data_with_features[test_data_with_features['box_type'] == box_type]
                plt.plot(box_test['week'], box_test['box_orders'], 'o-', label=f'{box_type} Actual')
                
                # Get predictions for this box type
                box_indices = box_test.index
                box_preds = model_results[best_model_name]['y_pred'][np.isin(X_test.index, box_indices)]
                
                plt.plot(box_test['week'], box_preds, 'x--', label=f'{box_type} Predicted')
        else:
            # For time series models, show total predictions
            plt.plot(ts_test['week'], ts_test['box_orders'], 'o-', label='Actual Total')
            plt.plot(ts_test['week'], model_results[best_model_name]['y_pred'], 'x--', label='Predicted Total')
        
        plt.title(f'Actual vs Predicted Values (Test Set) - {best_model_name} Model')
        plt.xlabel('Week')
        plt.ylabel('Box Orders')
        plt.legend()
        plt.grid(True)
            # Create a separate figure with subplots for each box type
        if best_model_type == 'ML':
            box_types_in_test = test_data_with_features['box_type'].unique()
            n_box_types = len(box_types_in_test)
            
            # Create a new figure for detailed box type analysis
            plt.figure(figsize=(12, 3 * n_box_types))
            
            for i, box_type in enumerate(box_types_in_test):
                plt.subplot(n_box_types, 1, i+1)
                
                box_test = test_data_with_features[test_data_with_features['box_type'] == box_type]
                plt.plot(box_test['week'], box_test['box_orders'], 'o-', label=f'Actual')
                
                # Get predictions for this box type
                box_indices = box_test.index
                box_preds = model_results[best_model_name]['y_pred'][np.isin(X_test.index, box_indices)]
                
                plt.plot(box_test['week'], box_preds, 'x--', label=f'Predicted')
                
                plt.title(f'Box Type: {box_type} - Actual vs Predicted Values')
                plt.xlabel('Week')
                plt.ylabel('Box Orders')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('box_type_predictions_detail.png')
            print("Detailed box type predictions visualization saved as box_type_predictions_detail.png")
            
            # Return to the main figure for the remaining plots
            plt.figure(figsize=(15, 15))    

        # Create a second figure for model components (if Prophet)
        if best_model_name == 'Prophet':
            try:
                plt.figure(figsize=(15, 10))
                model_results['Prophet']['model'].plot_components(model_results['Prophet']['forecast'])
                plt.tight_layout()
                plt.savefig('prophet_components.png')
                print("Prophet components visualization saved as prophet_components.png")
            except Exception as e:
                print(f"Error creating Prophet components visualization: {e}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Get feature importance for the best model
    if best_model_type == 'ML':
        if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            feature_importance = model_results[best_model_name]['feature_importance']
        elif best_model_name in ['Linear']:
            # For linear models, use coefficients as feature importance
            coeffs = best_model.coef_
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coeffs)  # Use absolute values
            }).sort_values('Importance', ascending=False)
        else:
            # For models without clear feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': [0] * len(feature_names)  # Placeholder
            })
            print(f"Note: {best_model_name} doesn't provide direct feature importance.")
    else:
        # Time series models don't have feature importance in the same way
        feature_importance = pd.DataFrame({
            'Feature': ['trend', 'seasonality', 'residual'],
            'Importance': [0.33, 0.33, 0.33]  # Placeholder
        })
        print(f"Note: Time series models don't provide feature importance like ML models.")
    
    # Prepare results for output
    results = {
        'model': best_model,
        'model_name': best_model_name,
        'model_type': best_model_type,
        'all_models': model_results,
        'mae': model_results[best_model_name]['mae'],
        'rmse': model_results[best_model_name]['rmse'],
        'mape': model_results[best_model_name]['mape'],
        'forecasts': forecasts,
        'feature_importance': feature_importance
    }
    
    return best_model, results, feature_importance

def generate_forecasts(model, data, box_types, forecast_periods=4):
    """
    Generate forecasts for each box type using the provided model
    
    Parameters:
    -----------
    model : trained model object
        The trained forecasting model
    data : DataFrame
        The processed data containing features
    box_types : list
        List of box types to generate forecasts for
    forecast_periods : int
        Number of periods to forecast ahead
        
    Returns:
    --------
    dict
        Dictionary with box types as keys and forecast lists as values
    """
    print(f"\nGenerating forecasts for {len(box_types)} box types...")
    
    # Initialize forecasts dictionary
    forecasts = {}
    
    # Check if data has box_type column or one-hot encoded columns
    has_box_type_column = 'box_type' in data.columns
    
    # If one-hot encoded, calculate historical proportions differently
    if not has_box_type_column:
        # Check for box_* columns
        box_columns = [col for col in data.columns if col.startswith('box_')]
        
        if not box_columns:
            raise ValueError("Data must contain either 'box_type' column or 'box_*' columns")
        
        # Calculate historical proportions based on available data
        box_proportions = {}
        
        # For each box type, find rows where that box type is True
        for box_type in box_types:
            box_col = f'box_{box_type}'
            if box_col in data.columns:
                # Get rows for this box type
                box_data = data[data[box_col] == True]
                if 'box_orders' in box_data.columns and not box_data['box_orders'].isna().all():
                    # Calculate proportion based on sum of orders
                    box_proportions[box_type] = box_data['box_orders'].sum()
                else:
                    # If no order data, use count of rows
                    box_proportions[box_type] = len(box_data)
            else:
                # If column doesn't exist, use default proportion
                box_proportions[box_type] = 1
        
        # Normalize proportions
        total_all_orders = sum(box_proportions.values())
        if total_all_orders > 0:
            for box_type in box_types:
                box_proportions[box_type] = box_proportions[box_type] / total_all_orders
                print(f"Historical proportion for {box_type}: {box_proportions[box_type]:.2%}")
    else:
        # Original code for when box_type column exists
        box_proportions = {}
        total_orders = data.groupby('box_type')['box_orders'].sum()
        total_all_orders = total_orders.sum()
        
        for box_type in box_types:
            if box_type in total_orders:
                box_proportions[box_type] = total_orders[box_type] / total_all_orders
            else:
                # Default proportion if box type not in data
                box_proportions[box_type] = 1 / len(box_types)
            print(f"Historical proportion for {box_type}: {box_proportions[box_type]:.2%}")
    
    # Check model type
    model_type = type(model).__name__
    
    # For time series models like Prophet or ARIMA
    if model_type in ['Prophet', 'ARIMAResults']:
        print(f"Generating forecasts using {model_type} model...")
        
        if model_type == 'Prophet':
            # For Prophet, create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods, freq='W')
            forecast = model.predict(future)
            total_forecast = forecast.tail(forecast_periods)['yhat'].values
        else:
            # For ARIMA, use forecast method
            total_forecast = model.forecast(steps=forecast_periods)
        
        # Apply proportion to each box type
        for box_type in box_types:
            box_forecast = total_forecast * box_proportions[box_type]
            forecasts[box_type] = list(np.maximum(0, box_forecast))
    
    # For ML models
    else:
        print(f"Generating forecasts using {model_type} model...")
        
        for box_type in box_types:
            print(f"\nGenerating forecast for {box_type}...")
            
            # Get the last data point for this box type
            if has_box_type_column:
                last_data = data[data['box_type'] == box_type].tail(1).copy()
            else:
                box_col = f'box_{box_type}'
                if box_col in data.columns:
                    last_data = data[data[box_col] == True].tail(1).copy()
                    if len(last_data) == 0:
                        # If no data with True value, take the last row and set this box type to True
                        last_data = data.tail(1).copy()
                        for col in [c for c in data.columns if c.startswith('box_')]:
                            last_data[col] = False
                        last_data[box_col] = True
                else:
                    # If column doesn't exist, use the last row and create the column
                    last_data = data.tail(1).copy()
                    for col in [c for c in data.columns if c.startswith('box_')]:
                        last_data[col] = False
                    last_data[box_col] = True
            
            if len(last_data) == 0:
                print(f"No data available for {box_type}. Using proportion-based forecast.")
                # Use the proportion method as fallback
                if 'total_forecast' not in locals():
                    # Generate a simple forecast based on average
                    avg_orders = data['box_orders'].mean() if 'box_orders' in data.columns and not data['box_orders'].isna().all() else 100
                    total_forecast = np.array([avg_orders] * forecast_periods)
                
                box_forecast = total_forecast * box_proportions[box_type]
                forecasts[box_type] = list(np.maximum(0, box_forecast))
                continue
            
            # Initialize forecast list
            future_forecast = []
            
            # Get feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                # For models without feature_names_in_ attribute
                feature_names = [col for col in data.columns 
                               if col not in ['week', 'box_type', 'box_orders']]
            
            # Make forecasts for each future period
            future_data = last_data.copy()
            
            for i in range(forecast_periods):
                # Update date-related features
                if 'week' in future_data.columns:
                    future_date = future_data['week'].iloc[0] + pd.Timedelta(weeks=i+1)
                    future_data['week'] = future_date
                
                    # Update time features
                    if 'month' in feature_names:
                        future_data['month'] = future_date.month
                    if 'day_of_year' in feature_names:
                        future_data['day_of_year'] = future_date.dayofyear
                    if 'week_of_year' in feature_names:
                        future_data['week_of_year'] = future_date.isocalendar()[1]
                    if 'quarter' in feature_names:
                        future_data['quarter'] = (future_date.month - 1) // 3 + 1
                    if 'day_of_week' in feature_names:
                        future_data['day_of_week'] = future_date.weekday()
                    
                    # Update cyclical features if they exist
                    if 'month_sin' in feature_names:
                        future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
                        future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
                    if 'week_of_year_sin' in feature_names:
                        future_data['week_of_year_sin'] = np.sin(2 * np.pi * future_data['week_of_year'] / 53)
                        future_data['week_of_year_cos'] = np.cos(2 * np.pi * future_data['week_of_year'] / 53)
                
                # Update lag features with previous predictions
                if i > 0:
                    for lag in range(1, 5):
                        lag_col = f'lag_{lag}'
                        if lag_col in feature_names:
                            if lag == 1:
                                future_data[lag_col] = future_forecast[-1]
                            elif i >= lag:
                                future_data[lag_col] = future_forecast[i-lag]
                    
                    # Update rolling features
                    if 'rolling_mean_4' in feature_names and i >= 3:
                        recent_values = future_forecast[i-4:i]
                        future_data['rolling_mean_4'] = np.mean(recent_values)
                        if 'rolling_std_4' in feature_names:
                            future_data['rolling_std_4'] = np.std(recent_values)
                        if 'rolling_min_4' in feature_names:
                            future_data['rolling_min_4'] = np.min(recent_values)
                        if 'rolling_max_4' in feature_names:
                            future_data['rolling_max_4'] = np.max(recent_values)
                
                # Select only the features the model knows about
                X_future = future_data[feature_names]
                
                # Make prediction
                prediction = model.predict(X_future)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                future_forecast.append(prediction)
            
            # Store forecast
            forecasts[box_type] = future_forecast
    
    return forecasts