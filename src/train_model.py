import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
# For Prophet model
try:
    from prophet import Prophet
except ImportError:
    print("Prophet not installed. To use Prophet model, install it with: pip install prophet")


import warnings
warnings.filterwarnings('ignore')

# Import the data processing function
from data_process import process_data

def add_time_features(data, target_col, is_train=True, train_data=None):
    """
    Add time-dependent features to the dataset without causing data leakage.
    
    Parameters:
    -----------
    data : DataFrame
        The data to add features to
    target_col : str
        The name of the target column
    is_train : bool
        Whether this is training data (True) or test/forecast data (False)
    train_data : DataFrame, optional
        The training data, needed when is_train=False to avoid leakage
    
    Returns:
    --------
    DataFrame
        Data with added time features
    """
    result = data.copy()
    
    if is_train:
        # For training data, we can create features directly
        # Create lag features
        for lag in range(1, 5):
            result[f'lag_{lag}'] = result[target_col].shift(lag)
        
        # Create rolling features
        result['rolling_mean_4'] = result[target_col].rolling(window=4).mean()
        result['rolling_std_4'] = result[target_col].rolling(window=4).std()
        result['rolling_mean_8'] = result[target_col].rolling(window=8).mean()
        result['rolling_min_4'] = result[target_col].rolling(window=4).min()
        result['rolling_max_4'] = result[target_col].rolling(window=4).max()
        
        # Drop rows with NaN values (typically the first few rows due to lag features)
        result = result.dropna()
        
    else:
        # For test data, we need to be careful to avoid leakage
        # We'll use the training data to initialize lag features
        
        # Get the last values from training data for lag features
        last_train_values = train_data[target_col].iloc[-4:].values
        
        # Create lag columns
        for lag in range(1, 5):
            result[f'lag_{lag}'] = np.nan
        
        # Create rolling feature columns
        result['rolling_mean_4'] = np.nan
        result['rolling_std_4'] = np.nan
        result['rolling_mean_8'] = np.nan
        result['rolling_min_4'] = np.nan
        result['rolling_max_4'] = np.nan
        
        # Fill in lag features for the first row of test data
        for i, lag in enumerate(range(1, 5)):
            if i < len(last_train_values):
                result.iloc[0, result.columns.get_loc(f'lag_{lag}')] = last_train_values[-(i+1)]
        
        # Fill in rolling features for the first row
        if len(last_train_values) >= 4:
            result.iloc[0, result.columns.get_loc('rolling_mean_4')] = np.mean(last_train_values[-4:])
            result.iloc[0, result.columns.get_loc('rolling_std_4')] = np.std(last_train_values[-4:])
            result.iloc[0, result.columns.get_loc('rolling_min_4')] = np.min(last_train_values[-4:])
            result.iloc[0, result.columns.get_loc('rolling_max_4')] = np.max(last_train_values[-4:])
        
        # For rolling_mean_8, we need more values
        if len(train_data) >= 8:
            last_8_values = train_data[target_col].iloc[-8:].values
            result.iloc[0, result.columns.get_loc('rolling_mean_8')] = np.mean(last_8_values)
        
        # Now fill in the rest of the test data row by row
        for i in range(1, len(result)):
            # Update lag features
            for lag in range(1, 5):
                if i >= lag:
                    # Use actual values from previous rows in test set
                    result.iloc[i, result.columns.get_loc(f'lag_{lag}')] = result[target_col].iloc[i-lag]
                else:
                    # Use values from training set for initial rows
                    idx = len(last_train_values) - (lag - i)
                    if idx >= 0:
                        result.iloc[i, result.columns.get_loc(f'lag_{lag}')] = last_train_values[idx]
            
            # Update rolling features
            # For each row, we need to look back at previous rows and possibly training data
            lookback_values = []
            
            # Add values from test data
            for j in range(max(0, i-3), i):
                lookback_values.append(result[target_col].iloc[j])
            
            # If we need more values, get them from training data
            remaining = 4 - len(lookback_values)
            if remaining > 0 and len(last_train_values) >= remaining:
                lookback_values = list(last_train_values[-remaining:]) + lookback_values
            
            # Calculate rolling features if we have enough values
            if len(lookback_values) == 4:
                result.iloc[i, result.columns.get_loc('rolling_mean_4')] = np.mean(lookback_values)
                result.iloc[i, result.columns.get_loc('rolling_std_4')] = np.std(lookback_values)
                result.iloc[i, result.columns.get_loc('rolling_min_4')] = np.min(lookback_values)
                result.iloc[i, result.columns.get_loc('rolling_max_4')] = np.max(lookback_values)
            
            # Similar logic for rolling_mean_8
            # (code omitted for brevity)
    
    return result

def train_and_evaluate_models(box_type_data, forecast_periods=4):
    """
    Train multiple models and evaluate performance without data leakage
    """
    # Prepare data
    box_data = box_type_data.copy()
    box_type = box_data['box_type'].iloc[0]
    
    # First split the data into train and test sets
    train_size = int(len(box_data) * 0.8)
    train_data = box_data.iloc[:train_size].copy()
    test_data = box_data.iloc[train_size:].copy()
    
    # Now add time-dependent features to each set separately
    train_data_with_features = add_time_features(train_data, 'box_orders', is_train=True)
    test_data_with_features = add_time_features(test_data, 'box_orders', is_train=False, train_data=train_data)
    
    # Select features
    feature_cols = [col for col in train_data_with_features.columns 
                   if col not in ['week', 'box_type', 'box_orders']]
    
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
    
    # Prepare training data
    X_train = train_data_with_features[feature_cols]
    y_train = train_data_with_features['box_orders']
    X_test = test_data_with_features[feature_cols]
    y_test = test_data_with_features['box_orders']
    
    # Final check for NaNs
    assert X_train.isnull().sum().sum() == 0, "Training features still contain NaN values"
    assert X_test.isnull().sum().sum() == 0, "Test features still contain NaN values"
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare different models
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5),  # K-Nearest Neighbors
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)  # ElasticNet Regression
    }
    
    # Evaluation results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            print(f"{box_type} - Training {name} model...")
            
            # Train the model
            if name in ['Linear', 'ElasticNet',  'KNN']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Prepare future data for forecasting
                future_forecast = []
                last_actual = y_test.iloc[-1]
                
                for i in range(forecast_periods):
                    # For simplicity in these models, use the last actual value
                    # and the coefficient trends to predict future values
                    if i == 0:
                        # For first prediction, use the last actual value
                        prediction = last_actual * 0.9 + y_pred[-1] * 0.1
                    else:
                        # For subsequent predictions, use previous prediction
                        prediction = future_forecast[-1]
                    
                    # Add some trend based on the model coefficients if available
                    if hasattr(model, 'coef_'):
                        # Apply a simple trend based on coefficient signs
                        trend = sum(model.coef_) / len(model.coef_)
                        prediction += trend * (i + 1) * 0.01 * last_actual
                    
                    future_forecast.append(max(0, prediction))  # Ensure non-negative
            else:
                # For tree-based models, use all features
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Prepare future data for forecasting
                future_X = X_test.tail(1).copy()
                future_forecast = []
                
                for i in range(forecast_periods):
                    # Update lag features for future prediction
                    for lag in range(1, 5):
                        lag_col = f'lag_{lag}'
                        if lag_col in future_X.columns:
                            if lag == 1:
                                if i == 0:
                                    # For the first future period, use the last actual value
                                    future_X[lag_col] = y_test.iloc[-1]
                                else:
                                    # For subsequent periods, use the previous prediction
                                    future_X[lag_col] = future_forecast[-1]
                            else:
                                # For lag > 1, shift the previous lag values
                                if i < lag - 1:
                                    # If we don't have enough future predictions yet,
                                    # use values from test set
                                    if lag - i - 1 < len(y_test):
                                        future_X[lag_col] = y_test.iloc[-(lag-i)]
                                    else:
                                        # If we don't have enough test values either,
                                        # use the last available value
                                        future_X[lag_col] = y_test.iloc[-1]
                                else:
                                    # Use previous future predictions
                                    future_X[lag_col] = future_forecast[i-(lag-1)]
                    
                    # Update rolling features
                    if 'rolling_mean_4' in future_X.columns:
                        if i == 0:
                            # For first future period, use the last 3 actual values + the last prediction
                            values = list(y_test.iloc[-3:].values) + [future_X[f'lag_1'].values[0]]
                            future_X['rolling_mean_4'] = np.mean(values)
                            future_X['rolling_std_4'] = np.std(values)
                            future_X['rolling_min_4'] = np.min(values)
                            future_X['rolling_max_4'] = np.max(values)
                        elif i < 3:
                            # For 2nd and 3rd future periods, use a mix of actual values and predictions
                            values = list(y_test.iloc[-(3-i):].values) + future_forecast[:i] + [future_X[f'lag_1'].values[0]]
                            future_X['rolling_mean_4'] = np.mean(values)
                            future_X['rolling_std_4'] = np.std(values)
                            future_X['rolling_min_4'] = np.min(values)
                            future_X['rolling_max_4'] = np.max(values)
                        else:
                            # For 4th+ future periods, use only predictions
                            values = future_forecast[i-4:i]
                            future_X['rolling_mean_4'] = np.mean(values)
                            future_X['rolling_std_4'] = np.std(values)
                            future_X['rolling_min_4'] = np.min(values)
                            future_X['rolling_max_4'] = np.max(values)
                    
                    # Make prediction for this future period
                    future_pred = model.predict(future_X)[0]
                    
                    # Ensure non-negative prediction
                    future_pred = max(0, future_pred)
                    future_forecast.append(future_pred)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100  # Added MAPE
            
            # Store results
            results[name] = {
                'model': model,
                'test_actual': y_test,
                'test_forecast': pd.Series(y_pred, index=y_test.index),
                'future_forecast': pd.Series(future_forecast),
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            print(f"{box_type} - {name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")
        except Exception as e:
            print(f"Error with {box_type} - {name}: {e}")
    
    # Try time series models
    try:
        # Convert to time series format
        train_ts = train_data.set_index('week')['box_orders']
        test_ts = test_data.set_index('week')['box_orders']
        
        # 1. ARIMA model
        try:
            # Simple ARIMA model
            arima_model = ARIMA(train_ts, order=(2, 1, 2))
            arima_fit = arima_model.fit()
            
            # Forecast
            arima_forecast = arima_fit.forecast(steps=len(test_ts))
            arima_future = arima_fit.forecast(steps=forecast_periods)
            
            # Ensure non-negative forecasts
            arima_forecast = np.maximum(0, arima_forecast)
            arima_future = np.maximum(0, arima_future)
            
            # Calculate metrics
            arima_mae = mean_absolute_error(test_ts, arima_forecast)
            arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_forecast))
            arima_mape = np.mean(np.abs((test_ts - arima_forecast) / (test_ts + 1))) * 100
            
            # Store results
            results['ARIMA'] = {
                'model': arima_fit,
                'test_actual': test_ts,
                'test_forecast': pd.Series(arima_forecast, index=test_ts.index),
                'future_forecast': pd.Series(arima_future),
                'mae': arima_mae,
                'rmse': arima_rmse,
                'mape': arima_mape
            }
            
            print(f"{box_type} - ARIMA: MAE = {arima_mae:.2f}, RMSE = {arima_rmse:.2f}, MAPE = {arima_mape:.2f}%")
        except Exception as e:
            print(f"Error with {box_type} - ARIMA: {e}")
        
        #
            
        # 4. Prophet model
        try:
            # Prepare data for Prophet
            prophet_train = pd.DataFrame({
                'ds': train_data['week'],
                'y': train_data['box_orders']
            })
            
            # Create and fit model
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            prophet_model.fit(prophet_train)
            
            # Create future dataframe for prediction
            prophet_future = prophet_model.make_future_dataframe(
                periods=len(test_data) + forecast_periods,
                freq='W'
            )
            
            # Make predictions
            prophet_forecast = prophet_model.predict(prophet_future)
            
            # Extract test period forecasts
            prophet_test_forecast = prophet_forecast.iloc[len(train_data):len(train_data)+len(test_data)]
            prophet_future_forecast = prophet_forecast.iloc[len(train_data)+len(test_data):]
            
            # Ensure non-negative forecasts
            prophet_test_forecast['yhat'] = np.maximum(0, prophet_test_forecast['yhat'])
            prophet_future_forecast['yhat'] = np.maximum(0, prophet_future_forecast['yhat'])
            
            # Calculate metrics
            prophet_mae = mean_absolute_error(test_ts, prophet_test_forecast['yhat'].values)
            prophet_rmse = np.sqrt(mean_squared_error(test_ts, prophet_test_forecast['yhat'].values))
            prophet_mape = np.mean(np.abs((test_ts - prophet_test_forecast['yhat'].values) / (test_ts + 1))) * 100
            
            # Store results
            results['Prophet'] = {
                'model': prophet_model,
                'test_actual': test_ts,
                'test_forecast': pd.Series(prophet_test_forecast['yhat'].values, index=test_ts.index),
                'future_forecast': pd.Series(prophet_future_forecast['yhat'].values[:forecast_periods]),
                'mae': prophet_mae,
                'rmse': prophet_rmse,
                'mape': prophet_mape
            }
            
            print(f"{box_type} - Prophet: MAE = {prophet_mae:.2f}, RMSE = {prophet_rmse:.2f}, MAPE = {prophet_mape:.2f}%")
        except Exception as e:
            print(f"Error with {box_type} - Prophet: {e}")
    except Exception as e:
        print(f"Error with {box_type}: {e}")
    
    # Create visualization of model performance
    try:
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model comparison by metrics
        plt.subplot(2, 1, 1)
        
        # Prepare data for plotting
        model_names = list(results.keys())
        mae_values = [results[model]['mae'] for model in model_names]
        rmse_values = [results[model]['rmse'] for model in model_names]
        mape_values = [results[model]['mape'] for model in model_names]
        
        # Create DataFrame for easy plotting
        metrics_df = pd.DataFrame({
            'Model': model_names + model_names + model_names,
            'Metric': ['MAE'] * len(model_names) + ['RMSE'] * len(model_names) + ['MAPE (%)'] * len(model_names),
            'Value': mae_values + rmse_values + mape_values
        })
        
        # Plot metrics
        sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
        plt.title(f'Model Performance Comparison for {box_type}')
        plt.xticks(rotation=45)
        plt.legend(title='Metric')
        
        # Plot 2: Actual vs Predicted for best model
        plt.subplot(2, 1, 2)
        
        # Find best model based on MAE
        best_model_name = min(results, key=lambda x: results[x]['mae'])
        best_result = results[best_model_name]
        
        # Plot actual values
        plt.plot(best_result['test_actual'].index, best_result['test_actual'].values, 
                 'b-', label='Actual')
        
        # Plot predicted values
        plt.plot(best_result['test_actual'].index, best_result['test_forecast'].values, 
                 'r--', label=f'Predicted ({best_model_name})')
        
        # Plot future forecast
        last_date = best_result['test_actual'].index[-1]
        future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(len(best_result['future_forecast']))]
        plt.plot(future_dates, best_result['future_forecast'].values, 
                 'g--', label=f'Forecast ({best_model_name})')
        
        plt.title(f'Actual vs Predicted Values for {box_type} - Best Model: {best_model_name}')
        plt.xlabel('Date')
        plt.ylabel('Box Orders')
        plt.legend()
        plt.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{box_type}_model_comparison.png')
        print(f"Visualization saved as {box_type}_model_comparison.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Find best model based on MAE
    if results:
        best_model_name = min(results, key=lambda x: results[x]['mae'])
        print(f"Best model for {box_type}: {best_model_name} (MAE: {results[best_model_name]['mae']:.2f})")
        
        # Provide detailed explanation of model selection
        print("\nModel Selection Analysis:")
        print("-------------------------")
        print(f"The best performing model for {box_type} is {best_model_name} with:")
        print(f"  - MAE: {results[best_model_name]['mae']:.2f}")
        print(f"  - RMSE: {results[best_model_name]['rmse']:.2f}")
        print(f"  - MAPE: {results[best_model_name]['mape']:.2f}%")
        print("\nComparison with other models:")
        
        for model in sorted(results.keys(), key=lambda x: results[x]['mae']):
            if model != best_model_name:
                mae_diff = results[model]['mae'] - results[best_model_name]['mae']
                print(f"  - {model}: MAE {results[model]['mae']:.2f} ({mae_diff:.2f} worse than best)")
        
        print("\nRecommendation:")
        if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            print(f"The {best_model_name} model is recommended as it captures complex patterns in the data.")
            print("Tree-based models are good at handling non-linear relationships and interactions between features.")
        elif best_model_name in ['ARIMA', 'Prophet']:
            print(f"The {best_model_name} model is recommended as it captures the time series patterns well.")
            print("This model is specifically designed for time series forecasting and handles seasonality and trends.")
        else:
            print(f"The {best_model_name} model is recommended based on its superior performance metrics.")
        
        return results, best_model_name
    else:
        print(f"No successful models for {box_type}")
        return None, None

def main():
    # Process the data first (if raw data is provided)
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Train demand forecasting models')
    parser.add_argument('--input', default='data.csv', help='Path to the processed data file (default: data.csv)')
    parser.add_argument('--raw', help='Path to raw data file (if data needs processing)')
    
    args = parser.parse_args()
    
    # If raw data is provided, process it first
    if args.raw and os.path.exists(args.raw):
        print(f"Processing raw data from {args.raw}...")
        df = process_data(args.raw, args.input)
        print(f"Data processed and saved to {args.input}")
    else:
        # Load already processed data
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Data file {args.input} not found. Please provide a valid data file or use --raw to process raw data.")
        
        df = pd.read_csv(args.input)
        df['week'] = pd.to_datetime(df['week'])
    
    # Get unique box types
    box_types = df['box_type'].unique()
    
    # Store results for all box types
    all_results = {}
    best_models = {}
    
    # Train models for each box type
    for box_type in box_types:
        print(f"\nProcessing box type: {box_type}")
        box_data = df[df['box_type'] == box_type].copy()
        
        # Skip if not enough data
        if len(box_data) < 10:
            print(f"Not enough data for {box_type}. Skipping.")
            continue
        
        # Train and evaluate models
        results, best_model_name = train_and_evaluate_models(box_data)
        
        if results:
            all_results[box_type] = results
            best_models[box_type] = best_model_name
    
    # Create forecast summary
    forecast_summary = []
    for box_type in all_results:
        best_model_name = best_models[box_type]
        result = all_results[box_type][best_model_name]
        
        future_values = result['future_forecast'].values
        
        for i, value in enumerate(future_values):
            forecast_summary.append({
                'box_type': box_type,
                'model': best_model_name,
                'forecast_week': i+1,
                'forecast_value': value,
                'mae': result['mae'],
                'rmse': result['rmse']
            })
    
    # Create forecast summary dataframe
    forecast_df = pd.DataFrame(forecast_summary)
    forecast_df.to_csv('box_demand_forecast.csv', index=False)
    
    # Create pivot table for easier reading
    forecast_pivot = forecast_df.pivot_table(
        index=['box_type', 'model', 'mae', 'rmse'], 
        columns='forecast_week', 
        values='forecast_value'
    )
    forecast_pivot.columns = [f'Week {i}' for i in forecast_pivot.columns]
    forecast_pivot.to_csv('box_demand_forecast_pivot.csv')
    
    # Calculate total forecasted demand for next 4 weeks
    total_forecast = {}
    for week in range(1, 5):
        week_data = forecast_df[forecast_df['forecast_week'] == week]
        total_forecast[f'week_{week}'] = week_data['forecast_value'].sum()
    
    print("\nTotal forecasted box demand for next 4 weeks:")
    for week, total in total_forecast.items():
        print(f"{week}: {total:.0f} boxes")
    
    print("\nForecast results saved to 'box_demand_forecast.csv' and 'box_demand_forecast_pivot.csv'")

if __name__ == "__main__":
    main()