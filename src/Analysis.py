import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def data_analysis(df):
# Convert week to datetime
    df['week'] = pd.to_datetime(df['week'])

    # Check for data issues
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")

    # Convert box_orders to numeric
    df['box_orders'] = pd.to_numeric(df['box_orders'], errors='coerce')

    # Check for NaN values after conversion
    print(f"NaN values in box_orders after conversion: {df['box_orders'].isna().sum()}")


    # Fill missing values in box_orders
    for box_type in df['box_type'].unique():
        median_orders = df[df['box_type'] == box_type]['box_orders'].median()
        df.loc[(df['box_type'] == box_type) & (df['box_orders'].isna()), 'box_orders'] = median_orders

    # Fill missing values in fortnightly_subscribers
    df['fortnightly_subscribers'].fillna(df['fortnightly_subscribers'].median(), inplace=True)

    # Basic statistics
    print(df.describe())

    # ===== Feature Engineering =====
    # Extract time features
    df['month'] = df['week'].dt.month
    df['week_of_year'] = df['week'].dt.isocalendar().week
    df['day_of_year'] = df['week'].dt.dayofyear
    df['quarter'] = df['week'].dt.quarter

    # Create lag features (previous 1-4 weeks orders)
    for box_type in df['box_type'].unique():
        box_data = df[df['box_type'] == box_type].sort_values('week')
        for lag in range(1, 5):
            df.loc[df['box_type'] == box_type, f'lag_{lag}'] = box_data['box_orders'].shift(lag)

    # Create rolling average features
    for box_type in df['box_type'].unique():
        box_data = df[df['box_type'] == box_type].sort_values('week')
        df.loc[df['box_type'] == box_type, 'rolling_mean_4'] = box_data['box_orders'].rolling(window=4).mean()
        df.loc[df['box_type'] == box_type, 'rolling_std_4'] = box_data['box_orders'].rolling(window=4).std()

    # Create subscriber ratio feature
    df['subscriber_ratio'] = df['weekly_subscribers'] / (df['fortnightly_subscribers'] + 1)  # Avoid division by zero

    # Create seasonality indicators
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
    df['is_winter'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)

    # Create one-hot encoding for box_type
    box_type_dummies = pd.get_dummies(df['box_type'], prefix='box')
    df = pd.concat([df, box_type_dummies], axis=1)

    # Fill missing feature values
    df = df.fillna(method='bfill').fillna(method='ffill')

    # ===== Data Visualization =====
    # Plot total box orders over time
    plt.figure(figsize=(14, 7))
    df_weekly_total = df.groupby('week')['box_orders'].sum().reset_index()
    plt.plot(df_weekly_total['week'], df_weekly_total['box_orders'], marker='o')
    plt.title('Total Box Orders Over Time', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Total Orders', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_orders_over_time.png')
    plt.close()

    # Plot orders by box type
    plt.figure(figsize=(14, 7))
    box_type_orders = df.groupby('box_type')['box_orders'].mean().sort_values(ascending=False)
    sns.barplot(x=box_type_orders.index, y=box_type_orders.values)
    plt.title('Average Orders by Box Type', fontsize=16)
    plt.xlabel('Box Type', fontsize=12)
    plt.ylabel('Average Orders', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('average_orders_by_box_type.png')
    plt.close()

    # Analyze impact of marketing weeks
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='is_marketing_week', y='box_orders', data=df)
    plt.title('Impact of Marketing on Box Orders', fontsize=16)
    plt.xlabel('Marketing Week', fontsize=12)
    plt.ylabel('Box Orders', fontsize=12)
    plt.tight_layout()
    plt.savefig('marketing_impact.png')
    plt.close()

    # Analyze impact of holiday weeks
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='holiday_week', y='box_orders', data=df)
    plt.title('Impact of Holidays on Box Orders', fontsize=16)
    plt.xlabel('Holiday Week', fontsize=12)
    plt.ylabel('Box Orders', fontsize=12)
    plt.tight_layout()
    plt.savefig('holiday_impact.png')
    plt.close()

    # Create correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_df = df[numeric_cols].copy()

    plt.figure(figsize=(16, 14))
    corr_matrix = corr_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

