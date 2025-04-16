import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def process_data(input_file, output_file='data.csv'):
    """
    Process raw data and prepare it for the demand forecasting model.
    Includes feature engineering.
    
    Parameters:
    -----------
    input_file : str
        Path to the raw data file
    output_file : str
        Path to save the processed data (default: 'data.csv')
    """

    df = pd.read_csv(input_file)

    
    # Display initial data info
    print(f"Raw data shape: {df.shape}")
    
    # Convert week to datetime if it's not already
    if 'week' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['week']):
        df['week'] = pd.to_datetime(df['week'])
    
    # Fix data types
    # Convert box_orders to numeric, handling any potential errors
    if 'box_orders' in df.columns:
        # Check for any non-numeric values (like "1O0" instead of "100")
        df['box_orders'] = df['box_orders'].astype(str).str.replace('O', '0')  # Replace letter O with number 0
        df['box_orders'] = pd.to_numeric(df['box_orders'], errors='coerce')
    
    # Convert subscriber columns to numeric
    if 'weekly_subscribers' in df.columns:
        df['weekly_subscribers'] = pd.to_numeric(df['weekly_subscribers'], errors='coerce')
    
    if 'fortnightly_subscribers' in df.columns:
        df['fortnightly_subscribers'] = pd.to_numeric(df['fortnightly_subscribers'], errors='coerce')
    
    # Convert boolean columns
    if 'is_marketing_week' in df.columns:
        df['is_marketing_week'] = df['is_marketing_week'].astype(bool)
    
    if 'holiday_week' in df.columns:
        df['holiday_week'] = df['holiday_week'].astype(bool)
    
    # Handle missing values in subscriber columns
    if 'weekly_subscribers' in df.columns:
        df['weekly_subscribers'] = df['weekly_subscribers'].fillna(method='ffill')
    
    if 'fortnightly_subscribers' in df.columns:
        # Replace NaN with the median of the column
        median_value = df['fortnightly_subscribers'].median()
        df['fortnightly_subscribers'] = df['fortnightly_subscribers'].fillna(median_value)
    
    # Ensure data is sorted by week and box_type
    df = df.sort_values(['box_type', 'week'])
    
    # ===== FEATURE ENGINEERING =====
    print("Performing feature engineering...")
    
    # Process each box type separately to create features
    box_types = df['box_type'].unique()
    all_data = []
    
    for box_type in box_types:
        print(f"Processing features for box type: {box_type}")
        box_data = df[df['box_type'] == box_type].copy()
        
        # Add time-based features
        box_data['month'] = box_data['week'].dt.month
        box_data['week_of_year'] = box_data['week'].dt.isocalendar().week
        box_data['quarter'] = box_data['week'].dt.quarter
        box_data['day_of_year'] = box_data['week'].dt.dayofyear

        # Create subscriber ratio
        box_data['subscriber_ratio'] = box_data['weekly_subscribers'] / (box_data['fortnightly_subscribers'] + 1)
        
        # Create seasonal indicators
        box_data['is_summer'] = ((box_data['month'] >= 6) & (box_data['month'] <= 8)).astype(int)
        box_data['is_winter'] = ((box_data['month'] >= 12) | (box_data['month'] <= 2)).astype(int)
        box_data['is_spring'] = ((box_data['month'] >= 3) & (box_data['month'] <= 5)).astype(int)
        box_data['is_fall'] = ((box_data['month'] >= 9) & (box_data['month'] <= 11)).astype(int)
        
        # Convert boolean columns to int for modeling
        box_data['is_marketing_week'] = box_data['is_marketing_week'].astype(int)
        box_data['holiday_week'] = box_data['holiday_week'].astype(int)
        
        # Add to the list
        all_data.append(box_data)
    
    # Combine all processed data
    processed_df = pd.concat(all_data)
    
    # Check for missing values after processing
    missing_values = processed_df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nRemaining missing values after processing:")
        print(missing_values[missing_values > 0])
        print("Note: Some missing values in lag and rolling features are expected at the beginning of time series.")
    
    # Save processed data
    processed_df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    print(f"Final data shape: {processed_df.shape}")
    
    return processed_df

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Process raw data for demand forecasting')
#     parser.add_argument('input_file', help='Path to the raw data file (CSV or Excel)')
#     parser.add_argument('--output', default='data.csv', help='Path to save the processed data (default: data.csv)')
    
#     args = parser.parse_args()
    
#     # Process the data
#     processed_data = process_data(args.input_file, args.output)
    
#     print("\nData processing complete. You can now run the demand forecasting model.")