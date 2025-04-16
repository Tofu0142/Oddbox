# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import uvicorn
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Box Order Prediction API",
    description="API for predicting box orders using a fine-tuned Gradient Boosting model",
    version="1.0.0"
)

# Load the trained model
try:
    model = joblib.load('./models/best_gradient_boosting_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define request and response models
class PredictionRequest(BaseModel):
    box_type: str
    date: str  # Format: YYYY-MM-DD
    additional_features: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    box_type: str
    predicted_orders: float
    prediction_date: str
    confidence_interval: Optional[Dict[str, float]] = None

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_version: str = "1.0.0"
    timestamp: str

# Helper functions
def add_time_features(date_str):
    """Add time-based features for prediction"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        
        features = {
            "month": date.month,
            "day_of_year": date.timetuple().tm_yday,
            "week_of_year": date.isocalendar()[1],
            "quarter": (date.month - 1) // 3 + 1,
            "day_of_week": date.weekday(),
            "is_weekend": 1 if date.weekday() >= 5 else 0,
            "is_holiday": 0  # Default value, would need a holiday calendar to set properly
        }
        
        return features
    except Exception as e:
        raise ValueError(f"Error processing date: {e}")

def prepare_features(request):
    """
    Prepare all 24 features required by the model
    """
    # Start with time features
    features = add_time_features(request.date)
    
    # Add box type one-hot encoding (assuming 5 box types)
    box_types = ["box_type_A", "box_type_B", "box_type_C", "box_type_D", "box_type_E"]
    for box_type in box_types:
        features[f"box_{box_type}"] = 1 if request.box_type == box_type else 0
    
    # Add default values for time series features
    time_series_features = {
        "lag_1": 0, 
        "lag_2": 0, 
        "lag_3": 0, 
        "lag_4": 0,
        "rolling_mean_4": 0, 
        "rolling_std_4": 0, 
        "rolling_mean_8": 0,
        "rolling_min_4": 0, 
        "rolling_max_4": 0,
        "trend": 0, 
        "seasonal": 0
    }
    features.update(time_series_features)
    
    # Add weather features
    weather_features = {
        "temperature": 20.0,  # Default temperature in Celsius
        "rainfall": 0.0       # Default rainfall in mm
    }
    features.update(weather_features)
    
    # Add any additional features provided in the request
    if request.additional_features:
        features.update(request.additional_features)
    
    # Create a DataFrame with a single row
    df = pd.DataFrame([features])
    
    # Ensure we have exactly 24 features
    if len(df.columns) < 24:
        # Add missing features with default values
        for i in range(len(df.columns), 24):
            df[f"feature_{i}"] = 0
    elif len(df.columns) > 24:
        # Keep only the first 24 features
        df = df.iloc[:, :24]
    
    print(f"Prepared features: {df.columns.tolist()}")
    print(f"Feature count: {len(df.columns)}")
    
    return df

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Box Order Prediction API", "status": "active"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict box orders for a single request
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(request)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        # Create response
        response = PredictionResponse(
            box_type=request.box_type,
            predicted_orders=round(prediction, 2),
            prediction_date=request.date,
            confidence_interval={
                "lower_bound": max(0, round(prediction * 0.9, 2)),  # Simple 10% confidence interval
                "upper_bound": round(prediction * 1.1, 2)
            }
        )
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    """
    Predict box orders for multiple requests
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for req in request.predictions:
            # Prepare features
            features_df = prepare_features(req)
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            # Create response
            predictions.append(
                PredictionResponse(
                    box_type=req.box_type,
                    predicted_orders=round(prediction, 2),
                    prediction_date=req.date,
                    confidence_interval={
                        "lower_bound": max(0, round(prediction * 0.9, 2)),
                        "upper_bound": round(prediction * 1.1, 2)
                    }
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about the model including expected features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get feature names if available
    feature_names = []
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "feature_names": feature_names
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)