import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the FastAPI app
try:
    from app.app import app, add_time_features, prepare_features  # Import directly from app.py
    from fastapi.testclient import TestClient
    # Create test client
    client = TestClient(app)
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('app'):
        print(f"app directory contents: {os.listdir('app')}")
    raise

class TestBoxPredictionAPI(unittest.TestCase):
    """Test cases for Box Prediction API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model for testing
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([42.0])
        self.mock_model.n_features_in_ = 24
        
        # Sample prediction request data
        self.sample_request = {
            "box_type": "box_type_A",
            "date": "2023-06-15",
            "additional_features": {
                "subscriber_ratio": 1.5
            }
        }
        
        # Sample batch prediction request data
        self.sample_batch_request = {
            "predictions": [
                {
                    "box_type": "box_type_A",
                    "date": "2023-06-15"
                },
                {
                    "box_type": "box_type_B",
                    "date": "2023-06-16"
                }
            ]
        }
    
    @patch('app.app.model')
    def test_health_endpoint(self, mock_model):
        """Test the health check endpoint"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Make request to health endpoint
        response = client.get("/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        health_data = response.json()
        self.assertEqual(health_data["status"], "healthy")
        if "model_loaded" in health_data:
            self.assertIsInstance(health_data["model_loaded"], bool)
    
    @patch('app.app.model')
    def test_predict_endpoint(self, mock_model):
        """Test the prediction endpoint"""
        # Configure mock to return our mock model
        mock_model.predict = self.mock_model.predict
        
        # Make request to predict endpoint
        with patch('app.app.pd.DataFrame.shape', new_callable=MagicMock) as mock_shape:
            # Mock the shape property to return (1, 24) - one row with 24 features
            mock_shape.__get__ = MagicMock(return_value=(1, 24))
            
            response = client.post("/predict", json=self.sample_request)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["box_type"], "box_type_A")
            self.assertIn("predicted_orders", data)
            self.assertEqual(data["prediction_date"], "2023-06-15")
            self.assertIn("confidence_interval", data)
    
    @patch('app.app.model')
    def test_batch_predict_endpoint(self, mock_model):
        """Test the batch prediction endpoint"""
        # Configure mock to return our mock model
        mock_model.predict = self.mock_model.predict
        
        # Make request to batch-predict endpoint
        with patch('app.app.pd.DataFrame.shape', new_callable=MagicMock) as mock_shape:
            # Mock the shape property to return (1, 24) - one row with 24 features
            mock_shape.__get__ = MagicMock(return_value=(1, 24))
            
            response = client.post("/batch-predict", json=self.sample_batch_request)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predictions", data)
            self.assertEqual(len(data["predictions"]), 2)
            
            # Check first prediction
            first_pred = data["predictions"][0]
            self.assertEqual(first_pred["box_type"], "box_type_A")
            self.assertIn("predicted_orders", first_pred)
            self.assertEqual(first_pred["prediction_date"], "2023-06-15")
            
            # Check second prediction
            second_pred = data["predictions"][1]
            self.assertEqual(second_pred["box_type"], "box_type_B")
            self.assertIn("predicted_orders", second_pred)
            self.assertEqual(second_pred["prediction_date"], "2023-06-16")
    
    @patch('app.app.model')
    def test_model_info_endpoint(self, mock_model):
        """Test the model info endpoint"""
        # Configure mock
        mock_model.__class__.__name__ = "GradientBoostingRegressor"
        mock_model.n_features_in_ = 24
        
        # Make request to model-info endpoint
        response = client.get("/model-info")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_type", data)
        self.assertIn("n_features", data)
    
    

class TestFeatureService(unittest.TestCase):
    """Test cases for feature engineering service"""
    
    def test_add_time_features(self):
        """Test time features generation"""
        # Use the function directly from app.py
        features = add_time_features("2023-06-15")
        
        # Check if all expected features are present
        self.assertIn("month", features)
        self.assertIn("day_of_year", features)
        self.assertIn("week_of_year", features)
        self.assertIn("quarter", features)
        
        # Check if values are correct
        self.assertEqual(features["month"], 6)  # June
        self.assertEqual(features["quarter"], 2)  # Q2
    
    def test_prepare_features(self):
        """Test feature preparation"""
        # Create a simple class to mimic PredictionRequest
        class PredictionRequest:
            def __init__(self, box_type, date, additional_features=None):
                self.box_type = box_type
                self.date = date
                self.additional_features = additional_features
        
        # Create a sample request
        request = PredictionRequest(
            box_type="box_type_A",
            date="2023-06-15",
            additional_features={"subscriber_ratio": 1.5}
        )
        
        # Prepare features
        features_df = prepare_features(request)
        
        # Check if DataFrame is returned
        self.assertIsInstance(features_df, pd.DataFrame)
        
        # Check if DataFrame has the correct number of rows
        self.assertEqual(len(features_df), 1)

if __name__ == '__main__':
    unittest.main() 