import unittest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)

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
    
    @patch('app.services.model_service.model')
    def test_health_endpoint(self, mock_model):
        """Test the health check endpoint"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Make request to health endpoint
        response = client.get("/health")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    @patch('app.services.model_service.model')
    def test_predict_endpoint(self, mock_model):
        """Test the prediction endpoint"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Make request to predict endpoint
        response = client.post("/predict", json=self.sample_request)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["box_type"], "box_type_A")
        self.assertEqual(data["predicted_orders"], 42.0)
        self.assertEqual(data["prediction_date"], "2023-06-15")
        self.assertIn("confidence_interval", data)
    
    @patch('app.services.model_service.model')
    def test_batch_predict_endpoint(self, mock_model):
        """Test the batch prediction endpoint"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Make request to batch-predict endpoint
        response = client.post("/batch-predict", json=self.sample_batch_request)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["predictions"][0]["box_type"], "box_type_A")
        self.assertEqual(data["predictions"][1]["box_type"], "box_type_B")
    
    @patch('app.services.model_service.model')
    def test_model_info_endpoint(self, mock_model):
        """Test the model info endpoint"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Make request to model-info endpoint
        response = client.get("/model-info")
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_type", data)
        self.assertIn("n_features", data)
    
    @patch('app.services.model_service.model')
    def test_predict_with_invalid_data(self, mock_model):
        """Test prediction with invalid data"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Invalid request (missing required field)
        invalid_request = {
            "date": "2023-06-15"
            # Missing box_type
        }
        
        # Make request with invalid data
        response = client.post("/predict", json=invalid_request)
        
        # Check response (should be 422 Unprocessable Entity)
        self.assertEqual(response.status_code, 422)
    
    @patch('app.services.model_service.model')
    def test_predict_with_invalid_date(self, mock_model):
        """Test prediction with invalid date format"""
        # Configure mock
        mock_model.return_value = self.mock_model
        
        # Invalid date format
        invalid_request = {
            "box_type": "box_type_A",
            "date": "invalid-date"
        }
        
        # Make request with invalid date
        response = client.post("/predict", json=invalid_request)
        
        # Check response (should be 400 Bad Request or 422 Unprocessable Entity)
        self.assertIn(response.status_code, [400, 422])

class TestFeatureService(unittest.TestCase):
    """Test cases for feature engineering service"""
    
    def test_add_time_features(self):
        """Test time features generation"""
        from app.services.feature_service import add_time_features
        
        # Test with a specific date
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
        from app.services.feature_service import prepare_features
        from app.api.models.schemas import PredictionRequest
        
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