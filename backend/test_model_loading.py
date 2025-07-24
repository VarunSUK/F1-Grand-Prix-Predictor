#!/usr/bin/env python3
"""
Test model loading from backend directory
"""

import sys
import os
from backend.models.predictor import RacePredictor

def test_model_loading():
    """Test model loading from backend directory"""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model file exists: {os.path.exists('model/lgbm_model.pkl')}")
    
    try:
        predictor = RacePredictor()
        print(f"Predictor initialized")
        print(f"Model loaded: {predictor.model is not None}")
        print(f"Feature names: {predictor.feature_names}")
        
        model_info = predictor.get_model_info()
        print(f"Model info: {model_info}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_model_loading() 