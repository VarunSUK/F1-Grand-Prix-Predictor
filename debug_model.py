#!/usr/bin/env python3

import sys
sys.path.append('backend')

import joblib
import pandas as pd
import numpy as np
from backend.models.predictor import RacePredictor
from backend.services.f1_data_service import F1DataService
import asyncio

def test_model_loading():
    print("Testing model loading...")
    
    try:
        model_data = joblib.load('backend/model/lgbm_model.pkl')
        print(f"Model loaded successfully")
        print(f"Model type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"Model keys: {model_data.keys()}")
            print(f"Feature names: {model_data.get('feature_names', [])}")
            print(f"Training info: {model_data.get('training_info', {})}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def test_predictor():
    print("\nTesting RacePredictor...")
    
    try:
        predictor = RacePredictor()
        print(f"Predictor initialized")
        print(f"Model loaded: {predictor.model is not None}")
        print(f"Feature names: {predictor.feature_names}")
        
        return predictor
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return None

def test_feature_extraction():
    print("\nTesting feature extraction...")
    
    try:
        f1_service = F1DataService()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        features_df = loop.run_until_complete(f1_service.get_race_features(2024, 1))
        loop.close()
        
        print(f"Features shape: {features_df.shape}")
        print(f"Features columns: {features_df.columns.tolist()}")
        print(f"Sample data:")
        print(features_df.head(2))
        
        return features_df
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def test_prediction(predictor, features_df):
    print("\nTesting prediction...")
    
    try:
        predictions = predictor.predict(features_df)
        print(f"Predictions generated: {len(predictions)}")
        print(f"Top 3 predictions:")
        for i, pred in enumerate(predictions[:3]):
            print(f"{i+1}. {pred['name']} - {pred['win_probability']:.3f}")
        
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def main():
    print("=" * 60)
    print("F1 MODEL DEBUG")
    print("=" * 60)
    
    if not test_model_loading():
        return
    
    predictor = test_predictor()
    if predictor is None:
        return
    
    features_df = test_feature_extraction()
    if features_df is None:
        return
    
    predictions = test_prediction(predictor, features_df)
    if predictions is None:
        return
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main() 