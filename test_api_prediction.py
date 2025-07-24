#!/usr/bin/env python3
"""
Test script to simulate FastAPI prediction endpoint
"""

import sys
import os
sys.path.append('backend')

import asyncio
from backend.services.f1_data_service import F1DataService
from backend.models.predictor import RacePredictor

async def test_prediction_endpoint():
    """Simulate the prediction endpoint"""
    print("Testing prediction endpoint simulation...")
    
    try:
        # Step 1: Get race features
        print("Step 1: Getting race features...")
        f1_service = F1DataService()
        features_df = await f1_service.get_race_features(2024, 1)
        
        if features_df.empty:
            print("No race data found")
            return
        
        print(f"Features shape: {features_df.shape}")
        print(f"Features columns: {features_df.columns.tolist()}")
        
        # Step 2: Initialize predictor
        print("\nStep 2: Initializing predictor...")
        predictor = RacePredictor()
        print(f"Model loaded: {predictor.model is not None}")
        print(f"Feature names: {predictor.feature_names}")
        
        # Step 3: Make predictions
        print("\nStep 3: Making predictions...")
        predictions = predictor.predict(features_df)
        
        if not predictions:
            print("Failed to generate predictions")
            return
        
        print(f"Predictions generated: {len(predictions)}")
        
        # Step 4: Get top prediction
        print("\nStep 4: Getting top prediction...")
        top_prediction = predictions[0]
        print(f"Top prediction: {top_prediction}")
        
        # Step 5: Create podium predictions
        print("\nStep 5: Creating podium predictions...")
        podium_predictions = []
        for i, pred in enumerate(predictions[:3]):
            podium_predictions.append({
                "position": i + 1,
                "driver": pred['name'],
                "probability": pred['win_probability']
            })
        
        print(f"Podium predictions: {podium_predictions}")
        
        # Step 6: Create response
        print("\nStep 6: Creating response...")
        prediction_response = {
            "season": 2024,
            "round": 1,
            "predicted_winner": top_prediction['name'],
            "confidence": top_prediction['win_probability'],
            "podium_predictions": podium_predictions,
            "key_features": {
                "qualifying_position": top_prediction['grid_position'],
                "qualifying_time": top_prediction['qualifying_time'],
                "qualifying_performance": top_prediction['qualifying_performance'],
                "team": top_prediction['team'],
                "model_loaded": predictor.get_model_info()['model_loaded']
            }
        }
        
        print(f"Response created successfully")
        print(f"Model loaded in response: {prediction_response['key_features']['model_loaded']}")
        
        return prediction_response
        
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("=" * 60)
    print("FASTAPI PREDICTION ENDPOINT SIMULATION")
    print("=" * 60)
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(test_prediction_endpoint())
        if result:
            print("\n" + "=" * 60)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SIMULATION FAILED")
            print("=" * 60)
    finally:
        loop.close()

if __name__ == "__main__":
    main() 