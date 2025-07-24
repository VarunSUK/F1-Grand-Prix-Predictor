from fastapi import APIRouter, HTTPException, status, Query, Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field
from utils.type_safety import to_python_types, to_podium
import pandas as pd
import os
from dotenv import load_dotenv

router = APIRouter()

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "model/lgbm_model.pkl")

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        from models.predictor import RacePredictor
        model_path = os.path.join(os.path.dirname(__file__), '..', MODEL_PATH)
        model_path = os.path.abspath(model_path)
        _predictor = RacePredictor(model_path)
    return _predictor

class PodiumPrediction(BaseModel):
    position: int = Field(..., description="Podium position (1=winner, 2=second, 3=third)")
    driver: str = Field(..., description="Driver name")
    team: str = Field(..., description="Team name")
    probability: float = Field(..., description="Predicted probability of this position")

class KeyFeatures(BaseModel):
    qualifying_position: int = Field(..., description="Grid position for the race")
    qualifying_time: float = Field(..., description="Qualifying lap time in seconds")
    qualifying_performance: float = Field(..., description="Normalized qualifying performance score")
    team: str = Field(..., description="Team name")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded or using mock predictions")

class PredictionResponse(BaseModel):
    season: int = Field(..., description="F1 season year", example=2024)
    round: int = Field(..., description="Race round number", example=1)
    predicted_winner: str = Field(..., description="Predicted winner's name", example="Max Verstappen")
    confidence: float = Field(..., description="Model confidence in the winner prediction", example=0.72)
    podium_predictions: List[PodiumPrediction] = Field(..., description="Top 3 podium predictions")
    key_features: KeyFeatures = Field(..., description="Key features for the predicted winner")
    note: Optional[str] = Field(None, description="Note about the prediction, if any")

class TestResponse(BaseModel):
    status: str = Field(..., description="Status of the test endpoint", example="success")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    feature_count: int = Field(..., description="Number of features expected by the model")

class TestPredictResponse(BaseModel):
    status: str = Field(..., example="success")
    predictions: List[PodiumPrediction] = Field(..., description="Top 3 mock predictions")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")

class RaceInfo(BaseModel):
    round: int = Field(..., description="Race round number", example=1)
    name: str = Field(..., description="Race name", example="Bahrain Grand Prix")
    date: str = Field(..., description="Race date (YYYY-MM-DD)", example="2024-03-02")

class SeasonRacesResponse(BaseModel):
    season: int = Field(..., description="F1 season year")
    races: List[RaceInfo] = Field(..., description="List of races for the season")

class FeaturesResponse(BaseModel):
    season: int = Field(..., description="F1 season year")
    round: int = Field(..., description="Race round number")
    features: List[Dict[str, Any]] = Field(..., description="List of feature dicts for each driver")
    driver_count: int = Field(..., description="Number of drivers")
    columns: List[str] = Field(..., description="List of feature column names")

class ModelInfo(BaseModel):
    model_info: Dict[str, Any] = Field(..., description="Model metadata and status")
    status: str = Field(..., description="Human-readable model status")

@router.get(
    "/predict/{season}/{round}",
    response_model=PredictionResponse,
    summary="Predict F1 Grand Prix Winner",
    description="Predict the winner and podium for a specific F1 Grand Prix using the ML model.",
    responses={
        200: {
            "description": "Prediction result",
            "content": {
                "application/json": {
                    "example": {
                        "season": 2024,
                        "round": 1,
                        "predicted_winner": "Max Verstappen",
                        "confidence": 0.72,
                        "podium_predictions": [
                            {"position": 1, "driver": "Max Verstappen", "team": "Red Bull", "probability": 0.72},
                            {"position": 2, "driver": "Lewis Hamilton", "team": "Mercedes", "probability": 0.18},
                            {"position": 3, "driver": "Charles Leclerc", "team": "Ferrari", "probability": 0.10}
                        ],
                        "key_features": {
                            "qualifying_position": 1,
                            "qualifying_time": 80.0,
                            "qualifying_performance": 1.0,
                            "team": "Red Bull",
                            "model_loaded": True
                        },
                        "note": "Prediction is based on data from round 3, as no real data is available for round 4."
                    }
                }
            }
        }
    }
)
async def predict_race_winner(
    season: int = Path(..., ge=1950, le=2100, description="F1 season year (1950-2100)"),
    round: int = Path(..., ge=1, le=30, description="Race round number (1-30)")
):
    try:
        from services.f1_data_service import F1DataService
        from models.predictor import RacePredictor

        f1_service = F1DataService()
        features_df, used_round, partial_fields, note = await f1_service.get_race_features(season, round)

        # For 2024/2025, if no real data, return 503 error or clear error JSON
        if season in (2024, 2025) and (features_df is None or features_df.empty):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "season": season,
                    "round": round,
                    "predicted_winner": None,
                    "confidence": 0.0,
                    "podium_predictions": [],
                    "key_features": {},
                    "note": note or "No real data available for this race. Please try again later."
                }
            )

        if note:
            prediction_response = {"note": note}
        else:
            prediction_response = {}
        if partial_fields:
            missing = ', '.join(sorted(partial_fields))
            note = (note + ' ' if note else '') + f"Prediction is based on partial real data; missing fields: {missing}."
            prediction_response["note"] = note

        # If no real data, use mock data
        if features_df is None or features_df.empty or 'driver' not in features_df.columns:
            from utils.mock_data import create_mock_features
            features_df = create_mock_features()
            note = "Prediction is based on mock data, as no real data is available for this season."
            prediction_response["note"] = note

        predictor = get_predictor()
        predictions = predictor.predict(features_df)

        if not predictions:
            from utils.mock_data import create_mock_predictions
            predictions = create_mock_predictions(features_df)
            note = "Prediction is based on mock data, as no real data is available for this season."
            prediction_response["note"] = note

        # Ensure predictions are in the correct schema and count
        n_drivers = len(features_df)
        podium_predictions = to_podium(predictions, n_drivers)
        top_prediction = podium_predictions[0] if podium_predictions else {"driver": "Unknown", "probability": 0.0, "team": "Unknown", "position": 1}

        prediction_response.update({
            "season": season,
            "round": round,
            "predicted_winner": top_prediction["driver"],
            "confidence": top_prediction["probability"],
            "podium_predictions": podium_predictions,
            "key_features": {
                "qualifying_position": predictions[0].get('grid_position', 0) if predictions else 0,
                "qualifying_time": predictions[0].get('qualifying_time', 0.0) if predictions else 0.0,
                "qualifying_performance": predictions[0].get('qualifying_performance', 0.0) if predictions else 0.0,
                "team": predictions[0].get('team', 'Unknown') if predictions else 'Unknown',
                "model_loaded": predictor.get_model_info().get('model_loaded', False) if predictions else False
            }
        })
        return PredictionResponse(**to_python_types(prediction_response))
    except Exception as e:
        import logging
        logging.error(f"Error in predict_race_winner: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "season": season,
                "round": round,
                "predicted_winner": None,
                "confidence": 0.0,
                "podium_predictions": [],
                "key_features": {},
                "note": f"Internal server error: {str(e)}"
            }
        )

@router.get("/test", response_model=TestResponse, summary="Test model status", description="Check if the ML model is loaded and get feature count.")
async def test_endpoint():
    """
    Test endpoint to check if the ML model is loaded and get feature count.
    Returns:
        TestResponse: Status, model_loaded, and feature_count.
    """
    try:
        predictor = get_predictor()
        model_info = predictor.get_model_info()
        return {
            "status": "success",
            "model_loaded": model_info['model_loaded'],
            "feature_count": model_info['feature_count']
        }
    except Exception as e:
        import logging
        logging.error(f"Error in test_endpoint: {e}")
        # Return all required fields with default values
        return {
            "status": "error",
            "model_loaded": False,
            "feature_count": 0
        }

@router.get("/test-predict", response_model=TestPredictResponse, summary="Test prediction with mock data", description="Test the prediction logic using mock data.")
async def test_prediction():
    try:
        from utils.mock_data import create_mock_features, create_mock_predictions
        mock_features = create_mock_features()
        predictor = get_predictor()
        predictions = create_mock_predictions(mock_features)
        # Ensure predictions are in the correct schema
        n_drivers = len(mock_features)
        status = "success" if predictions else "error"
        return to_python_types({
            "status": status,
            "predictions": to_podium(predictions, n_drivers),
            "model_loaded": predictor.get_model_info()['model_loaded']
        })
    except Exception as e:
        import logging
        logging.error(f"Error in test_prediction: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "predictions": [],
                "model_loaded": False,
                "note": f"Internal server error: {str(e)}"
            }
        )

@router.post(
    "/model/reload",
    summary="Reload the ML model from disk",
    description="Hot-reload the ML model without restarting the server. Useful after updating the model file.",
    status_code=status.HTTP_200_OK,
)
async def reload_model():
    """
    Reload the ML model from disk without restarting the server.
    Returns:
        dict: Status and model info after reload.
    """
    predictor = get_predictor()
    try:
        predictor.reload_model()
        model_info = predictor.get_model_info()
        return {
            "status": "Model reloaded successfully" if model_info.get("model_loaded") else "Model reload failed",
            "model_info": model_info
        }
    except Exception as e:
        import logging
        logging.error(f"Error in reload_model: {e}")
        # Return all required fields with default values
        return {
            "status": "Model reload failed",
            "model_info": {"model_loaded": False, "feature_count": 0, "feature_names": []}
        }

@router.get("/seasons/{season}/races", response_model=SeasonRacesResponse, summary="Get season race schedule", description="Get all races for a specific F1 season.")
async def get_season_races(
    season: int = Path(..., ge=1950, le=2100, description="F1 season year (1950-2100)")
):
    """
    Get all races for a specific F1 season.
    - **season**: F1 season year
    Returns:
        SeasonRacesResponse: List of races for the season.
    """
    # Full mock race lists for 2024 and 2025
    races_2024 = [
        {"round": 1, "name": "Bahrain Grand Prix", "date": "2024-03-02"},
        {"round": 2, "name": "Saudi Arabian Grand Prix", "date": "2024-03-09"},
        {"round": 3, "name": "Australian Grand Prix", "date": "2024-03-24"},
        {"round": 4, "name": "Japanese Grand Prix", "date": "2024-04-07"},
        {"round": 5, "name": "Chinese Grand Prix", "date": "2024-04-21"},
        {"round": 6, "name": "Miami Grand Prix", "date": "2024-05-05"},
        {"round": 7, "name": "Emilia Romagna Grand Prix", "date": "2024-05-19"},
        {"round": 8, "name": "Monaco Grand Prix", "date": "2024-05-26"},
        {"round": 9, "name": "Canadian Grand Prix", "date": "2024-06-09"},
        {"round": 10, "name": "Spanish Grand Prix", "date": "2024-06-23"},
        {"round": 11, "name": "Austrian Grand Prix", "date": "2024-06-30"},
        {"round": 12, "name": "British Grand Prix", "date": "2024-07-07"},
        {"round": 13, "name": "Hungarian Grand Prix", "date": "2024-07-21"},
        {"round": 14, "name": "Belgian Grand Prix", "date": "2024-07-28"},
        {"round": 15, "name": "Dutch Grand Prix", "date": "2024-08-25"},
        {"round": 16, "name": "Italian Grand Prix", "date": "2024-09-01"},
        {"round": 17, "name": "Azerbaijan Grand Prix", "date": "2024-09-15"},
        {"round": 18, "name": "Singapore Grand Prix", "date": "2024-09-22"},
        {"round": 19, "name": "United States Grand Prix", "date": "2024-10-20"},
        {"round": 20, "name": "Mexico City Grand Prix", "date": "2024-10-27"},
        {"round": 21, "name": "São Paulo Grand Prix", "date": "2024-11-03"},
        {"round": 22, "name": "Las Vegas Grand Prix", "date": "2024-11-23"},
        {"round": 23, "name": "Qatar Grand Prix", "date": "2024-12-01"},
        {"round": 24, "name": "Abu Dhabi Grand Prix", "date": "2024-12-08"}
    ]
    races_2025 = [
        {"round": 1, "name": "Australian Grand Prix", "date": "2025-03-16"},
        {"round": 2, "name": "Saudi Arabian Grand Prix", "date": "2025-03-23"},
        {"round": 3, "name": "Bahrain Grand Prix", "date": "2025-04-06"},
        {"round": 4, "name": "Japanese Grand Prix", "date": "2025-04-13"},
        {"round": 5, "name": "Chinese Grand Prix", "date": "2025-04-27"},
        {"round": 6, "name": "Miami Grand Prix", "date": "2025-05-11"},
        {"round": 7, "name": "Emilia Romagna Grand Prix", "date": "2025-05-25"},
        {"round": 8, "name": "Monaco Grand Prix", "date": "2025-06-01"},
        {"round": 9, "name": "Canadian Grand Prix", "date": "2025-06-15"},
        {"round": 10, "name": "Spanish Grand Prix", "date": "2025-06-29"},
        {"round": 11, "name": "Austrian Grand Prix", "date": "2025-07-06"},
        {"round": 12, "name": "British Grand Prix", "date": "2025-07-13"},
        {"round": 13, "name": "Hungarian Grand Prix", "date": "2025-07-27"},
        {"round": 14, "name": "Belgian Grand Prix", "date": "2025-08-03"},
        {"round": 15, "name": "Dutch Grand Prix", "date": "2025-08-24"},
        {"round": 16, "name": "Italian Grand Prix", "date": "2025-08-31"},
        {"round": 17, "name": "Azerbaijan Grand Prix", "date": "2025-09-14"},
        {"round": 18, "name": "Singapore Grand Prix", "date": "2025-09-21"},
        {"round": 19, "name": "United States Grand Prix", "date": "2025-10-19"},
        {"round": 20, "name": "Mexico City Grand Prix", "date": "2025-10-26"},
        {"round": 21, "name": "São Paulo Grand Prix", "date": "2025-11-09"},
        {"round": 22, "name": "Las Vegas Grand Prix", "date": "2025-11-22"},
        {"round": 23, "name": "Qatar Grand Prix", "date": "2025-11-30"},
        {"round": 24, "name": "Abu Dhabi Grand Prix", "date": "2025-12-07"}
    ]
    if season == 2024:
        return {"season": season, "races": races_2024}
    elif season == 2025:
        return {"season": season, "races": races_2025}
    else:
        # Dynamically fetch the full race schedule for all other years
        from services.f1_data_service import F1DataService
        f1_service = F1DataService()
        races = await f1_service.get_race_schedule(season)
        # Map to API schema
        race_list = [
            {"round": r["round"], "name": r["name"], "date": r["date"]}
            for r in races
        ]
        return {"season": season, "races": race_list}

@router.get("/features/{season}/{round}", response_model=FeaturesResponse, summary="Get engineered features for a race", description="Get all engineered features for all drivers in a specific race.")
async def get_race_features(
    season: int = Path(..., ge=1950, le=2100, description="F1 season year (1950-2100)"),
    round: int = Path(..., ge=1, le=30, description="Race round number (1-30)")
):
    """
    Get engineered features for all drivers in a specific race.
    - **season**: F1 season year
    - **round**: Race round number
    Returns:
        FeaturesResponse: List of features for each driver.
    """
    try:
        from services.f1_data_service import F1DataService
        
        f1_service = F1DataService()
        features_df = await f1_service.get_race_features(season, round)
        
        # Convert DataFrame to JSON-serializable format
        features_list = features_df.to_dict('records')
        
        return to_python_types({
            "season": season,
            "round": round,
            "features": features_list,
            "driver_count": len(features_list),
            "columns": features_df.columns.tolist()
        })
        
    except Exception as e:
        import logging
        logging.error(f"Error in get_race_features: {e}")
        # Return all required fields with default values
        return to_python_types({
            "season": season,
            "round": round,
            "features": [],
            "driver_count": 0,
            "columns": []
        })

@router.get("/model/info", response_model=ModelInfo, summary="Get ML model info", description="Get information about the loaded ML model.")
async def get_model_info():
    """
    Get information about the loaded ML model.
    Returns:
        ModelInfo: Model metadata and status.
    """
    try:
        predictor = get_predictor()
        model_info = predictor.get_model_info()
        
        return {
            "model_info": model_info,
            "status": "Model loaded successfully" if model_info['model_loaded'] else "Using mock predictions"
        }
        
    except Exception as e:
        import logging
        logging.error(f"Error in get_model_info: {e}")
        # Return all required fields with default values
        return {
            "model_info": {"model_loaded": False, "feature_count": 0, "feature_names": []},
            "status": "Failed to get model info"
        } 