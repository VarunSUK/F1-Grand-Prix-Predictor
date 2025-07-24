import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.predictor import RacePredictor, F1Predictor
from utils.mock_data import create_mock_features

def test_race_predictor_mock_predictions():
    predictor = RacePredictor(model_path=None)  # No model, should use mock
    features = create_mock_features()
    preds = predictor.predict(features)
    assert isinstance(preds, list)
    assert len(preds) > 0
    assert 'name' in preds[0]
    assert 'win_probability' in preds[0]
    # Probabilities should sum to ~1
    total_prob = sum(p['win_probability'] for p in preds)
    assert abs(total_prob - 1.0) < 0.01

def test_race_predictor_model_info():
    predictor = RacePredictor(model_path=None)
    info = predictor.get_model_info()
    assert isinstance(info, dict)
    assert 'model_loaded' in info
    assert 'feature_count' in info

def test_f1_predictor_prepare_features():
    predictor = F1Predictor()
    # Minimal qualifying data
    qualifying_data = pd.DataFrame({
        'Driver': ['A', 'B'],
        'Position': [1, 2],
        'Q3': [80.0, 81.0]
    })
    driver_data = {'A': {'avg_position': 2, 'total_points': 10, 'races_analyzed': 3},
                  'B': {'avg_position': 3, 'total_points': 8, 'races_analyzed': 3}}
    team_data = {'avg_team_position': 2.5, 'reliability_score': 0.9}
    track_data = {'track_type': 'street', 'avg_pit_stops': 2, 'overtaking_difficulty': 'hard'}
    features = predictor.prepare_features(driver_data, qualifying_data, team_data, track_data)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'driver' in features.columns

def test_f1_predictor_predict_winner_error():
    predictor = F1Predictor()
    # No model loaded, should raise and return empty dict
    features = create_mock_features().head(2)
    result = predictor.predict_winner(features)
    assert isinstance(result, dict)
    assert result == {} or 'predicted_winner' in result 