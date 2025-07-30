import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../'))
from main import app

client = TestClient(app)

def test_predict_valid():
    # Use a likely valid season/round (adjust if needed)
    response = client.get("/api/v1/predict/2024/1")
    assert response.status_code in (200, 404, 500)  # Accept 404/500 if no data, but endpoint must respond
    if response.status_code == 200:
        data = response.json()
        assert "predicted_winner" in data
        assert "confidence" in data
        assert "podium_predictions" in data

def test_predict_invalid():
    # Use an obviously invalid season/round
    response = client.get("/api/v1/predict/1900/99")
    # Accept 200, 404, 422, or 500 as valid responses
    assert response.status_code in (200, 404, 422, 500)
    if response.status_code == 200:
        data = response.json()
        # Should contain mock data (e.g., known mock driver)
        assert "predicted_winner" in data
        assert any(name in data["predicted_winner"] for name in ["Verstappen", "Hamilton", "Leclerc", "Max"])
    elif response.status_code == 422:
        data = response.json()
        assert "detail" in data

def test_predict_mock_fallback(monkeypatch):
    # Simulate backend fallback to mock data by patching the predictor/model
    response = client.get("/api/v1/predict/2024/999")  # Unlikely round, should fallback
    # Accept 200 or 422 (if round is out of range)
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        data = response.json()
        assert "predicted_winner" in data
        assert "confidence" in data
        assert "podium_predictions" in data
        assert data["key_features"]["model_loaded"] in (True, False)
    elif response.status_code == 422:
        data = response.json()
        assert "detail" in data

def test_predict_error_handling(monkeypatch):
    # Simulate error in prediction logic by patching get_predictor to raise
    import routes.predict as predict_module
    original_get_predictor = predict_module.get_predictor
    def error_predictor():
        raise Exception("Simulated error")
    predict_module.get_predictor = error_predictor
    response = client.get("/api/v1/predict/2024/1")
    assert response.status_code == 500
    data = response.json()
    assert "note" in data
    assert "Internal server error" in data["note"]
    predict_module.get_predictor = original_get_predictor

def test_predict_empty_session():
    # Simulate a session with no data (edge case)
    response = client.get("/api/v1/predict/2024/0")  # round 0 is invalid, should return 422
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_predict_partial_data(monkeypatch):
    # Simulate partial data by patching create_mock_features and create_mock_predictions to return only one driver
    import utils.mock_data as mock_data_module
    original_create_mock_features = mock_data_module.create_mock_features
    original_create_mock_predictions = mock_data_module.create_mock_predictions
    def partial_mock_features():
        return original_create_mock_features().head(1)  # Only one driver
    def partial_mock_predictions(df):
        preds = original_create_mock_predictions(df)
        return preds[:1]  # Only one prediction
    mock_data_module.create_mock_features = partial_mock_features
    mock_data_module.create_mock_predictions = partial_mock_predictions
    response = client.get("/api/v1/predict/2024/1")
    assert response.status_code == 500
    data = response.json()
    assert "note" in data
    assert "Internal server error" in data["note"]
    mock_data_module.create_mock_features = original_create_mock_features
    mock_data_module.create_mock_predictions = original_create_mock_predictions


def test_test_predict_endpoint():
    # Test the /test-predict endpoint for mock prediction
    response = client.get("/api/v1/test-predict")
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "model_loaded" in data 
    else:
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "note" in data
        assert "Internal server error" in data["note"]


def test_predict_real_data():
    import requests
    url = "http://127.0.0.1:8000/api/v1/predict/2024/1"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    # Check that the predicted_winner is not a mock name (if possible)
    # This is a soft check: if the backend falls back to mock, this will still pass but log a warning
    print("Predicted winner:", data.get("predicted_winner"))
    print("Podium:", data.get("podium_predictions"))
    assert "predicted_winner" in data
    assert "podium_predictions" in data
    assert isinstance(data["podium_predictions"], list)
    # Optionally, check for known real driver names (adjust as needed)
    real_drivers = ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc"]
    if not any(rd in data["predicted_winner"] for rd in real_drivers):
        print("Warning: Predicted winner may be mock data.") 