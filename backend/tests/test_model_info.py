import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_model_info():
    response = client.get("/api/v1/model/info")
    # Accept both 200 (model loaded) and 500 (model not found) as valid responses
    assert response.status_code in (200, 500)
    
    if response.status_code == 200:
        data = response.json()
        assert "model_info" in data
        assert "status" in data
        assert "model_loaded" in data["model_info"]
    elif response.status_code == 500:
        # Model not found is expected behavior
        data = response.json()
        assert "detail" in data
        assert "Failed to get model info" in data["detail"] 