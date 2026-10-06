import pytest
from fastapi.testclient import TestClient
from app import app, clean_and_tokenize

client = TestClient(app)

def test_preprocessing():
    tokens = clean_and_tokenize("The cinematography was brilliant and thrilling!")
    stems = [s for _, s in tokens]
    assert "brilliant" in stems or "thrill" in stems

def test_api_prediction_endpoint():
    response = client.post("/api/v1/predict", json={
        "text": "The performance was absolutely spectacular and moving.",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 200
    data = response.json()
    assert "ai_score" in data
    assert "confidence" in data
    assert data["sentiment"] == "Positive"
    assert data["latency_ms"] < 250.0  

def test_api_analytics_endpoint():
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    # Correctly verifying that the database is returning telemetry
    assert "total_analyzed" in response.json()