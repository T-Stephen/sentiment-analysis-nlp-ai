import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_validation_empty_text():
    response = client.post("/api/v1/predict", json={
        "text": "",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 422

def test_validation_whitespace_only():
    response = client.post("/api/v1/predict", json={
        "text": "    \n\t   ",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 422

def test_validation_oversized_text():
    oversized = "great " * 1200  # > 5000 characters
    assert len(oversized) > 5000
    response = client.post("/api/v1/predict", json={
        "text": oversized,
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 422

def test_validation_missing_text_field():
    response = client.post("/api/v1/predict", json={
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 422

def test_fallback_for_unknown_model_choice():
    # Application gracefully falls back to logistic_regression
    response = client.post("/api/v1/predict", json={
        "text": "A very solid and engaging film.",
        "model_choice": "non_existent_engine"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("Positive", "Negative", "Neutral")
