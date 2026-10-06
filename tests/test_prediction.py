import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_prediction_logistic_regression():
    response = client.post("/api/v1/predict", json={
        "text": "An absolute cinematic triumph with mesmerizing visuals and profound storytelling.",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert -1.0 <= data["ai_score"] <= 1.0
    assert 50.0 <= data["confidence"] <= 100.0
    assert data["request_id"].startswith("SN-")
    assert "explanation" in data
    assert "latency_ms" in data
    assert response.headers.get("x-content-type-options") == "nosniff"

def test_prediction_linear_svm():
    response = client.post("/api/v1/predict", json={
        "text": "A dreadful, disorganized and agonizing waste of valuable time.",
        "model_choice": "linear_svm"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Negative"
    assert -1.0 <= data["ai_score"] <= 1.0
    assert 50.0 <= data["confidence"] <= 100.0

def test_prediction_naive_bayes():
    response = client.post("/api/v1/predict", json={
        "text": "The film had standard dialogue and ordinary acting throughout.",
        "model_choice": "naive_bayes"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("Positive", "Negative", "Neutral")
    assert "tokens_count" in data

def test_model_consensus_endpoint():
    response = client.post("/api/v1/consensus", json={
        "text": "This film was an astonishing, unforgettable masterpiece!"
    })
    assert response.status_code == 200
    data = response.json()
    assert "agreement_ratio" in data
    assert "consensus_sentiment" in data
    assert "models" in data
    assert "logistic_regression" in data["models"]
    assert "linear_svm" in data["models"]
    assert "naive_bayes" in data["models"]
    assert data["consensus_sentiment"] == "Positive"
    assert isinstance(data["disagreement_detected"], bool)
