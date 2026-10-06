import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_xai_attribution_logistic_regression():
    response = client.post("/api/v1/predict", json={
        "text": "Brilliant masterpiece, absolutely moving and superb acting, but slightly slow pacing.",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 200
    data = response.json()
    exp = data["explanation"]
    assert "positive_drivers" in exp
    assert "negative_drivers" in exp
    assert len(exp["positive_drivers"]) > 0
    # Positive driver tokens should include positive words
    pos_tokens = [d["token"] for d in exp["positive_drivers"]]
    assert any(t in pos_tokens for t in ["brilliant", "masterpiec", "superb", "move"])

def test_xai_attribution_linear_svm():
    response = client.post("/api/v1/predict", json={
        "text": "Brilliant and delightful experience with fantastic direction.",
        "model_choice": "linear_svm"
    })
    assert response.status_code == 200
    data = response.json()
    exp = data["explanation"]
    assert len(exp["positive_drivers"]) > 0
    assert all("weight" in d and d["weight"] > 0 for d in exp["positive_drivers"])

def test_xai_attribution_naive_bayes():
    response = client.post("/api/v1/predict", json={
        "text": "Terrible and horrible disaster, awful experience and total waste.",
        "model_choice": "naive_bayes"
    })
    assert response.status_code == 200
    data = response.json()
    exp = data["explanation"]
    assert len(exp["negative_drivers"]) > 0
    assert all("weight" in d and d["weight"] < 0 for d in exp["negative_drivers"])

def test_xai_methodology_declaration():
    response = client.post("/api/v1/predict", json={
        "text": "Outstanding work.",
        "model_choice": "logistic_regression"
    })
    assert response.status_code == 200
    data = response.json()
    assert "methodology" in data["explanation"]
    assert "TF-IDF" in data["explanation"]["methodology"]
