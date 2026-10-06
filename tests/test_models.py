import pytest
from fastapi.testclient import TestClient
from app import app, loaded_models, vectorizer

client = TestClient(app)

def test_model_registry_loaded():
    assert "logistic_regression" in loaded_models
    assert "linear_svm" in loaded_models
    assert "naive_bayes" in loaded_models
    assert vectorizer is not None
    assert len(vectorizer.vocabulary_) == 5000

def test_api_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    
    # Check that each model exposes real benchmark metrics
    for model_key in ["logistic_regression", "linear_svm", "naive_bayes"]:
        assert model_key in models
        m_info = models[model_key]
        if "metrics" in m_info:
            metrics = m_info["metrics"]
            assert "accuracy" in metrics
            assert "f1_score" in metrics
            assert metrics["accuracy"] > 80.0
            assert metrics["f1_score"] > 0.80
        if "confusion_matrix" in m_info:
            cm = m_info["confusion_matrix"]
            assert "true_positive" in cm
            assert "true_negative" in cm
            assert cm["total_evaluated"] == 10000
