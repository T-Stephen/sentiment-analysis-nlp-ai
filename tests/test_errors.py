import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoints():
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "14.3.0-PROD"
        assert data["database_connected"] is True
        assert data["models_loaded"]["logistic_regression"] is True
        assert data["models_loaded"]["linear_svm"] is True
        assert data["models_loaded"]["naive_bayes"] is True
        assert data["vectorizer_loaded"] is True
        assert response.headers.get("x-request-id") is not None

def test_not_found_route():
    response = client.get("/api/v1/unknown_resource")
    assert response.status_code == 404
    assert response.headers.get("x-content-type-options") == "nosniff"

def test_request_id_propagation():
    custom_id = "SN-TESTING-99"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id
