import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_analytics_metrics_completeness():
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_analyzed" in data
    assert "positive" in data
    assert "negative" in data
    assert "neutral" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "p99_latency_ms" in data
    assert "model_distribution" in data
    assert data["positive"] + data["negative"] + data["neutral"] <= data["total_analyzed"]

def test_history_endpoint():
    response = client.get("/api/v1/history?limit=5")
    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert len(records) <= 5
    if len(records) > 0:
        first = records[0]
        assert "id" in first
        assert "request_id" in first
        assert "sentiment" in first
        assert "engine" in first
        assert "latency_ms" in first
