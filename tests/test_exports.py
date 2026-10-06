import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_export_csv():
    response = client.get("/api/v1/export/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert "sentinel_telemetry.csv" in response.headers.get("content-disposition", "")
    lines = response.text.strip().splitlines()
    assert len(lines) >= 1  # Header row at minimum
    assert "request_id" in lines[0]
    assert "sentiment" in lines[0]

def test_export_json():
    response = client.get("/api/v1/export/json")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "request_id" in data[0]

def test_export_invalid_format():
    response = client.get("/api/v1/export/xml")
    assert response.status_code == 400
    assert "Invalid format" in response.json()["detail"]
