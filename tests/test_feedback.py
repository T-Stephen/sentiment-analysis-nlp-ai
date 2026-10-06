import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_submit_valid_feedback():
    # Make a prediction first
    pred_res = client.post("/api/v1/predict", json={
        "text": "Splendid performance by the entire cast.",
        "model_choice": "logistic_regression"
    })
    assert pred_res.status_code == 200
    pred_id = pred_res.json()["prediction_id"]

    # Submit positive confirmation
    fb_res = client.post("/api/v1/feedback", json={
        "prediction_id": pred_id,
        "is_correct": True
    })
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] == "Telemetry recorded."

    # Submit negative correction
    fb_res2 = client.post("/api/v1/feedback", json={
        "prediction_id": pred_id,
        "is_correct": False
    })
    assert fb_res2.status_code == 200

def test_feedback_nonexistent_id():
    response = client.post("/api/v1/feedback", json={
        "prediction_id": 99999999,
        "is_correct": True
    })
    assert response.status_code == 404

def test_feedback_invalid_payload():
    response = client.post("/api/v1/feedback", json={
        "prediction_id": -5,
        "is_correct": True
    })
    assert response.status_code == 422
