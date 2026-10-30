from fastapi.testclient import TestClient
from src.api import app


def test_health_reports_missing_model_when_no_artifact():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ("ready", "model_missing")


def test_invalid_request_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={"lead_time": -1})
        assert response.status_code == 422
