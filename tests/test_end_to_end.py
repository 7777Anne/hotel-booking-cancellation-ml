"""Train -> serialize -> API -> inference with synthetic test fixtures."""
import json
import joblib
from fastapi.testclient import TestClient
import src.api as api
from src.train import train
from tests.test_train import sample_data

def test_synthetic_training_to_serving(tmp_path,monkeypatch):
    frame=sample_data(120)
    model,report=train(frame)
    artifact=tmp_path/"baseline.joblib"
    joblib.dump(model,artifact)
    (tmp_path/"metrics.json").write_text(json.dumps(report))
    monkeypatch.setattr(api,"MODEL_PATH",artifact)
    booking=frame.iloc[0][list(api.FEATURES)].to_dict()
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code==200
        response=client.post("/predict",json=booking)
        assert response.status_code==200,response.text
        payload=response.json()
        assert 0<=payload["cancellation_probability"]<=1
        assert payload["threshold"]==report["threshold_selected_on_validation"]
