import pandas as pd
from fastapi.testclient import TestClient
from src.api import app
from src.data import build_audit_report,save_audit_report
from tests.test_train import sample_data

def test_json_audit_file(tmp_path):
    report=build_audit_report(sample_data(60))
    dest=save_audit_report(report,tmp_path/"audit"/"result.json")
    assert dest.exists()
    assert '"rows": 60' in dest.read_text()

def test_readiness_signals_model_absence_or_presence():
    with TestClient(app) as client:
        state=client.get("/health").json()["status"]
        readiness=client.get("/ready")
        assert readiness.status_code==(200 if state=="ready" else 503)
