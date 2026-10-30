from fastapi.testclient import TestClient
from src.api import app

def test_health_route():
    response=TestClient(app).get('/health')
    assert response.status_code == 200
    assert 'model_present' in response.json()
