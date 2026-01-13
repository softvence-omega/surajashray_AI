from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_all_endpoints():
    response_root = client.get("/")
    assert response_root.status_code == 200
