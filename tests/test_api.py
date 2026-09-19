import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(artifact):
    with TestClient(create_app(artifact)) as client:
        yield client


def test_health_and_info(client):
    assert client.get("/health").json()["model_loaded"]
    assert client.get("/model-info").json()["language"] == "en"


def test_full_training_to_api(client):
    response = client.post("/predict", json={"text": "lost card"})
    assert response.status_code == 200
    result = response.json()["predictions"][0]
    assert result["suggested_category"] == "card"
    assert (result["category"] is None) == result["requires_human"]


def test_batch(client):
    response = client.post("/predict", json={"text": ["lost card", "reset pin", "世界💳"]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": " "},
        {"text": 12},
        {"text": [None]},
        {"text": []},
        {"text": "x" * 4001},
        {"text": ["x"] * 65},
        {"text": "card", "unexpected": True},
        {},
    ],
)
def test_invalid_request(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_unloaded_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"text": "hello"}).status_code == 503
