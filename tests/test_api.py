import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _example() -> dict:
    return json.loads((ROOT / "reports" / "api_example.json").read_text(encoding="utf-8"))


def _metadata() -> dict:
    return json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    metadata = _metadata()
    assert body["status"] == "ok"
    assert body["model_name"] == metadata["best_model"]
    assert body["n_cities"] == len(metadata["cities"])
    assert body["target"] == "pm25"


def test_predict_documented_example():
    example = _example()
    response = client.post("/predict", json=example["request"])
    assert response.status_code == 200
    assert response.json() == example["response"]


def test_predict_accepts_lowercase_city():
    example = _example()
    payload = json.loads(json.dumps(example["request"]))
    payload["city"] = payload["city"].lower()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["city"] == example["response"]["city"]
    assert response.json()["predicted_pm25"] == example["response"]["predicted_pm25"]


def test_predict_rejects_negative_pm25():
    example = _example()
    payload = json.loads(json.dumps(example["request"]))
    payload["history"][0]["pm25"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_city():
    example = _example()
    payload = json.loads(json.dumps(example["request"]))
    payload["city"] = "Atlantis"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Unknown city" in response.json()["detail"]


def test_predict_requires_yesterday():
    example = _example()
    payload = json.loads(json.dumps(example["request"]))
    payload["history"] = payload["history"][:-1]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "calendar day before" in response.json()["detail"]


def test_openapi_lists_health_and_predict():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/predict" in paths
