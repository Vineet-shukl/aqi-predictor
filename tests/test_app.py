import pytest

from aqi.app import create_app
from aqi.calculator import calculate
from aqi.dataset import co_mg_to_ppm, load_hourly, no2_ug_to_ppb


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_and_page(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert b"AQI Predictor" in page.data
    assert b"Calculate AQI" in page.data
    assert b"Predict next hour" in page.data


def test_calculate_endpoint(client):
    response = client.post("/api/calculate", json={"pm25": 12.0, "no2": 100})
    body = response.get_json()
    assert response.status_code == 200
    assert body["aqi"] == 100
    assert body["category"] == "Moderate"
    assert body["dominant"] == ["NO2"]


def test_calculate_rejects_empty_payload(client):
    response = client.post("/api/calculate", json={})
    assert response.status_code == 400
    assert "at least one" in response.get_json()["error"]


def test_estimate_endpoint_returns_a_category(client):
    response = client.post(
        "/api/estimate",
        json={"current_aqi": 57, "temperature_c": 13.6, "relative_humidity": 48.9, "hour": 19, "month": 3},
    )
    body = response.get_json()
    assert response.status_code == 200
    assert 0 <= body["aqi"] <= 500
    assert body["category"]
    assert body["mae"] > 0
    assert body["persistence_mae"] > body["mae"]
    assert body["mode"] == "estimate"


def test_estimate_rejects_impossible_weather(client):
    response = client.post(
        "/api/estimate",
        json={"current_aqi": 57, "temperature_c": 13.6, "relative_humidity": 140, "hour": 19, "month": 3},
    )
    assert response.status_code == 400


def test_saved_hourly_labels_match_the_calculator():
    rows = load_hourly()
    assert len(rows) > 1000
    for row in rows[::500]:
        report = calculate(co=row.co_ppm, no2=row.no2_ppb)
        assert report.aqi == row.aqi


def test_first_uci_hour_conversion():
    # 10 March 2004, 18:00: CO 2.6 mg/m³ and NO2 113 µg/m³.
    assert co_mg_to_ppm(2.6) == pytest.approx(2.6 * 24.45 / 28.01)
    assert no2_ug_to_ppb(113) == pytest.approx(113 * 24.45 / 46.0055)
    first = load_hourly()[0]
    assert first.hour == 18
    assert first.month == 3
    assert first.temperature_c == pytest.approx(13.6)
    assert first.co_ppm == pytest.approx(co_mg_to_ppm(2.6))
    assert first.no2_ppb == pytest.approx(no2_ug_to_ppb(113))
