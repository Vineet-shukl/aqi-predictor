from aqi.dataset import load_hourly
from aqi.model import consecutive_pairs, estimate, load_metrics, train


def test_saved_model_metrics_are_finite():
    metrics = load_metrics()
    assert metrics["n_train"] > metrics["n_test"] > 0
    assert 0 < metrics["mae"] < 80
    assert metrics["rmse"] >= metrics["mae"]


def test_morning_rush_hour_rises():
    report = estimate(current_aqi=41, temperature_c=10.2, relative_humidity=59.6, hour=8, month=3)
    assert report["aqi"] == 55
    assert report["category"] == "Moderate"


def test_estimate_stays_on_the_aqi_scale():
    report = estimate(current_aqi=57, temperature_c=20, relative_humidity=50, hour=8, month=6)
    assert 0 <= report["aqi"] <= 500
    assert report["category"] in {
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    }


def test_retraining_on_the_saved_table_is_reproducible():
    rows = load_hourly()
    _, metrics = train(rows)
    assert metrics["mae"] == load_metrics()["mae"]
    assert metrics["n_train"] + metrics["n_test"] == len(consecutive_pairs(rows))
