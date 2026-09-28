import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    add_features,
    clean,
    features_for_forecast,
    prepare,
    supervised_frame,
    time_split,
)


def _raw_city(values: list[float], start: str = "2020-01-01", city: str = "Delhi") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values))
    return pd.DataFrame(
        {
            "City": city,
            "Date": dates.strftime("%Y-%m-%d"),
            "PM2.5": values,
            "PM10": [value * 1.5 for value in values],
            "NO": values,
            "NO2": values,
            "CO": [0.5] * len(values),
            "SO2": [5.0] * len(values),
            "O3": [20.0] * len(values),
        }
    )


def test_lag_uses_previous_calendar_day_not_previous_observation():
    raw = _raw_city([10.0, 0.0, 100.0, 50.0], start="2020-01-01")
    raw = raw[raw["Date"] != "2020-01-02"]
    featured = add_features(clean(raw))
    by_date = featured.set_index("date")
    assert np.isnan(by_date.loc[pd.Timestamp("2020-01-03"), "pm25_lag_1"])
    assert by_date.loc[pd.Timestamp("2020-01-04"), "pm25_lag_1"] == pytest.approx(100.0)


def test_same_day_pm25_does_not_enter_features():
    base = add_features(clean(_raw_city([10, 20, 30, 40, 50, 60, 70, 80])))
    spiked = add_features(clean(_raw_city([10, 20, 30, 40, 50, 60, 70, 999])))
    target = pd.Timestamp("2020-01-08")
    left = base.set_index("date").loc[target]
    right = spiked.set_index("date").loc[target]
    for column in FEATURE_COLUMNS:
        if column == "city":
            assert left[column] == right[column]
            continue
        if pd.isna(left[column]) and pd.isna(right[column]):
            continue
        assert left[column] == pytest.approx(float(right[column]))
    assert right["pm25"] == 999
    assert right["pm25_lag_1"] == pytest.approx(70)


def test_rolling_mean_excludes_target_day():
    featured = add_features(clean(_raw_city([1, 2, 3, 4, 5, 6, 7, 8])))
    row = featured.set_index("date").loc[pd.Timestamp("2020-01-08")]
    assert row["pm25_roll_mean_7"] == pytest.approx(np.mean([1, 2, 3, 4, 5, 6, 7]))
    assert row["pm25_lag_1"] == pytest.approx(7)


def test_time_split_has_no_date_overlap_on_real_data():
    supervised = supervised_frame(prepare())
    train, test, cutoff, _excluded = time_split(supervised)
    assert train["date"].max() < cutoff
    assert test["date"].min() == cutoff
    assert train["date"].max() < test["date"].min()
    assert set(test["city"]).issubset(set(train["city"]))
    assert "pm25" not in FEATURE_COLUMNS
    assert "aqi" not in FEATURE_COLUMNS
    assert not train.empty
    assert not test.empty


def test_inference_features_match_training_row():
    featured = add_features(clean(_raw_city(list(range(1, 21)), city="Mumbai")))
    forecast_date = pd.Timestamp("2020-01-20")
    history = featured.loc[
        featured["date"] < forecast_date, ["date", "pm25", "pm10", "no", "no2", "co", "so2", "o3"]
    ]
    built = features_for_forecast(history, "Mumbai", forecast_date)
    original = featured.set_index("date").loc[forecast_date]
    assert built["city"] == "Mumbai"
    for column in FEATURE_COLUMNS:
        if column == "city":
            continue
        assert built[column] == pytest.approx(original[column])


def test_forecast_rejects_history_on_the_target_date():
    history = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "pm25": [10.0],
            "pm10": [12.0],
            "no": [1.0],
            "no2": [1.0],
            "co": [0.4],
            "so2": [2.0],
            "o3": [10.0],
        }
    )
    with pytest.raises(ValueError, match="before the forecast date"):
        features_for_forecast(history, "Delhi", pd.Timestamp("2020-01-02"))
