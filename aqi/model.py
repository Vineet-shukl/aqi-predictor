"""Predict the next hour's AQI from the current index and weather.

The model is a histogram gradient-boosting regressor trained on consecutive
hours from the UCI Air Quality roadside station (Italy, March 2004–April
2005). The target is the next hour's EPA-style index of reference CO and
NO2. Inputs are the current index, the hour being predicted, the month, and
the current temperature and humidity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from aqi.calculator import category_for
from aqi.dataset import HOURLY_PATH, HourlyRow, load_hourly

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "aqi_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURE_NAMES = (
    "current_aqi",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temperature_c",
    "relative_humidity",
)


def feature_row(current_aqi: float, hour: int, month: int, temperature_c: float, relative_humidity: float) -> list[float]:
    return [
        current_aqi,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * (month - 1) / 12),
        math.cos(2 * math.pi * (month - 1) / 12),
        temperature_c,
        relative_humidity,
    ]


def consecutive_pairs(rows: list[HourlyRow]) -> list[tuple[HourlyRow, HourlyRow]]:
    pairs: list[tuple[HourlyRow, HourlyRow]] = []
    for previous, nxt in zip(rows, rows[1:]):
        if (nxt.timestamp - previous.timestamp).total_seconds() == 3600:
            pairs.append((previous, nxt))
    return pairs


def _matrix(pairs: list[tuple[HourlyRow, HourlyRow]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array(
        [
            feature_row(previous.aqi, nxt.hour, nxt.month, previous.temperature_c, previous.relative_humidity)
            for previous, nxt in pairs
        ],
        dtype=float,
    )
    target = np.array([nxt.aqi for _, nxt in pairs], dtype=float)
    return features, target


def train(rows: list[HourlyRow] | None = None, holdout_fraction: float = 0.2) -> tuple[HistGradientBoostingRegressor, dict]:
    """Fit on earlier hours and score the later hours."""
    source = rows if rows is not None else load_hourly()
    pairs = consecutive_pairs(source)
    if len(pairs) < 50:
        raise ValueError("Need at least 50 consecutive hours to train.")
    split = int(len(pairs) * (1 - holdout_fraction))
    split = min(max(split, 1), len(pairs) - 1)
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=0,
    )
    train_x, train_y = _matrix(train_pairs)
    test_x, test_y = _matrix(test_pairs)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    errors = np.abs(predictions - test_y)
    persistence = np.abs(test_y - test_x[:, 0])
    current_values = [previous.aqi for previous, _ in pairs]
    temperatures = [previous.temperature_c for previous, _ in pairs]
    humidities = [previous.relative_humidity for previous, _ in pairs]
    metrics = {
        "mae": round(float(errors.mean()), 2),
        "rmse": round(float(np.sqrt(np.mean((predictions - test_y) ** 2))), 2),
        "persistence_mae": round(float(persistence.mean()), 2),
        "n_train": len(train_pairs),
        "n_test": len(test_pairs),
        "holdout": "last 20% of consecutive hours, in time order",
        "site": "UCI Air Quality roadside station, Italy, March 2004–April 2005",
        "features": list(FEATURE_NAMES),
        "current_aqi": {"min": min(current_values), "max": max(current_values)},
        "temperature_c": {"min": min(temperatures), "max": max(temperatures)},
        "relative_humidity": {"min": min(humidities), "max": max(humidities)},
    }
    return model, metrics


def save(model: HistGradientBoostingRegressor, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def load_model() -> HistGradientBoostingRegressor:
    return joblib.load(MODEL_PATH)


def estimate(
    *,
    current_aqi: int,
    temperature_c: float,
    relative_humidity: float,
    hour: int,
    month: int,
    model: HistGradientBoostingRegressor | None = None,
    metrics: dict | None = None,
) -> dict:
    """Predict the AQI for `hour` from the current index and weather."""
    if isinstance(current_aqi, bool) or not isinstance(current_aqi, (int, float)):
        raise ValueError("Current AQI must be a number.")
    if int(current_aqi) != current_aqi:
        raise ValueError("Current AQI must be a whole number.")
    current_aqi = int(current_aqi)
    if not 0 <= current_aqi <= 500:
        raise ValueError("Current AQI must be between 0 and 500.")
    if isinstance(hour, bool) or isinstance(month, bool):
        raise ValueError("Hour and month must be whole numbers.")
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("Hour must be an integer from 0 through 23.")
    if not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError("Month must be an integer from 1 through 12.")
    if not -50 <= temperature_c <= 60:
        raise ValueError("Temperature must be between -50°C and 60°C.")
    if not 0 <= relative_humidity <= 100:
        raise ValueError("Relative humidity must be between 0 and 100.")

    fitted = model if model is not None else load_model()
    summary = metrics if metrics is not None else load_metrics()
    raw = float(
        fitted.predict(np.array([feature_row(current_aqi, hour, month, temperature_c, relative_humidity)]))[0]
    )
    aqi = int(min(500, max(0, math.floor(raw + 0.5))))
    name, color, text_color, message = category_for(aqi)
    aqi_range = summary["current_aqi"]
    temp_range = summary["temperature_c"]
    humidity_range = summary["relative_humidity"]
    extrapolated = (
        current_aqi < aqi_range["min"]
        or current_aqi > aqi_range["max"]
        or temperature_c < temp_range["min"]
        or temperature_c > temp_range["max"]
        or relative_humidity < humidity_range["min"]
        or relative_humidity > humidity_range["max"]
    )
    return {
        "mode": "estimate",
        "aqi": aqi,
        "category": name,
        "color": color,
        "text_color": text_color,
        "message": message,
        "dominant": [],
        "pollutants": [],
        "notes": [
            (
                f"On later months from the same station, the typical error is {summary['mae']} AQI points, "
                f"compared with {summary['persistence_mae']} if the index simply stayed the same."
            )
        ],
        "mae": summary["mae"],
        "persistence_mae": summary["persistence_mae"],
        "site": summary["site"],
        "extrapolated": extrapolated,
    }


def main() -> None:
    if not HOURLY_PATH.exists():
        raise SystemExit(f"Missing training data at {HOURLY_PATH}")
    model, metrics = train()
    save(model, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
