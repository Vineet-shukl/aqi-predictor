"""Turn a validated forecast payload into a prediction dictionary."""

from __future__ import annotations

import pandas as pd

from src.aqi import pm25_category, pm25_subindex
from src.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ForecastInputError,
    features_for_forecast,
    resolve_city,
)


def predict_payload(model, known_cities: list[str], model_name: str, payload: dict) -> dict:
    city = resolve_city(str(payload["city"]), known_cities)
    if city is None:
        known = ", ".join(known_cities)
        raise ForecastInputError(f"Unknown city '{payload['city']}'. Known cities: {known}")

    history = pd.DataFrame(payload["history"])
    forecast_date = pd.Timestamp(payload["date"])
    features = features_for_forecast(history, city, forecast_date)
    if pd.isna(features["pm25_lag_1"]):
        raise ForecastInputError(
            "History must include PM2.5 for the calendar day before the forecast date"
        )

    row = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    for column in NUMERIC_FEATURES:
        row[column] = pd.to_numeric(row[column], errors="coerce")

    raw_prediction = float(model.predict(row)[0])
    predicted = max(raw_prediction, 0.0)
    return {
        "city": city,
        "date": forecast_date.date().isoformat(),
        "predicted_pm25": round(predicted, 2),
        "aqi_subindex": round(pm25_subindex(predicted), 1),
        "aqi_category": pm25_category(predicted),
        "model_name": model_name,
        "aqi_basis": "CPCB PM2.5 sub-index",
    }
