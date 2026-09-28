"""Train baseline and supervised models with a time-based split and save the best one.

Run from the repository root:

    python -m src.train
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.download import sha256_file
from src.estimators import PersistenceRegressor
from src.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_PATH,
    prepare,
    supervised_frame,
    time_split,
)
from src.inference import predict_payload

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pm25_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
EXAMPLE_PATH = ROOT / "reports" / "api_example.json"
EXAMPLE_REQUEST_PATH = ROOT / "reports" / "api_example_request.json"

# Capacities are fixed before looking at test scores. Tree size is capped so the
# committed artifact stays small.
RANDOM_STATE = 42
BASELINE_NAME = "naive_persistence"
# Persistence is scored and shown. It is not eligible to be the served model.
SELECTION_RULE = (
    "lowest test RMSE among trained models (naive_persistence is scored but not served); "
    "ties break on higher R², then lower MAE"
)
MAX_MODEL_BYTES = 50 * 1024 * 1024


def _numeric_pipeline(scale: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _preprocessor(scale: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(scale), NUMERIC_FEATURES),
            (
                "city",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["city"],
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "naive_persistence": PersistenceRegressor(),
        "linear_regression": Pipeline(
            [
                ("preprocess", _preprocessor(scale=True)),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor(scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=10,
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", _preprocessor(scale=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.06,
                        max_depth=6,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def regression_scores(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _fmt(scores: dict[str, float]) -> dict[str, str]:
    return {
        "mae": f"{scores['mae']:.3f}",
        "rmse": f"{scores['rmse']:.3f}",
        "r2": f"{scores['r2']:.3f}",
    }


def _json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.floating | float):
        return round(float(value), 4)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _history_records(daily: pd.DataFrame, city: str, forecast_date: pd.Timestamp) -> list[dict]:
    start = forecast_date - pd.Timedelta(days=14)
    window = daily[
        (daily["city"] == city) & (daily["date"] >= start) & (daily["date"] < forecast_date)
    ].sort_values("date")
    records = []
    for row in window.itertuples(index=False):
        records.append(
            {
                "date": pd.Timestamp(row.date).date().isoformat(),
                "pm25": _json_value(row.pm25),
                "pm10": _json_value(row.pm10),
                "no": _json_value(row.no),
                "no2": _json_value(row.no2),
                "co": _json_value(row.co),
                "so2": _json_value(row.so2),
                "o3": _json_value(row.o3),
            }
        )
    return records


def _select_example(daily: pd.DataFrame, test: pd.DataFrame) -> tuple[str, pd.Timestamp]:
    """Pick a test day that has a complete 14-day PM2.5 history, preferring Delhi."""
    candidates = test.sort_values("date")
    preferred = candidates[candidates["city"] == "Delhi"]
    ordered = pd.concat([preferred, candidates[candidates["city"] != "Delhi"]], ignore_index=True)
    for row in ordered.itertuples(index=False):
        forecast_date = pd.Timestamp(row.date)
        history = _history_records(daily, row.city, forecast_date)
        if len(history) < 14:
            continue
        if any(item["pm25"] is None for item in history):
            continue
        yesterday = (forecast_date - pd.Timedelta(days=1)).date().isoformat()
        if history[-1]["date"] != yesterday:
            continue
        return row.city, forecast_date
    raise RuntimeError("Could not find a test row with 14 complete prior PM2.5 days")


def _slice_scores(y_true, y_pred, dates: pd.Series, start: str, end: str) -> dict[str, str] | None:
    mask = ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).to_numpy()
    if int(mask.sum()) == 0:
        return None
    scored = _fmt(regression_scores(y_true[mask], y_pred[mask]))
    scored["n"] = str(int(mask.sum()))
    return scored


def main() -> None:
    daily = prepare(RAW_PATH)
    supervised = supervised_frame(daily)
    train, test, cutoff, excluded = time_split(supervised, test_fraction=0.2)
    if train.empty or test.empty:
        raise RuntimeError("Time split produced an empty train or test set")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("Time split leaked future dates into training")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["pm25"].to_numpy(dtype=float)
    x_test = test[FEATURE_COLUMNS]
    y_test = test["pm25"].to_numpy(dtype=float)

    fitted: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        scores = regression_scores(y_test, pred)
        fitted[name] = model
        predictions[name] = pred
        rows.append({"model": name, **_fmt(scores), "n_test": int(len(y_test))})
        print(
            f"{name:24s}  MAE {scores['mae']:.3f}  "
            f"RMSE {scores['rmse']:.3f}  R2 {scores['r2']:.3f}"
        )

    trained_rows = [row for row in rows if row["model"] != BASELINE_NAME]
    if not trained_rows:
        raise RuntimeError("No trained model available to serve")
    best_name = min(
        trained_rows,
        key=lambda row: (float(row["rmse"]), -float(row["r2"]), float(row["mae"])),
    )["model"]
    best_model = fitted[str(best_name)]
    best_pred = predictions[str(best_name)]

    lockdown_start = "2020-03-25"
    test_dates = pd.to_datetime(test["date"])
    pre_lock = _slice_scores(y_test, best_pred, test_dates, "1900-01-01", "2020-03-24")
    during_lock = _slice_scores(
        y_test, best_pred, test_dates, lockdown_start, str(test_dates.max().date())
    )

    example_city, example_date = _select_example(daily, test)
    example_request = {
        "city": example_city,
        "date": example_date.date().isoformat(),
        "history": _history_records(daily, example_city, example_date),
    }
    known_cities = sorted(train["city"].unique())
    example_response = predict_payload(best_model, known_cities, str(best_name), example_request)
    actual = daily[(daily["city"] == example_city) & (daily["date"] == example_date)]["pm25"]
    if actual.empty or pd.isna(actual.iloc[0]):
        actual_pm25 = None
    else:
        actual_pm25 = round(float(actual.iloc[0]), 2)

    metadata = {
        "owner": "Vineet Shukla",
        "target": "pm25",
        "target_unit": "µg/m³",
        "task": "Predict city-day PM2.5 on date t using only observations from dates before t",
        "best_model": best_name,
        "comparison": rows,
        "cutoff_date": cutoff.date().isoformat(),
        "train_date_min": pd.Timestamp(train["date"].min()).date().isoformat(),
        "train_date_max": pd.Timestamp(train["date"].max()).date().isoformat(),
        "test_date_min": pd.Timestamp(test["date"].min()).date().isoformat(),
        "test_date_max": pd.Timestamp(test["date"].max()).date().isoformat(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_cities_train": int(train["city"].nunique()),
        "cities": known_cities,
        "cities_excluded_from_test": excluded,
        "features": FEATURE_COLUMNS,
        "selection_rule": SELECTION_RULE,
        "test_fraction": 0.2,
        "data_sha256": sha256_file(RAW_PATH),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "best_model_pre_lockdown": pre_lock,
        "best_model_from_lockdown": during_lock,
        "lockdown_split_date": lockdown_start,
        "example_actual_pm25": actual_pm25,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH, compress=3)
    model_bytes = MODEL_PATH.stat().st_size
    if model_bytes > MAX_MODEL_BYTES:
        raise RuntimeError(f"Model artifact is {model_bytes} bytes, above the 50 MB cap")
    metadata["model_bytes"] = model_bytes
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    example = {
        "request": example_request,
        "response": example_response,
        "actual_pm25": actual_pm25,
        "note": (
            "Single held-out day from the time-based test period. "
            "Not a substitute for the test-set metrics."
        ),
    }
    EXAMPLE_PATH.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    EXAMPLE_REQUEST_PATH.write_text(
        json.dumps(example_request, indent=2) + "\n", encoding="utf-8"
    )

    print(f"best_model {best_name}")
    print(f"cutoff {metadata['cutoff_date']}")
    print(f"n_train {metadata['n_train']} n_test {metadata['n_test']}")
    print(f"excluded {excluded}")
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
    print(f"example {example_city} {example_date.date().isoformat()} -> {example_response}")


if __name__ == "__main__":
    main()
