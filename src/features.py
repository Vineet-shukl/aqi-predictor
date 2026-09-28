"""Load, clean, and build leakage-safe features for next-day PM2.5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "city_day.csv"

REQUIRED_RAW_COLUMNS = ["City", "Date", "PM2.5"]
OPTIONAL_RAW_COLUMNS = [
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
    "AQI",
    "AQI_Bucket",
]

RENAME = {
    "City": "city",
    "Date": "date",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO": "no",
    "NO2": "no2",
    "NOx": "nox",
    "NH3": "nh3",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "Benzene": "benzene",
    "Toluene": "toluene",
    "Xylene": "xylene",
    "AQI": "aqi",
    "AQI_Bucket": "aqi_bucket",
}

NUMERIC_RAW = [name for name in RENAME if name not in {"City", "Date", "AQI_Bucket"}]

# Same-day PM2.5, AQI, and AQI_Bucket are excluded. Every pollutant feature is lagged.
FEATURE_COLUMNS = [
    "city",
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_7",
    "pm25_lag_14",
    "pm25_roll_mean_7",
    "pm25_roll_mean_14",
    "pm25_roll_std_7",
    "pm25_delta_1",
    "pm10_lag_1",
    "no2_lag_1",
    "no_lag_1",
    "co_lag_1",
    "so2_lag_1",
    "o3_lag_1",
    "month",
    "dayofweek",
    "month_sin",
    "month_cos",
]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "city"]

POLLUTANT_LAGS = (
    ("pm10", "pm10_lag_1"),
    ("no2", "no2_lag_1"),
    ("no", "no_lag_1"),
    ("co", "co_lag_1"),
    ("so2", "so2_lag_1"),
    ("o3", "o3_lag_1"),
)


class ForecastInputError(ValueError):
    """Raised when a forecast request cannot be turned into features."""


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else RAW_PATH
    frame = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse types, drop invalid dates, and expand each city onto a daily calendar.

    Expanding the calendar makes lag-1 mean "yesterday", including when a date
    is absent. On this CPCB extract the published dates are already continuous
    inside each city's span, so the expansion does not add rows, but the step
    keeps the definition honest if a gap appears.
    """
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    out = frame.copy()
    for column in OPTIONAL_RAW_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["City"] = out["City"].astype(str).str.strip()
    out = out.dropna(subset=["Date", "City"])
    out = out[out["City"].ne("") & out["City"].str.lower().ne("nan")]

    for column in NUMERIC_RAW:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        out.loc[out[column] < 0, column] = np.nan

    out = out.rename(columns=RENAME)
    numeric_cols = [RENAME[column] for column in NUMERIC_RAW]

    if out.duplicated(["city", "date"]).any():
        aggregated = out.groupby(["city", "date"], as_index=False)[numeric_cols].mean()
        buckets = (
            out.dropna(subset=["aqi_bucket"])
            .groupby(["city", "date"], as_index=False)["aqi_bucket"]
            .agg(lambda values: values.iloc[0])
        )
        out = aggregated.merge(buckets, on=["city", "date"], how="left")
    else:
        out = out[["city", "date", *numeric_cols, "aqi_bucket"]].copy()

    daily_frames: list[pd.DataFrame] = []
    for city, group in out.groupby("city", sort=False):
        group = group.sort_values("date").set_index("date")
        full_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(full_index)
        group["city"] = city
        group.index.name = "date"
        daily_frames.append(group.reset_index())

    daily = pd.concat(daily_frames, ignore_index=True)
    return daily.sort_values(["city", "date"]).reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add past-only lags, rolling stats, and calendar fields.

    Rolling windows are applied to the series after shift(1), so the value on
    the target date is not included. Calendar fields describe the target date,
    which is known when the forecast is issued.
    """
    required = {"city", "date", "pm25", "pm10", "no", "no2", "co", "so2", "o3"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature frame is missing columns: {sorted(missing)}")

    out = frame.sort_values(["city", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    grouped = out.groupby("city", group_keys=False)

    out["pm25_lag_1"] = grouped["pm25"].shift(1)
    out["pm25_lag_2"] = grouped["pm25"].shift(2)
    out["pm25_lag_3"] = grouped["pm25"].shift(3)
    out["pm25_lag_7"] = grouped["pm25"].shift(7)
    out["pm25_lag_14"] = grouped["pm25"].shift(14)
    out["pm25_roll_mean_7"] = grouped["pm25"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    out["pm25_roll_mean_14"] = grouped["pm25"].transform(
        lambda series: series.shift(1).rolling(14, min_periods=14).mean()
    )
    out["pm25_roll_std_7"] = grouped["pm25"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).std()
    )
    out["pm25_delta_1"] = out["pm25_lag_1"] - out["pm25_lag_2"]
    for source, destination in POLLUTANT_LAGS:
        out[destination] = grouped[source].shift(1)

    out["month"] = out["date"].dt.month.astype(int)
    out["dayofweek"] = out["date"].dt.dayofweek.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def prepare(path: Path | None = None) -> pd.DataFrame:
    return add_features(clean(load_raw(path)))


def supervised_frame(featured: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where the target and the persistence feature are both observed."""
    columns = ["date", "pm25", *FEATURE_COLUMNS]
    present = [column for column in columns if column in featured.columns]
    out = featured.dropna(subset=["pm25", "pm25_lag_1"]).loc[:, present].copy()
    return out.reset_index(drop=True)


def time_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, list[str]]:
    """Split on a single cutoff date. Training dates are strictly earlier.

    Cities that never appear before the cutoff are removed from the test set.
    Scoring them would ask the model for a city effect it could not have learned.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    dates = np.sort(frame["date"].unique())
    if len(dates) < 2:
        raise ValueError("Need at least two distinct dates to split")
    cut_index = int(len(dates) * (1 - test_fraction))
    cut_index = min(max(cut_index, 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[cut_index])
    train = frame[frame["date"] < cutoff].copy()
    test = frame[frame["date"] >= cutoff].copy()
    train_cities = set(train["city"].unique())
    excluded = sorted(set(test["city"].unique()) - train_cities)
    test = test[test["city"].isin(train_cities)].copy()
    return train, test, cutoff, excluded


def features_for_forecast(
    history: pd.DataFrame, city: str, forecast_date: pd.Timestamp
) -> pd.Series:
    """Build one feature row for `forecast_date` from prior daily observations."""
    forecast_date = pd.Timestamp(forecast_date).normalize()
    hist = history.copy()
    if "date" not in hist.columns or "pm25" not in hist.columns:
        raise ForecastInputError("History must include date and pm25")
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    if hist["date"].isna().any():
        raise ForecastInputError("History contains an unparseable date")
    if (hist["date"] >= forecast_date).any():
        raise ForecastInputError("History dates must be before the forecast date")
    if hist["date"].duplicated().any():
        raise ForecastInputError("History dates must be unique")
    if hist.empty:
        raise ForecastInputError("History must contain at least one day")

    span_days = int((forecast_date - hist["date"].min()).days)
    if span_days > 60:
        raise ForecastInputError("History spans more than 60 days; send a shorter window")

    for column in ("pm10", "no", "no2", "co", "so2", "o3"):
        if column not in hist.columns:
            hist[column] = np.nan
        hist[column] = pd.to_numeric(hist[column], errors="coerce")
    hist["pm25"] = pd.to_numeric(hist["pm25"], errors="coerce")

    hist = hist[["date", "pm25", "pm10", "no", "no2", "co", "so2", "o3"]]
    placeholder = {
        "date": forecast_date,
        "pm25": np.nan,
        "pm10": np.nan,
        "no": np.nan,
        "no2": np.nan,
        "co": np.nan,
        "so2": np.nan,
        "o3": np.nan,
    }
    frame = pd.concat([hist, pd.DataFrame([placeholder])], ignore_index=True)
    frame = frame.set_index("date").sort_index()
    full_index = pd.date_range(frame.index.min(), forecast_date, freq="D")
    frame = frame.reindex(full_index)
    frame["city"] = city
    frame.index.name = "date"
    frame = frame.reset_index()
    featured = add_features(frame)
    row = featured.loc[featured["date"] == forecast_date]
    if row.empty:
        raise ForecastInputError("Could not build features for the forecast date")
    return row.iloc[0]


def resolve_city(name: str, known_cities: list[str]) -> str | None:
    key = name.strip().casefold()
    if not key:
        return None
    for city in known_cities:
        if city.casefold() == key:
            return city
    return None
