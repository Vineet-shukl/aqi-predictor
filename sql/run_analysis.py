#!/usr/bin/env python3
"""Load city_day.csv into DuckDB, run sql/analysis.sql, and write the outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "city_day.csv"
SQL_PATH = ROOT / "sql" / "analysis.sql"
OUT_DIR = ROOT / "reports" / "sql"


def split_named_queries(sql: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^-- name:\s*(\w+)\s*$", sql)
    queries: list[tuple[str, str]] = []
    iterator = iter(parts[1:])
    for name, body in zip(iterator, iterator, strict=True):
        statement = body.strip()
        if statement.endswith(";"):
            statement = statement[:-1].strip()
        if not statement:
            raise ValueError(f"Query '{name}' is empty")
        queries.append((name, statement))
    if not queries:
        raise ValueError(f"No named queries found in {SQL_PATH}")
    return queries


def load_city_day(connection: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TABLE city_day AS
        SELECT
            City AS city,
            CAST(Date AS DATE) AS date,
            "PM2.5" AS pm25,
            PM10 AS pm10,
            "NO" AS no,
            NO2 AS no2,
            NOx AS nox,
            NH3 AS nh3,
            CO AS co,
            SO2 AS so2,
            O3 AS o3,
            Benzene AS benzene,
            Toluene AS toluene,
            Xylene AS xylene,
            AQI AS aqi,
            AQI_Bucket AS aqi_bucket
        FROM read_csv(?, header = true)
        """,
        [str(csv_path)],
    )


def _num(value: object, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def highlights_from(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    ranking = frames["city_pm25_ranking"]
    bottom = ranking.iloc[-1]
    top = ranking.iloc[0]
    season = frames["seasonal_trend"].iloc[0]
    corr = frames["pollutant_correlations"].iloc[0]
    missing = frames["missingness"].iloc[0]
    return {
        "top_city": str(top["city"]),
        "top_city_mean_pm25": _num(top["mean_pm25"], 2),
        "top_city_median_pm25": _num(top["median_pm25"], 2),
        "top_city_mean_aqi": _num(top["mean_aqi"], 2),
        "top_city_n_days": int(top["n_days"]),
        "lowest_city": str(bottom["city"]),
        "lowest_city_mean_pm25": _num(bottom["mean_pm25"], 2),
        "highest_season": str(season["season"]),
        "highest_season_mean_pm25": _num(season["mean_pm25"], 2),
        "pm25_pm10_corr": _num(corr["pm25_pm10"], 3),
        "pm25_aqi_corr": _num(corr["pm25_aqi"], 3),
        "pm25_missing_pct": _num(missing["pm25_pct"], 2),
        "pm10_missing_pct": _num(missing["pm10_pct"], 2),
        "xylene_missing_pct": _num(missing["xylene_pct"], 2),
    }


def run(csv_path: Path = CSV_PATH, out_dir: Path = OUT_DIR) -> dict[str, object]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run `python data/download.py` first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    load_city_day(connection, csv_path)
    frames: dict[str, pd.DataFrame] = {}
    for name, statement in split_named_queries(SQL_PATH.read_text(encoding="utf-8")):
        frame = connection.execute(statement).fetchdf()
        frames[name] = frame
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows -> {out_dir / f'{name}.csv'}")
    summary = highlights_from(frames)
    (out_dir / "highlights.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
