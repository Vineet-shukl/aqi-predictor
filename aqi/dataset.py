"""Build hourly training rows from the UCI Air Quality data set.

Reference analyzer concentrations are converted to the units used by the EPA
AQI tables (CO mg/m³ → ppm, NO2 µg/m³ → ppb at 25°C and 1 atm) and labeled
with the same calculator the app uses. Those labels are an hourly proxy: the
regulatory CO index uses an 8-hour average, and this file does not contain
PM or ozone reference values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aqi.calculator import calculate

ROOT = Path(__file__).resolve().parent.parent
HOURLY_PATH = ROOT / "data" / "hourly.csv"

# EPA molar volume at 25°C and 1 atm.
MOLAR_VOLUME_L = 24.45
CO_MOLECULAR_WEIGHT = 28.01
NO2_MOLECULAR_WEIGHT = 46.0055

HOURLY_FIELDS = (
    "timestamp",
    "hour",
    "month",
    "temperature_c",
    "relative_humidity",
    "co_ppm",
    "no2_ppb",
    "aqi",
)


@dataclass(frozen=True)
class HourlyRow:
    timestamp: datetime
    hour: int
    month: int
    temperature_c: float
    relative_humidity: float
    co_ppm: float
    no2_ppb: float
    aqi: int


def co_mg_to_ppm(mg_per_m3: float) -> float:
    return mg_per_m3 * MOLAR_VOLUME_L / CO_MOLECULAR_WEIGHT


def no2_ug_to_ppb(ug_per_m3: float) -> float:
    return ug_per_m3 * MOLAR_VOLUME_L / NO2_MOLECULAR_WEIGHT


def _european_float(raw: str) -> float | None:
    text = raw.strip().replace(",", ".")
    if text == "":
        return None
    return float(text)


def rows_from_uci_text(text: str) -> list[HourlyRow]:
    """Parse the semicolon-separated UCI export and drop missing analyzer rows."""
    parsed: list[HourlyRow] = []
    reader = csv.reader(text.splitlines(), delimiter=";")
    next(reader, None)
    for record in reader:
        if len(record) < 15:
            continue
        co_mg = _european_float(record[2])
        no2_ug = _european_float(record[9])
        temperature = _european_float(record[12])
        humidity = _european_float(record[13])
        if None in (co_mg, no2_ug, temperature, humidity):
            continue
        if min(co_mg, no2_ug, temperature, humidity) <= -100:
            continue
        try:
            timestamp = datetime.strptime(f"{record[0].strip()} {record[1].strip()}", "%d/%m/%Y %H.%M.%S")
        except ValueError:
            continue
        co_ppm = co_mg_to_ppm(co_mg)
        no2_ppb = no2_ug_to_ppb(no2_ug)
        report = calculate(co=co_ppm, no2=no2_ppb)
        parsed.append(
            HourlyRow(
                timestamp=timestamp,
                hour=timestamp.hour,
                month=timestamp.month,
                temperature_c=temperature,
                relative_humidity=humidity,
                co_ppm=co_ppm,
                no2_ppb=no2_ppb,
                aqi=report.aqi,
            )
        )
    parsed.sort(key=lambda row: row.timestamp)
    return parsed


def write_hourly(rows: list[HourlyRow], path: Path = HOURLY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HOURLY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": row.timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "hour": row.hour,
                    "month": row.month,
                    "temperature_c": f"{row.temperature_c:.4f}",
                    "relative_humidity": f"{row.relative_humidity:.4f}",
                    "co_ppm": f"{row.co_ppm:.6f}",
                    "no2_ppb": f"{row.no2_ppb:.6f}",
                    "aqi": row.aqi,
                }
            )


def load_hourly(path: Path = HOURLY_PATH) -> list[HourlyRow]:
    rows: list[HourlyRow] = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                HourlyRow(
                    timestamp=datetime.strptime(record["timestamp"], "%Y-%m-%dT%H:%M:%S"),
                    hour=int(record["hour"]),
                    month=int(record["month"]),
                    temperature_c=float(record["temperature_c"]),
                    relative_humidity=float(record["relative_humidity"]),
                    co_ppm=float(record["co_ppm"]),
                    no2_ppb=float(record["no2_ppb"]),
                    aqi=int(record["aqi"]),
                )
            )
    return rows
