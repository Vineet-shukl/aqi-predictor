"""US EPA Air Quality Index from pollutant concentrations.

Breakpoints follow the EPA AQS table as of 2026, including the PM2.5
revisions that took effect on 6 May 2024. Concentrations are truncated to
the precision of each breakpoint table, then linearly interpolated and
rounded half up to an integer sub-index. The reported AQI is the highest
sub-index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# (low concentration, high concentration, low AQI, high AQI)
Segment = tuple[float, float, int, int]

PM25_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
    (325.5, 99999.9, 501, 999),
)

PM10_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 54.0, 0, 50),
    (55.0, 154.0, 51, 100),
    (155.0, 254.0, 101, 150),
    (255.0, 354.0, 151, 200),
    (355.0, 424.0, 201, 300),
    (425.0, 604.0, 301, 500),
    (605.0, 99999.9, 501, 999),
)

CO_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 50.4, 301, 500),
    (50.5, 99999.9, 501, 999),
)

OZONE_8HR_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
)

OZONE_1HR_SEGMENTS: tuple[Segment, ...] = (
    (0.125, 0.164, 101, 150),
    (0.165, 0.204, 151, 200),
    (0.205, 0.404, 201, 300),
    (0.405, 0.604, 301, 500),
    (0.605, 99999.9, 501, 999),
)

SO2_1HR_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 35.0, 0, 50),
    (36.0, 75.0, 51, 100),
    (76.0, 185.0, 101, 150),
    (186.0, 304.0, 151, 200),
    (305.0, 99999.0, 200, 200),
)

SO2_24HR_SEGMENTS: tuple[Segment, ...] = (
    (305.0, 604.0, 201, 300),
    (605.0, 1004.0, 301, 500),
    (1005.0, 99999.0, 501, 999),
)

NO2_SEGMENTS: tuple[Segment, ...] = (
    (0.0, 53.0, 0, 50),
    (54.0, 100.0, 51, 100),
    (101.0, 360.0, 101, 150),
    (361.0, 649.0, 151, 200),
    (650.0, 1249.0, 201, 300),
    (1250.0, 2049.0, 301, 500),
    (2050.0, 99999.0, 501, 999),
)

CATEGORIES: tuple[tuple[int, int, str, str, str, str], ...] = (
    (0, 50, "Good", "#00e400", "#14210a", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00", "#1c1a05", "Unusually sensitive people should consider limiting prolonged outdoor exertion."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "#1c1004", "Sensitive groups should reduce prolonged or heavy outdoor exertion."),
    (151, 200, "Unhealthy", "#ff0000", "#ffffff", "Everyone should reduce prolonged or heavy outdoor exertion."),
    (201, 300, "Very Unhealthy", "#8f3f97", "#ffffff", "Everyone should avoid prolonged or heavy outdoor exertion."),
    (301, 999, "Hazardous", "#7e0023", "#ffffff", "Everyone should avoid all outdoor exertion."),
)


class AQIError(ValueError):
    """The concentrations cannot be turned into an AQI."""


@dataclass(frozen=True)
class SubIndex:
    pollutant_id: str
    name: str
    aqi: int


@dataclass(frozen=True)
class AQIReport:
    aqi: int
    category: str
    color: str
    text_color: str
    message: str
    dominant: tuple[str, ...]
    pollutants: tuple[SubIndex, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "mode": "epa",
            "aqi": self.aqi,
            "category": self.category,
            "color": self.color,
            "text_color": self.text_color,
            "message": self.message,
            "dominant": list(self.dominant),
            "pollutants": [
                {"id": item.pollutant_id, "name": item.name, "aqi": item.aqi}
                for item in self.pollutants
            ],
            "notes": list(self.notes),
        }


def truncate(value: float, decimals: int) -> float:
    """Drop digits beyond `decimals` without rounding."""
    factor = 10 ** decimals
    return math.floor(value * factor + 1e-8) / factor


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def category_for(aqi: int) -> tuple[str, str, str, str]:
    for low, high, name, color, text_color, message in CATEGORIES:
        if low <= aqi <= high:
            return name, color, text_color, message
    raise AQIError(f"AQI {aqi} is outside the supported scale.")


def _interpolate(concentration: float, segments: tuple[Segment, ...], decimals: int) -> int | None:
    if concentration < 0:
        raise AQIError("Concentrations cannot be negative.")
    clipped = truncate(concentration, decimals)
    last_high = segments[-1][1]
    if clipped > last_high:
        clipped = last_high
    for low, high, index_low, index_high in segments:
        if low <= clipped <= high:
            if index_low == index_high or high == low:
                return index_low
            raw = (index_high - index_low) / (high - low) * (clipped - low) + index_low
            rounded = round_half_up(raw)
            return min(index_high, max(index_low, rounded))
    return None


def _require_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AQIError(f"{label} must be a number.")
    if math.isnan(value) or math.isinf(value):
        raise AQIError(f"{label} must be a finite number.")
    return float(value)


def calculate(
    *,
    pm25: float | None = None,
    pm10: float | None = None,
    ozone_8hr: float | None = None,
    ozone_1hr: float | None = None,
    co: float | None = None,
    so2_1hr: float | None = None,
    so2_24hr: float | None = None,
    no2: float | None = None,
) -> AQIReport:
    """Calculate the AQI from whichever pollutant concentrations are provided.

    Units and averaging periods match the EPA breakpoint tables:
    PM2.5 and PM10 are 24-hour µg/m³, ozone is ppm (8-hour and 1-hour),
    CO is 8-hour ppm, and SO2 and NO2 are ppb.
    """
    provided = {
        "pm25": pm25,
        "pm10": pm10,
        "ozone_8hr": ozone_8hr,
        "ozone_1hr": ozone_1hr,
        "co": co,
        "so2_1hr": so2_1hr,
        "so2_24hr": so2_24hr,
        "no2": no2,
    }
    if all(value is None for value in provided.values()):
        raise AQIError("Enter at least one pollutant concentration.")

    notes: list[str] = []
    subindices: list[SubIndex] = []

    if pm25 is not None:
        aqi = _interpolate(_require_number(pm25, "PM2.5"), PM25_SEGMENTS, 1)
        if aqi is None:
            raise AQIError("PM2.5 is outside the AQI breakpoint table.")
        subindices.append(SubIndex("pm25", "PM2.5", aqi))

    if pm10 is not None:
        aqi = _interpolate(_require_number(pm10, "PM10"), PM10_SEGMENTS, 0)
        if aqi is None:
            raise AQIError("PM10 is outside the AQI breakpoint table.")
        subindices.append(SubIndex("pm10", "PM10", aqi))

    ozone_parts: list[int] = []
    if ozone_8hr is not None:
        value = _require_number(ozone_8hr, "8-hour ozone")
        truncated = truncate(value, 3)
        if truncated > 0.200:
            notes.append(
                "8-hour ozone above 0.200 ppm is outside the 8-hour scale, so that value was not used."
            )
        else:
            aqi = _interpolate(value, OZONE_8HR_SEGMENTS, 3)
            if aqi is None:
                raise AQIError("8-hour ozone is outside the AQI breakpoint table.")
            ozone_parts.append(aqi)
    if ozone_1hr is not None:
        value = _require_number(ozone_1hr, "1-hour ozone")
        truncated = truncate(value, 3)
        if truncated < 0.125:
            notes.append("1-hour ozone below 0.125 ppm does not define an AQI, so that value was not used.")
        else:
            aqi = _interpolate(value, OZONE_1HR_SEGMENTS, 3)
            if aqi is None:
                raise AQIError("1-hour ozone is outside the AQI breakpoint table.")
            ozone_parts.append(aqi)
    if ozone_parts:
        subindices.append(SubIndex("ozone", "Ozone", max(ozone_parts)))

    if co is not None:
        aqi = _interpolate(_require_number(co, "CO"), CO_SEGMENTS, 1)
        if aqi is None:
            raise AQIError("CO is outside the AQI breakpoint table.")
        subindices.append(SubIndex("co", "CO", aqi))

    so2_parts: list[int] = []
    if so2_1hr is not None:
        value = _require_number(so2_1hr, "1-hour SO2")
        truncated = truncate(value, 0)
        aqi = _interpolate(value, SO2_1HR_SEGMENTS, 0)
        if aqi is None:
            raise AQIError("1-hour SO2 is outside the AQI breakpoint table.")
        so2_parts.append(aqi)
        if truncated >= 305:
            notes.append(
                "1-hour SO2 at or above 305 ppb is reported as AQI 200. A 24-hour average is required for a higher category."
            )
    if so2_24hr is not None:
        value = _require_number(so2_24hr, "24-hour SO2")
        truncated = truncate(value, 0)
        if truncated <= 304:
            notes.append("24-hour SO2 at or below 304 ppb does not define an AQI above the 1-hour scale, so that value was not used.")
        else:
            aqi = _interpolate(value, SO2_24HR_SEGMENTS, 0)
            if aqi is None:
                raise AQIError("24-hour SO2 is outside the AQI breakpoint table.")
            so2_parts.append(aqi)
    if so2_parts:
        subindices.append(SubIndex("so2", "SO2", max(so2_parts)))

    if no2 is not None:
        aqi = _interpolate(_require_number(no2, "NO2"), NO2_SEGMENTS, 0)
        if aqi is None:
            raise AQIError("NO2 is outside the AQI breakpoint table.")
        subindices.append(SubIndex("no2", "NO2", aqi))

    if not subindices:
        if notes:
            raise AQIError(" ".join(notes))
        raise AQIError("These concentrations do not define an AQI. Check the averaging period and try again.")

    overall = max(item.aqi for item in subindices)
    name, color, text_color, message = category_for(overall)
    dominant = tuple(item.name for item in subindices if item.aqi == overall)
    return AQIReport(
        aqi=overall,
        category=name,
        color=color,
        text_color=text_color,
        message=message,
        dominant=dominant,
        pollutants=tuple(subindices),
        notes=tuple(notes),
    )
