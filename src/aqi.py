"""CPCB PM2.5 sub-index and AQI category.

Breakpoints follow the Central Pollution Control Board National Air Quality
Index (24-hour PM2.5). The category is the PM2.5 sub-index category, not the
full multi-pollutant AQI (which takes the maximum sub-index across pollutants).

Concentration bands (µg/m³) and index bands:
    0–30   -> 0–50    Good
    30–60  -> 51–100  Satisfactory
    60–90  -> 101–200 Moderate
    90–120 -> 201–300 Poor
    120–250 -> 301–400 Very Poor
    250–380 -> 401–500 Severe

Values above 380 µg/m³ stay in Severe. The numeric sub-index is extrapolated
with the slope of the last published segment.
"""

from __future__ import annotations

# (conc_lo, conc_hi, index_lo, index_hi)
BREAKPOINTS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 30.0, 0.0, 50.0),
    (30.0, 60.0, 50.0, 100.0),
    (60.0, 90.0, 100.0, 200.0),
    (90.0, 120.0, 200.0, 300.0),
    (120.0, 250.0, 300.0, 400.0),
    (250.0, 380.0, 400.0, 500.0),
)


def pm25_subindex(pm25: float) -> float:
    """Linearly interpolate the CPCB PM2.5 sub-index."""
    concentration = float(pm25)
    if concentration < 0:
        raise ValueError("PM2.5 must be non-negative")
    for conc_lo, conc_hi, index_lo, index_hi in BREAKPOINTS:
        if concentration <= conc_hi:
            span = conc_hi - conc_lo
            return (index_hi - index_lo) / span * (concentration - conc_lo) + index_lo
    conc_lo, conc_hi, index_lo, index_hi = BREAKPOINTS[-1]
    span = conc_hi - conc_lo
    return (index_hi - index_lo) / span * (concentration - conc_lo) + index_lo


def pm25_category(pm25: float) -> str:
    """Map a PM2.5 concentration to the CPCB category name."""
    concentration = float(pm25)
    if concentration < 0:
        raise ValueError("PM2.5 must be non-negative")
    if concentration <= 30:
        return "Good"
    if concentration <= 60:
        return "Satisfactory"
    if concentration <= 90:
        return "Moderate"
    if concentration <= 120:
        return "Poor"
    if concentration <= 250:
        return "Very Poor"
    return "Severe"
