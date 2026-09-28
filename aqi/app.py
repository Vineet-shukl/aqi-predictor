"""HTTP API and page for the AQI calculator and weather estimator."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from aqi.calculator import AQIError, calculate
from aqi.model import estimate

POLLUTANT_FIELDS = ("pm25", "pm10", "ozone_8hr", "ozone_1hr", "co", "so2_1hr", "so2_24hr", "no2")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/calculate")
    def calculate_aqi():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object of pollutant concentrations."}), 400
        try:
            values = {field: _optional_number(payload.get(field), field) for field in POLLUTANT_FIELDS}
            report = calculate(**values)
        except AQIError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(report.to_dict())

    @app.post("/api/estimate")
    def estimate_aqi():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object with weather and time."}), 400
        try:
            current_aqi = _required_int(payload.get("current_aqi"), "Current AQI")
            temperature = _required_number(payload.get("temperature_c"), "Temperature")
            humidity = _required_number(payload.get("relative_humidity"), "Relative humidity")
            hour = _required_int(payload.get("hour"), "Hour")
            month = _required_int(payload.get("month"), "Month")
            result = estimate(
                current_aqi=current_aqi,
                temperature_c=temperature,
                relative_humidity=humidity,
                hour=hour,
                month=month,
            )
        except (AQIError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


def _optional_number(value, label: str) -> float | None:
    if value is None or value == "":
        return None
    return _required_number(value, label)


def _required_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise AQIError(f"{label} must be a number.")
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            raise AQIError(f"{label} must be a number.")
        try:
            value = float(text)
        except ValueError as exc:
            raise AQIError(f"{label} must be a number.") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise AQIError(f"{label} must be a finite number.")
    return float(value)


def _required_int(value, label: str) -> int:
    number = _required_number(value, label)
    if int(number) != number:
        raise AQIError(f"{label} must be a whole number.")
    return int(number)
