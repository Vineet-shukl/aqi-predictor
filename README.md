# AQI Predictor

Calculate the US EPA Air Quality Index from pollutant concentrations, and estimate an index from temperature, humidity, and time of day.

The calculator uses the EPA AQS breakpoint table, including the PM2.5 revisions effective 6 May 2024. Concentrations are truncated to the table precision, interpolated inside the matching breakpoint, and rounded half up. The reported index is the highest pollutant sub-index.

The next-hour model is a gradient-boosting regressor trained on consecutive hours from the [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) roadside station in Italy (March 2004–April 2005). Each hour is labeled with an EPA-style index from that hour's reference CO and NO2 readings. Given the current index, temperature, and humidity, it predicts the following hour at that station. It is not a forecast for another city. Regulatory CO uses an 8-hour average; these labels use the hourly reading as a proxy.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000.

## Test

```bash
pytest
```

## Retrain

`data/hourly.csv` is the training table. Rebuild the saved model with:

```bash
python -m aqi.model
```
