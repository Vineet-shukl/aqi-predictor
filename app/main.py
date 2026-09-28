"""PM2.5 forecast API.

Interactive docs (local): http://127.0.0.1:8000/docs
On a Hugging Face Docker Space the same docs are served at `/docs` on port 7860.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features import ForecastInputError
from src.inference import predict_payload

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "pm25_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"


class HistoryDay(BaseModel):
    date: date
    pm25: float = Field(..., ge=0, le=2000)
    pm10: float | None = Field(None, ge=0, le=2000)
    no: float | None = Field(None, ge=0, le=500)
    no2: float | None = Field(None, ge=0, le=500)
    co: float | None = Field(None, ge=0, le=50)
    so2: float | None = Field(None, ge=0, le=500)
    o3: float | None = Field(None, ge=0, le=500)


class PredictRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=80)
    date: date
    history: list[HistoryDay] = Field(..., min_length=1, max_length=60)

    @field_validator("city")
    @classmethod
    def city_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("city must not be blank")
        return value.strip()


class PredictResponse(BaseModel):
    city: str
    date: date
    predicted_pm25: float
    aqi_subindex: float
    aqi_category: str
    model_name: str
    aqi_basis: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_cities: int
    target: str


def _load_artifacts() -> tuple[object, dict]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError(
            "Model artifacts are missing. Train them with `python -m src.train`."
        )
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


MODEL, METADATA = _load_artifacts()

app = FastAPI(
    title="India PM2.5 Predictor",
    description=(
        "One-day-ahead city-level PM2.5 forecast trained on CPCB city-day "
        "measurements (2015–2020). The AQI category is the CPCB category of the "
        "PM2.5 sub-index. OpenAPI interactive docs: /docs"
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=str(METADATA["best_model"]),
        n_cities=len(METADATA["cities"]),
        target=str(METADATA["target"]),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    body = payload.model_dump(mode="json")
    try:
        result = predict_payload(
            MODEL,
            list(METADATA["cities"]),
            str(METADATA["best_model"]),
            body,
        )
    except ForecastInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(**result)
