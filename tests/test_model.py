import json
from pathlib import Path

import joblib

from src.inference import predict_payload

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_reproduces_the_documented_example():
    model = joblib.load(ROOT / "models" / "pm25_model.joblib")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    example = json.loads((ROOT / "reports" / "api_example.json").read_text(encoding="utf-8"))
    result = predict_payload(
        model,
        metadata["cities"],
        metadata["best_model"],
        example["request"],
    )
    assert result == example["response"]
    assert result["predicted_pm25"] >= 0
    assert result["model_name"] == metadata["best_model"]


def test_served_model_is_lowest_rmse_among_trained_models():
    metadata = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    trained = [row for row in metadata["comparison"] if row["model"] != "naive_persistence"]
    best = min(trained, key=lambda row: (float(row["rmse"]), -float(row["r2"]), float(row["mae"])))
    assert metadata["best_model"] == best["model"]
    assert metadata["best_model"] != "naive_persistence"
    assert "lowest test RMSE" in metadata["selection_rule"]
    assert metadata["model_bytes"] < 50 * 1024 * 1024
