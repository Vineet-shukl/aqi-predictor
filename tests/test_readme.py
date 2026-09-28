import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_copies_model_metrics():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    for row in metrics["comparison"]:
        assert row["model"] in README
        assert row["mae"] in README
        assert row["rmse"] in README
        assert row["r2"] in README
    assert metrics["cutoff_date"] in README
    assert str(metrics["n_train"]) in README
    assert str(metrics["n_test"]) in README
    assert metrics["best_model"] in README
    assert str(metrics["model_bytes"]) in README
    assert f"{metrics['n_cities_train']} cities" in README
    for key in ("best_model_pre_lockdown", "best_model_from_lockdown"):
        block = metrics[key]
        assert block["mae"] in README
        assert block["rmse"] in README
        assert block["r2"] in README
        assert block["n"] in README


def test_readme_copies_sql_and_eda_highlights():
    highlights_path = ROOT / "reports" / "sql" / "highlights.json"
    highlights = json.loads(highlights_path.read_text(encoding="utf-8"))
    for key in (
        "top_city",
        "top_city_mean_pm25",
        "top_city_median_pm25",
        "lowest_city",
        "lowest_city_mean_pm25",
        "highest_season",
        "highest_season_mean_pm25",
        "pm25_pm10_corr",
        "pm25_missing_pct",
        "pm10_missing_pct",
    ):
        assert str(highlights[key]) in README

    eda = json.loads((ROOT / "reports" / "eda_summary.json").read_text(encoding="utf-8"))
    for key in (
        "n_rows",
        "n_cities",
        "date_min",
        "date_max",
        "pm25_mean",
        "pm25_median",
        "pm25_max",
        "highest_median_city",
        "highest_median_pm25",
        "lowest_median_city",
        "lowest_median_pm25",
        "highest_month_mean_pm25",
        "lowest_month_mean_pm25",
    ):
        assert eda[key] in README


def test_readme_embeds_figures_and_the_worked_example():
    for name in ("city_comparison.png", "monthly_seasonality.png", "correlation_heatmap.png"):
        figure = ROOT / "reports" / "figures" / name
        assert figure.stat().st_size > 5000
        assert f"reports/figures/{name}" in README
    example = json.loads((ROOT / "reports" / "api_example.json").read_text(encoding="utf-8"))
    assert str(example["response"]["predicted_pm25"]) in README
    assert str(example["actual_pm25"]) in README
    assert example["response"]["aqi_category"] in README
    assert example["response"]["date"] in README


def test_notebook_was_executed():
    notebook = json.loads((ROOT / "notebooks" / "eda.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") for cell in code_cells)
    image_outputs = 0
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if "image/png" in (output.get("data") or {}):
                image_outputs += 1
    assert image_outputs >= 3
