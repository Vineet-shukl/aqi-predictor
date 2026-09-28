import json
from pathlib import Path

from sql.run_analysis import run, split_named_queries

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_sql_has_the_core_questions():
    sql = (ROOT / "sql" / "analysis.sql").read_text(encoding="utf-8")
    names = [name for name, _statement in split_named_queries(sql)]
    assert names == [
        "city_pm25_ranking",
        "monthly_trend",
        "seasonal_trend",
        "yearly_trend",
        "pollutant_correlations",
        "aqi_bucket_share",
        "missingness",
        "city_season_pm25",
    ]


def test_sql_runner_writes_ranking_and_highlights(tmp_path: Path):
    summary = run(ROOT / "data" / "city_day.csv", tmp_path)
    ranking = (tmp_path / "city_pm25_ranking.csv").read_text(encoding="utf-8")
    assert "mean_pm25" in ranking
    assert summary["top_city"]
    assert float(summary["top_city_mean_pm25"]) > 0
    assert float(summary["pm25_pm10_corr"]) > 0
    committed_path = ROOT / "reports" / "sql" / "highlights.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    assert summary == committed
