import pytest

from src.aqi import pm25_category, pm25_subindex


def test_category_breakpoints():
    assert pm25_category(0) == "Good"
    assert pm25_category(30) == "Good"
    assert pm25_category(30.01) == "Satisfactory"
    assert pm25_category(60) == "Satisfactory"
    assert pm25_category(60.01) == "Moderate"
    assert pm25_category(90) == "Moderate"
    assert pm25_category(90.01) == "Poor"
    assert pm25_category(120) == "Poor"
    assert pm25_category(120.01) == "Very Poor"
    assert pm25_category(250) == "Very Poor"
    assert pm25_category(250.01) == "Severe"
    assert pm25_category(500) == "Severe"


def test_subindex_knots():
    assert pm25_subindex(0) == pytest.approx(0)
    assert pm25_subindex(15) == pytest.approx(25)
    assert pm25_subindex(30) == pytest.approx(50)
    assert pm25_subindex(60) == pytest.approx(100)
    assert pm25_subindex(90) == pytest.approx(200)
    assert pm25_subindex(120) == pytest.approx(300)
    assert pm25_subindex(250) == pytest.approx(400)
    assert pm25_subindex(380) == pytest.approx(500)


def test_subindex_extrapolates_above_published_table():
    assert pm25_subindex(510) > 500


def test_negative_pm25_rejected():
    with pytest.raises(ValueError):
        pm25_category(-1)
    with pytest.raises(ValueError):
        pm25_subindex(-0.1)
