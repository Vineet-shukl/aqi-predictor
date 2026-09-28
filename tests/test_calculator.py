import math

import pytest

from aqi.calculator import AQIError, calculate, truncate


def test_truncates_without_rounding():
    assert truncate(9.19, 1) == 9.1
    assert truncate(9.09, 1) == 9.0
    assert truncate(0.1249, 3) == 0.124


@pytest.mark.parametrize(
    ("pm25", "expected"),
    [
        (0.0, 0),
        (9.0, 50),
        (9.1, 51),
        (12.0, 56),
        (35.4, 100),
        (35.5, 101),
        (55.4, 150),
        (55.5, 151),
        (125.4, 200),
        (125.5, 201),
        (225.4, 300),
        (225.5, 301),
        (325.4, 500),
        (9.19, 51),
        (9.09, 50),
    ],
)
def test_pm25_breakpoints(pm25, expected):
    assert calculate(pm25=pm25).aqi == expected


def test_pm25_category_and_dominant():
    report = calculate(pm25=12.0)
    assert report.category == "Moderate"
    assert report.dominant == ("PM2.5",)


def test_other_pollutant_boundaries():
    assert calculate(co=0).aqi == 0
    assert calculate(co=4.4).aqi == 50
    assert calculate(co=4.5).aqi == 51
    assert calculate(co=2.2).aqi == 25
    assert calculate(co=50.4).aqi == 500
    assert calculate(no2=53).aqi == 50
    assert calculate(no2=54).aqi == 51
    assert calculate(no2=100).aqi == 100
    assert calculate(no2=101).aqi == 101
    assert calculate(pm10=54).aqi == 50
    assert calculate(pm10=55).aqi == 51
    assert calculate(pm10=154).aqi == 100
    assert calculate(ozone_8hr=0.054).aqi == 50
    assert calculate(ozone_8hr=0.055).aqi == 51
    assert calculate(ozone_8hr=0.070).aqi == 100
    assert calculate(ozone_8hr=0.200).aqi == 300
    assert calculate(ozone_1hr=0.125).aqi == 101
    assert calculate(ozone_1hr=0.164).aqi == 150
    assert calculate(so2_1hr=35).aqi == 50
    assert calculate(so2_1hr=36).aqi == 51
    assert calculate(so2_1hr=304).aqi == 200
    assert calculate(so2_1hr=305).aqi == 200
    assert calculate(so2_24hr=305).aqi == 201


def test_overall_index_is_the_highest_subindex():
    report = calculate(pm25=12.0, no2=100)
    assert report.aqi == 100
    assert report.dominant == ("NO2",)
    assert {item.name: item.aqi for item in report.pollutants} == {"PM2.5": 56, "NO2": 100}


def test_tied_pollutants_are_both_dominant():
    report = calculate(pm25=9.0, co=4.4)
    assert report.aqi == 50
    assert report.dominant == ("PM2.5", "CO")


def test_ozone_uses_the_higher_of_the_two_averages():
    report = calculate(ozone_8hr=0.070, ozone_1hr=0.164)
    assert report.aqi == 150
    assert report.dominant == ("Ozone",)


def test_low_one_hour_ozone_is_ignored():
    report = calculate(ozone_8hr=0.054, ozone_1hr=0.100)
    assert report.aqi == 50
    assert any("1-hour ozone" in note for note in report.notes)


def test_high_eight_hour_ozone_is_ignored_when_one_hour_exists():
    report = calculate(ozone_8hr=0.250, ozone_1hr=0.165)
    assert report.aqi == 151
    assert any("8-hour ozone" in note for note in report.notes)


def test_eight_hour_ozone_alone_above_scale_is_rejected():
    with pytest.raises(AQIError, match="8-hour ozone"):
        calculate(ozone_8hr=0.250)


def test_so2_one_hour_cap_note():
    report = calculate(so2_1hr=400)
    assert report.aqi == 200
    assert any("305 ppb" in note for note in report.notes)


def test_twenty_four_hour_so2_can_exceed_the_one_hour_cap():
    report = calculate(so2_1hr=400, so2_24hr=305)
    assert report.aqi == 201
    assert report.category == "Very Unhealthy"


def test_empty_and_negative_inputs():
    with pytest.raises(AQIError, match="at least one"):
        calculate()
    with pytest.raises(AQIError, match="negative"):
        calculate(pm25=-1)


def test_half_up_rounding_for_carbon_monoxide():
    # 50/4.4 * 0.4 = 4.545... which rounds half up to 5.
    assert calculate(co=0.4).aqi == 5
    assert math.isclose(50 / 4.4 * 0.4, 4.545454, rel_tol=1e-6)
