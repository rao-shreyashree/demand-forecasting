import math

import pandas as pd
import pytest

from src.metrics import bias, forecast_errors, mape, rmse, wape


def test_wape_hand_calculated():
    # |10-12| + |20-18| = 4 ; sum(actual) = 30
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_wape_perfect_forecast_is_zero():
    assert wape([5, 0, 7], [5, 0, 7]) == 0


def test_wape_handles_zero_sales_days():
    assert wape([0, 10], [2, 8]) == pytest.approx(4 / 10)


def test_wape_all_zero_actuals_is_nan():
    assert math.isnan(wape([0, 0], [1, 1]))


def test_mape_hand_calculated():
    # |10-12|/10 = 0.2 ; |20-18|/20 = 0.1 -> mean 0.15
    assert mape([10, 20], [12, 18]) == pytest.approx(0.15)


def test_mape_skips_zero_actuals():
    # zero row skipped; remaining |10-12|/10 = 0.2
    assert mape([0, 10], [5, 12]) == pytest.approx(0.2)


def test_mape_all_zero_actuals_is_nan():
    assert math.isnan(mape([0, 0], [1, 2]))


def test_rmse_hand_calculated():
    # errors 2 and -2 -> sqrt((4+4)/2) = 2
    assert rmse([10, 20], [12, 18]) == pytest.approx(2.0)


def test_rmse_perfect_forecast_is_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0


def test_bias_sign():
    assert bias([10, 10], [12, 14]) == pytest.approx(3.0)    # over-forecast
    assert bias([10, 10], [8, 6]) == pytest.approx(-3.0)     # under-forecast


@pytest.mark.parametrize("fn", [wape, mape, rmse, bias])
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError):
        fn([1, 2, 3], [1, 2])


@pytest.mark.parametrize("fn", [wape, mape, rmse, bias])
def test_empty_input_raises(fn):
    with pytest.raises(ValueError):
        fn([], [])


def _frames():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    actual = pd.DataFrame(
        {"date": dates, "store": 1, "item": 1, "sales": [10, 20]}
    )
    forecast = pd.DataFrame(
        {"date": dates, "store": 1, "item": 1, "y_hat": [12, 18], "model": "m"}
    )
    return actual, forecast


def test_forecast_errors_actual_minus_forecast():
    actual, forecast = _frames()
    out = forecast_errors(actual, forecast)
    assert list(out.columns) == ["date", "store", "item", "error"]
    assert out["error"].tolist() == [-2, 2]


def test_forecast_errors_missing_column_raises():
    actual, forecast = _frames()
    with pytest.raises(ValueError):
        forecast_errors(actual.drop(columns="sales"), forecast)


def test_forecast_errors_no_overlap_raises():
    actual, forecast = _frames()
    forecast["date"] = forecast["date"] + pd.Timedelta(days=30)
    with pytest.raises(ValueError):
        forecast_errors(actual, forecast)