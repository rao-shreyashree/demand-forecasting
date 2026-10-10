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
    assert list(out.columns) == ["date", "store", "item", "model", "error"]
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


def test_bias_is_negative_of_mean_error():
    actual, forecast = _frames()
    err = forecast_errors(actual, forecast)["error"].mean()
    assert bias(actual["sales"], forecast["y_hat"]) == pytest.approx(-err)


def test_forecast_errors_keeps_model_column_for_multiple_models():
    actual, forecast = _frames()
    other = forecast.copy()
    other["model"] = "ma_7"
    other["y_hat"] = [10, 25]
    out = forecast_errors(actual, pd.concat([forecast, other], ignore_index=True))
    assert len(out) == 4                      # 2 dates x 2 models, no mixing
    assert out[out["model"] == "m"]["error"].tolist() == [-2, 2]
    assert out[out["model"] == "ma_7"]["error"].tolist() == [0, -5]


def test_forecast_errors_duplicate_keys_without_model_raises():
    actual, forecast = _frames()
    dup = pd.concat([forecast, forecast], ignore_index=True).drop(columns="model")
    with pytest.raises(ValueError):
        forecast_errors(actual, dup)


def test_forecast_errors_duplicate_actual_raises():
    actual, forecast = _frames()
    with pytest.raises(ValueError):
        forecast_errors(pd.concat([actual, actual]), forecast)


def test_forecast_errors_partial_window_raises_when_strict():
    actual, forecast = _frames()
    with pytest.raises(ValueError):
        forecast_errors(actual.iloc[:1], forecast)


def test_forecast_errors_partial_window_allowed_when_not_strict():
    actual, forecast = _frames()
    out = forecast_errors(actual.iloc[:1], forecast, strict=False)
    assert len(out) == 1