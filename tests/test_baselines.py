import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    forecast_by_group,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_naive_repeats_last_value():
    assert naive_forecast([1, 2, 3], horizon=4).tolist() == [3, 3, 3, 3]


def test_moving_average_hand_calculated():
    # mean of last 2 values of [1, 2, 3, 5] = (3 + 5) / 2 = 4
    assert moving_average_forecast([1, 2, 3, 5], horizon=3, window=2).tolist() == [4, 4, 4]


def test_moving_average_window_equals_history():
    assert moving_average_forecast([2, 4], horizon=1, window=2).tolist() == [3]


def test_seasonal_naive_repeats_last_cycle():
    out = seasonal_naive_forecast([1, 2, 3, 4, 5, 6], horizon=5, season=3)
    assert out.tolist() == [4, 5, 6, 4, 5]


@pytest.mark.parametrize(
    "fn, kwargs",
    [
        (naive_forecast, {}),
        (seasonal_naive_forecast, {"season": 7}),
        (moving_average_forecast, {"window": 7}),
    ],
)
def test_empty_history_raises(fn, kwargs):
    with pytest.raises(ValueError):
        fn([], horizon=3, **kwargs)


def test_history_shorter_than_window_raises():
    with pytest.raises(ValueError):
        moving_average_forecast([1, 2], horizon=3, window=5)


def test_history_shorter_than_season_raises():
    with pytest.raises(ValueError):
        seasonal_naive_forecast([1, 2], horizon=3, season=7)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_bad_horizon_raises(bad):
    with pytest.raises(ValueError):
        naive_forecast([1, 2, 3], horizon=bad)


def _train_df():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    rows = []
    for store, item, vals in [(1, 1, [1, 2, 3]), (1, 2, [10, 20, 40])]:
        for d, v in zip(dates, vals):
            rows.append({"date": d, "store": store, "item": item, "sales": v})
    return pd.DataFrame(rows)


def test_forecast_by_group_contract_format():
    out = forecast_by_group(_train_df(), method="naive", horizon=4)
    assert list(out.columns) == ["date", "store", "item", "y_hat", "model"]
    assert len(out) == 2 * 4                       # 2 series x 4 days
    assert (out["model"] == "naive").all()


def test_forecast_by_group_values_and_dates():
    out = forecast_by_group(_train_df(), method="naive", horizon=2)
    s1 = out[(out["store"] == 1) & (out["item"] == 1)]
    s2 = out[(out["store"] == 1) & (out["item"] == 2)]
    assert s1["y_hat"].tolist() == [3, 3]
    assert s2["y_hat"].tolist() == [40, 40]
    # forecast starts the day AFTER the last training date (no leakage)
    assert s1["date"].min() == pd.Timestamp("2020-01-04")


def test_forecast_by_group_moving_average_name():
    out = forecast_by_group(_train_df(), method="moving_average", horizon=1, window=2)
    assert (out["model"] == "ma_2").all()
    s2 = out[out["item"] == 2]
    assert s2["y_hat"].iloc[0] == pytest.approx(30)   # (20 + 40) / 2


def test_forecast_by_group_unsorted_input():
    df = _train_df().sample(frac=1, random_state=0)
    out = forecast_by_group(df, method="naive", horizon=1)
    assert out[out["item"] == 1]["y_hat"].iloc[0] == 3


def test_forecast_by_group_unknown_method_raises():
    with pytest.raises(ValueError):
        forecast_by_group(_train_df(), method="magic")


def test_forecast_by_group_missing_column_raises():
    with pytest.raises(ValueError):
        forecast_by_group(_train_df().drop(columns="sales"), method="naive")