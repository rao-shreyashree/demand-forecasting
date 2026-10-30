import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, ".")

from src.schema import DATE_COL, STORE_COL, ITEM_COL, SALES_COL
from src.metrics import wape, rmse, mape, bias
from src.backtest import rolling_origin_windows, evaluate_forecast, run_backtest
from src.data import time_split


@pytest.fixture
def sample_df():
    base = datetime(2017, 1, 1)
    rows = []
    for store in [1, 2]:
        for item in [1, 2]:
            for d in range(40):
                rows.append({
                    DATE_COL:  base + timedelta(days=d),
                    STORE_COL: store,
                    ITEM_COL:  item,
                    SALES_COL: 10 + d % 7
                })
    df = pd.DataFrame(rows)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)


def test_windows_no_leakage(sample_df):
    windows = rolling_origin_windows(sample_df, horizon=7, n_windows=2)
    for w in windows:
        assert w["train"][DATE_COL].max() < w["val"][DATE_COL].min()


def test_windows_val_size(sample_df):
    windows = rolling_origin_windows(sample_df, horizon=7, n_windows=2)
    for w in windows:
        assert w["val"][DATE_COL].nunique() == 7


def test_last_window_matches_time_split(sample_df):
    horizon = 7
    windows = rolling_origin_windows(sample_df, horizon=horizon, n_windows=2)
    last_window_val = windows[-1]["val"]
    _, ts_val = time_split(sample_df, horizon=horizon)
    assert last_window_val[DATE_COL].min() == ts_val[DATE_COL].min()
    assert last_window_val[DATE_COL].max() == ts_val[DATE_COL].max()


def test_evaluate_forecast_perfect(sample_df):
    windows = rolling_origin_windows(sample_df, horizon=7, n_windows=1)
    val_df = windows[0]["val"]
    perfect_fc = val_df[[DATE_COL, STORE_COL, ITEM_COL, SALES_COL]].copy()
    perfect_fc["y_hat"] = perfect_fc[SALES_COL].astype(float)
    perfect_fc["model"] = "perfect"
    result = evaluate_forecast(perfect_fc, val_df, window_num=1)
    assert result.loc[result["model"] == "perfect", "wape"].values[0] == 0.0
    assert result.loc[result["model"] == "perfect", "rmse"].values[0] == 0.0


def test_evaluate_forecast_hand_computed(sample_df):
    windows = rolling_origin_windows(sample_df, horizon=7, n_windows=1)
    val_df = windows[0]["val"]
    fc = val_df[[DATE_COL, STORE_COL, ITEM_COL, SALES_COL]].copy()
    fc["y_hat"] = fc[SALES_COL] + 1.0
    fc["model"] = "overforecast"
    result = evaluate_forecast(fc, val_df, window_num=1)
    computed_wape = result.loc[result["model"] == "overforecast", "wape"].values[0]
    n = len(val_df)
    expected_wape = n / val_df[SALES_COL].abs().sum()
    assert abs(computed_wape - expected_wape) < 1e-6


def test_wape_zero_actuals():
    assert np.isnan(wape([0, 0, 0], [1, 2, 3]))


def test_evaluate_forecast_n_rows(sample_df):
    windows = rolling_origin_windows(sample_df, horizon=7, n_windows=1)
    val_df = windows[0]["val"]
    fc = val_df[[DATE_COL, STORE_COL, ITEM_COL, SALES_COL]].copy()
    fc["y_hat"] = fc[SALES_COL].astype(float)
    fc["model"] = "test"
    result = evaluate_forecast(fc, val_df, window_num=1)
    assert result.loc[result["model"] == "test", "n_rows"].values[0] == len(val_df)
