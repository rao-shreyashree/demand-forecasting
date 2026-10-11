import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')

from src.schema import DATE_COL, STORE_COL, ITEM_COL, SALES_COL
from src.metrics import wape, rmse, mape, bias


@pytest.fixture
def sample_df():
    """40 days of data, 2 stores x 2 items = 4 series."""
    base = datetime(2017, 1, 1)
    rows = []
    for store in [1, 2]:
        for item in [1, 2]:
            for d in range(40):
                rows.append({
                    DATE_COL:  base + timedelta(days=d),
                    STORE_COL: store,
                    ITEM_COL:  item,
                    SALES_COL: 10 + d % 7  # simple pattern
                })
    df = pd.DataFrame(rows)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)


def make_windows(df, horizon=28, n_windows=3):
    """Duplicate of rolling_origin_windows for test isolation."""
    all_dates = sorted(df[DATE_COL].unique())
    total_days = len(all_dates)
    windows = []
    for i in range(n_windows):
        val_end_idx   = total_days - 1 - (i * horizon)
        val_start_idx = val_end_idx - horizon + 1
        if val_start_idx < 0:
            break
        val_start = all_dates[val_start_idx]
        val_end   = all_dates[val_end_idx]
        train_df  = df[df[DATE_COL] < val_start].copy()
        val_df    = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)].copy()
        windows.append({'window': i+1, 'train': train_df, 'val': val_df})
    windows.reverse()
    return windows


def test_windows_no_leakage(sample_df):
    windows = make_windows(sample_df, horizon=7, n_windows=2)
    for w in windows:
        assert w['train'][DATE_COL].max() < w['val'][DATE_COL].min()


def test_windows_val_size(sample_df):
    windows = make_windows(sample_df, horizon=7, n_windows=2)
    for w in windows:
        unique_val_days = w['val'][DATE_COL].nunique()
        assert unique_val_days == 7


def test_evaluate_forecast_perfect():
    """Perfect forecast should give wape=0, rmse=0."""
    actual = pd.DataFrame({
        DATE_COL:  pd.date_range('2017-01-01', periods=5),
        STORE_COL: 1, ITEM_COL: 1, SALES_COL: [10, 12, 11, 13, 10]
    })
    forecast = actual.copy()
    forecast['y_hat'] = forecast[SALES_COL]
    forecast['model'] = 'perfect'

    assert wape(actual[SALES_COL], forecast['y_hat']) == 0.0
    assert rmse(actual[SALES_COL], forecast['y_hat']) == 0.0


def test_evaluate_forecast_ranking():
    """Better forecast should rank higher (lower WAPE)."""
    y_true = np.array([10.0] * 10)
    good   = np.array([10.5] * 10)  # small error
    bad    = np.array([15.0] * 10)  # large error

    assert wape(y_true, good) < wape(y_true, bad)


def test_wape_zero_actuals():
    """WAPE should return NaN when all actuals are zero."""
    assert np.isnan(wape([0, 0, 0], [1, 2, 3]))