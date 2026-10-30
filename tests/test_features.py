import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLS, LAG_DAYS, MIN_LAG, ROLLING_WINDOWS, add_features
from src.schema import DATE_COL, ITEM_COL, SALES_COL, STORE_COL


def _series(n_days=100, store=1, item=1):
    return pd.DataFrame(
        {
            DATE_COL: pd.date_range("2017-01-01", periods=n_days),
            STORE_COL: store,
            ITEM_COL: item,
            SALES_COL: np.arange(n_days, dtype=float),
        }
    )


def test_all_lags_and_rolling_shift_are_at_least_28_days():
    assert min(LAG_DAYS) >= 28
    assert MIN_LAG >= 28


def test_lag_and_rolling_values_hand_computed():
    f = add_features(_series(60))
    row = f.iloc[40]
    assert row["lag_28"] == 12                                            # sales at index 12
    assert row["rolling_mean_7"] == pytest.approx(np.mean(range(6, 13)))  # indices 6..12
    assert np.isnan(f.iloc[27]["lag_28"])                                 # nothing 28 days back yet


def test_perturbing_sales_after_t_minus_28_does_not_change_features_at_t():
    base = _series(100)
    t = 80
    changed = base.copy()
    changed.loc[t - 27:, SALES_COL] += 1000          # everything after t-28, incl. t and later
    fa, fb = add_features(base), add_features(changed)
    sales_features = [c for c in FEATURE_COLS if c.startswith(("lag_", "rolling_"))]
    pd.testing.assert_series_equal(fa.loc[t, sales_features], fb.loc[t, sales_features])


def test_perturbing_sales_at_t_minus_28_does_change_lag_28():
    base = _series(100)
    changed = base.copy()
    changed.loc[52, SALES_COL] += 1000               # t=80 -> t-28 = 52
    assert add_features(base).loc[80, "lag_28"] != add_features(changed).loc[80, "lag_28"]


def test_features_are_computed_per_series():
    two = pd.concat([_series(60, store=1), _series(60, store=2).assign(**{SALES_COL: 500.0})])
    f = add_features(two)
    store1_lag = f[f[STORE_COL] == 1].iloc[40]["lag_28"]
    assert store1_lag == 12                           # not contaminated by store 2


def test_calendar_features():
    f = add_features(_series(10))
    assert f.loc[0, "dayofweek"] == 6 and f.loc[0, "is_weekend"] == 1   # 2017-01-01 is a Sunday
    assert f.loc[0, "is_month_start"] == 1 and f.loc[0, "year"] == 2017
    assert set(ROLLING_WINDOWS) == {7, 14, 28}