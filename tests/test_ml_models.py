import numpy as np
import pandas as pd
import pytest

from src.baselines import forecast_by_group
from src.data import time_split
from src.metrics import wape
from src.ml_models import MODEL_NAME, add_features, forecast_lightgbm
from src.schema import DATE_COL, ITEM_COL, MODEL_COL, SALES_COL, STORE_COL, Y_HAT_COL

FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5, "learning_rate": 0.2}
HORIZON = 28


def _data(n_days=200, series=((1, 1), (1, 2), (2, 1)), weekly=True):
    """Deterministic weekly pattern: Saturdays/Sundays sell more, per-series level."""
    dates = pd.date_range("2017-01-01", periods=n_days)
    rows = []
    for k, (s, i) in enumerate(series):
        for d in dates:
            base = 10 + 5 * k
            rows.append((d, s, i, base + (20 if weekly and d.dayofweek >= 5 else 0)))
    return pd.DataFrame(rows, columns=[DATE_COL, STORE_COL, ITEM_COL, SALES_COL])


def test_output_contract():
    train, _ = time_split(_data(), horizon=HORIZON)
    out = forecast_lightgbm(train, HORIZON, params=FAST)
    assert list(out.columns) == [DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL]
    assert len(out) == 3 * HORIZON
    assert (out[MODEL_COL] == MODEL_NAME).all()
    assert out[Y_HAT_COL].notna().all()
    assert (out[Y_HAT_COL] >= 0).all()
    assert pd.api.types.is_integer_dtype(out[STORE_COL])
    assert pd.api.types.is_integer_dtype(out[ITEM_COL])


def test_forecast_dates_start_after_train_end():
    train, valid = time_split(_data(), horizon=HORIZON)
    out = forecast_lightgbm(train, HORIZON, params=FAST)
    assert out[DATE_COL].min() == train[DATE_COL].max() + pd.Timedelta(days=1)
    assert out[DATE_COL].max() == valid[DATE_COL].max()


def test_same_seed_same_output():
    train, _ = time_split(_data(), horizon=HORIZON)
    a = forecast_lightgbm(train, HORIZON, seed=7, params=FAST)
    b = forecast_lightgbm(train, HORIZON, seed=7, params=FAST)
    pd.testing.assert_frame_equal(a, b)


def test_learns_weekly_pattern_better_than_naive():
    train, valid = time_split(_data(), horizon=HORIZON)
    lgb_fc = forecast_lightgbm(train, HORIZON, params=FAST)
    naive = forecast_by_group(train, "naive", HORIZON)
    keys = [DATE_COL, STORE_COL, ITEM_COL]
    m_lgb = valid.merge(lgb_fc, on=keys)
    m_naive = valid.merge(naive, on=keys)
    assert wape(m_lgb[SALES_COL], m_lgb[Y_HAT_COL]) < wape(m_naive[SALES_COL], m_naive[Y_HAT_COL])
    assert wape(m_lgb[SALES_COL], m_lgb[Y_HAT_COL]) < 0.1


def test_features_use_only_data_from_28_days_back():
    df = _data(n_days=60, series=((1, 1),))
    df[SALES_COL] = np.arange(60, dtype=float)
    f = add_features(df)
    row = f.iloc[40]
    assert row["lag_28"] == 12                              # sales at index 40 - 28
    assert row["rolling_mean_7"] == pytest.approx(np.mean(range(6, 13)))  # idx 6..12
    assert np.isnan(f.iloc[27]["lag_28"])                   # nothing 28 days back yet


def test_future_rows_never_see_values_after_the_cutoff():
    train, _ = time_split(_data(n_days=100, series=((1, 1),)), horizon=HORIZON)
    cutoff_sales = train[SALES_COL].to_numpy()
    out = forecast_lightgbm(train, HORIZON, params=FAST)
    # h=1 row uses lag_28 = sales at (cutoff - 27); h=28 uses sales at the cutoff
    full = pd.concat([train, pd.DataFrame({DATE_COL: out[DATE_COL], STORE_COL: 1, ITEM_COL: 1,
                                           SALES_COL: np.nan})], ignore_index=True)
    f = add_features(full)
    assert f.iloc[len(train)]["lag_28"] == cutoff_sales[-27]
    assert f.iloc[len(train) + 27]["lag_28"] == cutoff_sales[-1]


@pytest.mark.parametrize("bad", [0, -1, 2.5, 29])
def test_bad_horizon_raises(bad):
    train, _ = time_split(_data(), horizon=HORIZON)
    with pytest.raises(ValueError):
        forecast_lightgbm(train, bad, params=FAST)


def test_missing_days_raise():
    train, _ = time_split(_data(), horizon=HORIZON)
    gappy = train.drop(index=[10, 11, 12]).reset_index(drop=True)
    with pytest.raises(ValueError, match="missing days"):
        forecast_lightgbm(gappy, HORIZON, params=FAST)


def test_series_ending_on_different_dates_raise():
    train, _ = time_split(_data(), horizon=HORIZON)
    last = train.groupby([STORE_COL, ITEM_COL])[DATE_COL].transform("max")
    uneven = train[~((train[STORE_COL] == 2) & (train[DATE_COL] == last))]
    with pytest.raises(ValueError, match="same date"):
        forecast_lightgbm(uneven, HORIZON, params=FAST)


def test_too_little_history_raises():
    train = _data(n_days=20)
    with pytest.raises(ValueError, match="history"):
        forecast_lightgbm(train, 5, params=FAST)


def test_missing_column_raises():
    train, _ = time_split(_data(), horizon=HORIZON)
    with pytest.raises(ValueError):
        forecast_lightgbm(train.drop(columns=[SALES_COL]), HORIZON, params=FAST)
        