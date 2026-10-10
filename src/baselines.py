"""Baseline demand forecasts: naive, seasonal naive, moving average.

All functions only look at the history passed in, so there is no leakage as
long as the caller passes TRAIN data only (e.g. the train part returned by
time_split).
"""
import numpy as np
import pandas as pd

from src.schema import DATE_COL as DATE, STORE_COL as STORE
from src.schema import ITEM_COL as ITEM, SALES_COL as SALES
from src.schema import validate_schema

# TODO: swap for imports once schema.py defines Y_HAT_COL and MODEL_COL
Y_HAT = "y_hat"
MODEL = "model"


def _check(history, horizon):
    h = np.asarray(history, dtype=float)
    if h.ndim != 1:
        raise ValueError("history must be 1-D")
    if h.size == 0:
        raise ValueError("history must not be empty")
    if np.isnan(h).any():
        raise ValueError("history contains NaN; clean the data first")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return h


def naive_forecast(history, horizon=28):
    """Repeat the last observed value for `horizon` steps."""
    h = _check(history, horizon)
    return np.full(horizon, h[-1])


def seasonal_naive_forecast(history, horizon=28, season=7):
    """Repeat the last `season` values (default: weekly pattern)."""
    h = _check(history, horizon)
    if season < 1:
        raise ValueError("season must be >= 1")
    if h.size < season:
        raise ValueError(f"history ({h.size}) shorter than season ({season})")
    last_cycle = h[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_cycle, reps)[:horizon]


def moving_average_forecast(history, horizon=28, window=7):
    """Mean of the last `window` values, repeated for `horizon` steps."""
    h = _check(history, horizon)
    if window < 1:
        raise ValueError("window must be >= 1")
    if h.size < window:
        raise ValueError(f"history ({h.size}) shorter than window ({window})")
    return np.full(horizon, h[-window:].mean())


def forecast_by_group(train_df, method="naive", horizon=28, **kwargs):
    """Run a baseline for every (store, item) and return the contract format.

    train_df : columns date, store, item, sales (TRAIN data only); checked
               with schema.validate_schema
    method   : "naive" | "seasonal_naive" | "moving_average"
    kwargs   : season=... or window=...

    Returns a DataFrame with columns date, store, item, y_hat, model
    (one model per call). Forecast dates start the day after each series'
    last date.
    """
    methods = {
        "naive": (naive_forecast, lambda kw: "naive"),
        "seasonal_naive": (
            seasonal_naive_forecast,
            lambda kw: f"seasonal_naive_{kw.get('season', 7)}",
        ),
        "moving_average": (
            moving_average_forecast,
            lambda kw: f"ma_{kw.get('window', 7)}",
        ),
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(methods)}")
    func, namer = methods[method]
    model_name = namer(kwargs)

    validate_schema(train_df)

    parts = []
    for (store, item), g in train_df.groupby([STORE, ITEM]):
        g = g.sort_values(DATE)
        y_hat = func(g[SALES].to_numpy(), horizon=horizon, **kwargs)
        start = pd.Timestamp(g[DATE].max()) + pd.Timedelta(days=1)
        parts.append(
            pd.DataFrame(
                {
                    DATE: pd.date_range(start, periods=horizon, freq="D"),
                    STORE: store,
                    ITEM: item,
                    Y_HAT: y_hat,
                    MODEL: model_name,
                }
            )
        )
    if not parts:
        raise ValueError("train_df is empty")
    return pd.concat(parts, ignore_index=True)