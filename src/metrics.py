"""Forecast accuracy metrics.

WAPE is the primary metric for this project. MAPE is reported only as a
secondary metric because it is undefined when actual sales are zero
(common with intermittent demand).

Sign conventions (do not mix them up):
  * forecast_errors(): error = actual - forecast  (positive = under-forecast)
  * bias():            mean(forecast - actual)    (positive = over-forecast)
  so mean(error column) == -bias() for the same data.
"""
import numpy as np
import pandas as pd

from src.schema import DATE_COL as DATE, STORE_COL as STORE
from src.schema import ITEM_COL as ITEM, SALES_COL as SALES

# TODO: swap for imports once schema.py defines Y_HAT_COL, MODEL_COL, ERROR_COL
Y_HAT = "y_hat"
MODEL = "model"
ERROR = "error"


def _prepare(y_true, y_pred):
    """Convert to 1-D float arrays and validate them."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D")
    if yt.size == 0 or yp.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if yt.shape != yp.shape:
        raise ValueError(
            f"Length mismatch: y_true has {yt.size}, y_pred has {yp.size}"
        )
    return yt, yp


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error = sum|y - y_hat| / sum|y|.

    Safe for zero-sales days (unlike MAPE). Returns NaN if total actual
    sales are zero, since the ratio is undefined.
    """
    yt, yp = _prepare(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(yt - yp).sum() / denom)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, as a fraction (0.2 = 20%).

    Rows where the actual value is 0 are SKIPPED because the percentage
    error is undefined there. Returns NaN if every actual is 0.
    Prefer WAPE for intermittent demand.
    """
    yt, yp = _prepare(y_true, y_pred)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    yt, yp = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def bias(y_true, y_pred):
    """Mean forecast error = mean(y_pred - y_true).

    Positive = over-forecasting (overstock risk),
    negative = under-forecasting (stockout risk).
    NOTE: opposite sign to the `error` column of forecast_errors()
    (which is actual - forecast).
    """
    yt, yp = _prepare(y_true, y_pred)
    return float(np.mean(yp - yt))


def forecast_errors(actual_df, forecast_df, strict=True):
    """Per-row forecast errors (actual - forecast) for the inventory module.

    actual_df   : columns date, store, item, sales (one row per key)
    forecast_df : columns date, store, item, y_hat, optionally model
    strict      : if True, raise when some forecast rows have no matching
                  actual (a partial validation window would skew sigma).

    Returns a DataFrame with columns date, store, item, [model,] error.
    If forecast_df has a `model` column it is kept, so callers can filter or
    group by model. Without a `model` column, duplicate (date, store, item)
    keys in forecast_df raise instead of silently duplicating rows.
    """
    keys = [DATE, STORE, ITEM]
    for name, df, cols in (
        ("actual_df", actual_df, keys + [SALES]),
        ("forecast_df", forecast_df, keys + [Y_HAT]),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    has_model = MODEL in forecast_df.columns
    fc_keys = keys + ([MODEL] if has_model else [])
    if actual_df.duplicated(keys).any():
        raise ValueError("actual_df has duplicate (date, store, item) rows")
    if forecast_df.duplicated(fc_keys).any():
        raise ValueError(
            "forecast_df has duplicate (date, store, item) rows; "
            "filter to one model or include a 'model' column"
        )

    fc_cols = fc_keys + [Y_HAT]
    merged = actual_df[keys + [SALES]].merge(
        forecast_df[fc_cols],
        on=keys,
        how="inner",
        validate="one_to_many" if has_model else "one_to_one",
    )
    if merged.empty:
        raise ValueError("No matching (date, store, item) rows to compare")
    if strict and len(merged) < len(forecast_df):
        raise ValueError(
            f"Only {len(merged)} of {len(forecast_df)} forecast rows have an "
            "actual value; the validation window is incomplete"
        )
    merged[ERROR] = merged[SALES] - merged[Y_HAT]
    return merged[fc_keys + [ERROR]].reset_index(drop=True)