"""Lag, shifted-rolling and calendar features for demand forecasting.

Every sales-derived feature is lagged by >= MIN_LAG (28) days, so a row at
date t only uses sales up to t - 28. That lets one model forecast any day of
a <= 28-day horizon from information available at the cutoff (direct
strategy), with no recursive forecasting and no leakage.

Single implementation: the EDA notebook should import add_features from here
instead of redefining the feature code.
"""
import pandas as pd

from src.schema import DATE_COL, ITEM_COL, SALES_COL, STORE_COL

LAG_DAYS = [28, 35, 42, 56, 91, 364]
ROLLING_WINDOWS = [7, 14, 28]
MIN_LAG = min(LAG_DAYS)  # largest forecast horizon these features support

_KEYS = [STORE_COL, ITEM_COL]
CALENDAR_COLS = [
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear",
    "quarter", "is_weekend", "is_month_start", "is_month_end",
]
FEATURE_COLS = (
    _KEYS
    + [f"lag_{lag}" for lag in LAG_DAYS]
    + [f"rolling_{stat}_{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")]
    + CALENDAR_COLS
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, shifted-rolling and calendar columns per (store, item) series.

    Needs complete daily rows per series (shifts are by row). Rolling windows
    are computed on sales shifted by MIN_LAG, e.g. rolling_mean_7 at t is the
    mean of days [t-34 .. t-28].
    """
    out = df.sort_values(_KEYS + [DATE_COL]).reset_index(drop=True)
    sales_by_series = out.groupby(_KEYS)[SALES_COL]

    for lag in LAG_DAYS:
        out[f"lag_{lag}"] = sales_by_series.shift(lag)

    shifted = sales_by_series.shift(MIN_LAG)
    shifted_by_series = shifted.groupby([out[STORE_COL], out[ITEM_COL]])
    for w in ROLLING_WINDOWS:
        out[f"rolling_mean_{w}"] = shifted_by_series.transform(
            lambda x, w=w: x.rolling(w, min_periods=1).mean()
        )
        out[f"rolling_std_{w}"] = shifted_by_series.transform(
            lambda x, w=w: x.rolling(w, min_periods=1).std()
        )

    d = out[DATE_COL]
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["dayofweek"] = d.dt.dayofweek
    out["dayofyear"] = d.dt.dayofyear
    out["weekofyear"] = d.dt.isocalendar().week.astype(int)
    out["quarter"] = d.dt.quarter
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    return out