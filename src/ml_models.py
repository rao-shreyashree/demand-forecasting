"""LightGBM demand forecast, one global model for all (store, item) series.

Direct strategy: every feature is lagged by >= 28 days, so ONE model predicts
any day of a <= 28-day horizon from information available at the cutoff. No
recursive forecasting is needed.

Leakage safety: forecast_lightgbm() receives TRAIN data only. It appends the
horizon's future dates (sales = NaN) and builds features on that frame, so a
future row can only see sales at t-28 or earlier, which are all in train.

TODO: the feature code below mirrors notebooks/eda_features_shryes.ipynb.
Replace add_features() with an import once Shryes moves it to src/features.py.
"""
import numpy as np
import pandas as pd

from src.schema import (
    DATE_COL,
    ITEM_COL,
    MODEL_COL,
    SALES_COL,
    STORE_COL,
    Y_HAT_COL,
    validate_schema,
)

MODEL_NAME = "lightgbm"
RANDOM_SEED = 42
LAG_DAYS = [28, 35, 42, 56, 91, 364]
ROLLING_WINDOWS = [7, 14, 28]
MIN_LAG = min(LAG_DAYS)  # largest horizon the features can support

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

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "verbosity": -1,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, shifted-rolling and calendar features per (store, item) series.

    Needs complete daily rows per series (shifts are by row). All lags are
    >= 28 days and rolling windows are shifted by 28, so a row at date t only
    uses sales up to t - 28.
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


def _check_train(train: pd.DataFrame, horizon: int) -> None:
    validate_schema(train)
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if horizon > MIN_LAG:
        raise ValueError(
            f"horizon {horizon} > {MIN_LAG}: features are lagged by {MIN_LAG} days, "
            "so a longer horizon would need values that don't exist yet"
        )
    if train[SALES_COL].isna().any():
        raise ValueError("train contains NaN sales; clean the data first")
    if train.duplicated(_KEYS + [DATE_COL]).any():
        raise ValueError("train has duplicate (store, item, date) rows")
    if train[DATE_COL].nunique() <= MIN_LAG:
        raise ValueError(f"need more than {MIN_LAG} days of history")

    last_dates = train.groupby(_KEYS)[DATE_COL].max()
    if last_dates.nunique() != 1:
        raise ValueError("all series must end on the same date")
    steps = train.sort_values(_KEYS + [DATE_COL]).groupby(_KEYS)[DATE_COL].diff().dropna()
    if (steps != pd.Timedelta(days=1)).any():
        raise ValueError("series have missing days; fill gaps before forecasting")


def _future_frame(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    series = train[_KEYS].drop_duplicates()
    start = train[DATE_COL].max() + pd.Timedelta(days=1)
    dates = pd.DataFrame({DATE_COL: pd.date_range(start, periods=horizon, freq="D")})
    future = series.merge(dates, how="cross")
    future[SALES_COL] = np.nan
    return future


def forecast_lightgbm(
    train: pd.DataFrame,
    horizon: int = 28,
    seed: int = RANDOM_SEED,
    params: dict | None = None,
) -> pd.DataFrame:
    """Train one LightGBM on `train` and forecast the next `horizon` days.

    train : date, store, item, sales (TRAIN data only, e.g. from time_split)
    Returns date, store, item, y_hat, model (clipped at 0), sorted by
    store, item, date.
    """
    import lightgbm as lgb

    _check_train(train, horizon)
    cols = [DATE_COL, STORE_COL, ITEM_COL, SALES_COL]
    cutoff = train[DATE_COL].max()
    full = pd.concat([train[cols], _future_frame(train, horizon)], ignore_index=True)
    feat = add_features(full)
    for c in _KEYS:
        feat[c] = feat[c].astype("category")

    is_future = feat[DATE_COL] > cutoff
    fit = feat[~is_future & feat["lag_28"].notna()]
    future = feat[is_future]

    model_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed, "deterministic": True,
                    "force_row_wise": True}
    model = lgb.LGBMRegressor(**model_params)
    model.fit(fit[FEATURE_COLS], fit[SALES_COL])

    out = future[[DATE_COL, STORE_COL, ITEM_COL]].copy()
    out[STORE_COL] = out[STORE_COL].astype(int)
    out[ITEM_COL] = out[ITEM_COL].astype(int)
    out[Y_HAT_COL] = np.clip(model.predict(future[FEATURE_COLS]), 0, None)
    out[MODEL_COL] = MODEL_NAME
    return out.sort_values(_KEYS + [DATE_COL]).reset_index(drop=True)