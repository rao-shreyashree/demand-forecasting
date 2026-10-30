"""LightGBM demand forecast, one global model for all (store, item) series.

Direct strategy: every feature is lagged by >= 28 days, so ONE model predicts
any day of a <= 28-day horizon from information available at the cutoff. No
recursive forecasting is needed.

Leakage safety: forecast_lightgbm() receives TRAIN data only. It appends the
horizon's future dates (sales = NaN) and builds features on that frame, so a
future row can only see sales at t-28 or earlier, which are all in train.

Features come from src/features.py (lag, shifted-rolling and calendar).
"""
import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, MIN_LAG, add_features
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
_KEYS = [STORE_COL, ITEM_COL]

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