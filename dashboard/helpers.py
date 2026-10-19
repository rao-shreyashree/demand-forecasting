"""Pure helpers for the dashboard (no streamlit imports -> unit-testable).

Inputs:
  forecasts : date, store, item, y_hat, model   (forecast contract; CSV written by the model branches)
  actuals   : date, store, item, sales          (src.data.load_data + clean_data on raw train.csv)
  plan      : output of build_inventory_plan (CSV written by feature/forecast-to-plan_shreya)
"""
from pathlib import Path

import pandas as pd

from src.data import clean_data, load_data
from src.metrics import bias, rmse, wape
from src.schema import (  # NOTE: if any of these live elsewhere (e.g. src.inventory), fix the import
    DATE_COL,
    EOQ_COL,
    FLAG_COL,
    ITEM_COL,
    MODEL_COL,
    ORDER_QTY_COL,
    REORDER_POINT_COL,
    SAFETY_STOCK_COL,
    SALES_COL,
    STOCK_ON_HAND_COL,
    STORE_COL,
    Y_HAT_COL,
)

RAW_PATH = Path("data/raw/train.csv")
PROCESSED = Path("data/processed")
FORECASTS_PATH = PROCESSED / "forecasts_classical.csv"  # TODO: swap to combined / best-model forecasts later
PLAN_PATH = PROCESSED / "inventory_plan.csv"            # written by forecast-to-plan branch

_KEYS = [STORE_COL, ITEM_COL]


def _read(path, parse_dates=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


def load_actuals(path=RAW_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return clean_data(load_data(str(path)))


def load_forecasts(path=FORECASTS_PATH) -> pd.DataFrame:
    return _read(path, parse_dates=[DATE_COL])


def load_plan(path=PLAN_PATH) -> pd.DataFrame:
    return _read(path)


def available_series(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Unique (store, item) pairs that have forecasts."""
    return forecasts[_KEYS].drop_duplicates().sort_values(_KEYS).reset_index(drop=True)


def forecast_vs_actual(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    store: int,
    item: int,
    models=None,
    history_days: int = 28,
) -> pd.DataFrame:
    """Wide frame indexed by date: `actual` + one column per model (y_hat).

    Includes `history_days` of actuals before the forecast window for context.
    """
    f = forecasts[(forecasts[STORE_COL] == store) & (forecasts[ITEM_COL] == item)]
    if models is not None:
        f = f[f[MODEL_COL].isin(models)]
    if f.empty:
        return pd.DataFrame(columns=["actual"])

    wide = f.pivot_table(index=DATE_COL, columns=MODEL_COL, values=Y_HAT_COL, aggfunc="mean")
    start = wide.index.min() - pd.Timedelta(days=history_days)

    a = actuals[(actuals[STORE_COL] == store) & (actuals[ITEM_COL] == item)]
    a = a[a[DATE_COL] >= start].set_index(DATE_COL)[SALES_COL].rename("actual")

    out = pd.concat([a, wide], axis=1).sort_index()
    out.index.name = DATE_COL
    return out


def trim_actuals(actuals: pd.DataFrame, forecasts: pd.DataFrame, days: int = 120) -> pd.DataFrame:
    """Keep only actuals from `days` before the first forecast date (the dashboard doesn't need the rest)."""
    start = forecasts[DATE_COL].min() - pd.Timedelta(days=days)
    return actuals[actuals[DATE_COL] >= start].reset_index(drop=True)


def _overlap(forecasts, actuals, store=None, item=None, models=None) -> pd.DataFrame:
    """Forecast rows joined with the actual sales on the same (store, item, date); adds `error` = y_hat - sales."""
    f = forecasts
    if store is not None:
        f = f[f[STORE_COL] == store]
    if item is not None:
        f = f[f[ITEM_COL] == item]
    if models is not None:
        f = f[f[MODEL_COL].isin(models)]
    m = f.merge(actuals[[*_KEYS, DATE_COL, SALES_COL]], on=[*_KEYS, DATE_COL], how="inner")
    m["error"] = m[Y_HAT_COL] - m[SALES_COL]
    return m


def forecast_metrics(forecasts, actuals, store=None, item=None, models=None, by=None) -> pd.DataFrame:
    """Accuracy per model on the forecast window (lowest WAPE first), using src.metrics.

    wape  : sum|y - y_hat| / sum|y|  (fraction)
    bias  : mean(y_hat - y) per row, in units/day (+ = over-forecast, - = under-forecast)
    rmse  : root mean squared error, in units
    """
    by = list(by or [])
    keys = [MODEL_COL, *by]
    out_cols = [*keys, "wape", "bias", "rmse", "actual_total", "forecast_total", "n"]
    m = _overlap(forecasts, actuals, store, item, models)
    if m.empty:
        return pd.DataFrame(columns=out_cols)
    rows = []
    for key, g in m.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(keys, key)),
            "wape": wape(g[SALES_COL], g[Y_HAT_COL]),
            "bias": bias(g[SALES_COL], g[Y_HAT_COL]),
            "rmse": rmse(g[SALES_COL], g[Y_HAT_COL]),
            "actual_total": float(g[SALES_COL].sum()),
            "forecast_total": float(g[Y_HAT_COL].sum()),
            "n": len(g),
        })
    return pd.DataFrame(rows)[out_cols].sort_values([*by, "wape"]).reset_index(drop=True)


def error_series(forecasts, actuals, store, item, models=None) -> pd.DataFrame:
    """Daily forecast error per model for one series, as y_hat - actual.

    Same sign as metrics.bias() (+ = over-forecast); opposite of metrics.forecast_errors().
    """
    m = _overlap(forecasts, actuals, store, item, models)
    return m[[DATE_COL, MODEL_COL, "error"]].sort_values([MODEL_COL, DATE_COL]).reset_index(drop=True)


def series_plan(plan: pd.DataFrame, store: int, item: int):
    """The plan row (as a Series) for one (store, item), or None."""
    r = plan[(plan[STORE_COL] == store) & (plan[ITEM_COL] == item)]
    return None if r.empty else r.iloc[0]


def plan_kpis(plan: pd.DataFrame) -> dict:
    to_order = plan[plan[ORDER_QTY_COL] > 0]
    return {
        "series": int(len(plan)),
        "to_order": int(len(to_order)),
        "order_units": float(to_order[ORDER_QTY_COL].sum()),
    }


def flag_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Number of series per stock flag (whatever flag values the plan contains)."""
    counts = plan[FLAG_COL].value_counts().sort_index()
    return counts.rename_axis(FLAG_COL).reset_index(name="series")


def flag_by_store(plan: pd.DataFrame) -> pd.DataFrame:
    return plan.groupby([STORE_COL, FLAG_COL]).size().reset_index(name="series")


def reorder_table(plan: pd.DataFrame, flags=None, stores=None) -> pd.DataFrame:
    """Series with a recommended order (> 0), most urgent (largest shortfall vs reorder point) first."""
    df = plan.copy()
    if flags is not None:
        df = df[df[FLAG_COL].isin(flags)]
    if stores is not None:
        df = df[df[STORE_COL].isin(stores)]
    df = df[df[ORDER_QTY_COL] > 0].copy()
    df["shortfall"] = df[REORDER_POINT_COL] - df[STOCK_ON_HAND_COL]
    df["stock_vs_rop_pct"] = (df[STOCK_ON_HAND_COL] / df[REORDER_POINT_COL].where(df[REORDER_POINT_COL] != 0) * 100)
    cols = [*_KEYS, STOCK_ON_HAND_COL, REORDER_POINT_COL, "stock_vs_rop_pct", SAFETY_STOCK_COL,
            "shortfall", EOQ_COL, ORDER_QTY_COL, FLAG_COL]
    return df[cols].sort_values("shortfall", ascending=False).reset_index(drop=True)