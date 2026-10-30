"""Phase 2 - classical baselines (SARIMA + Prophet) on a fixed top-N subset.

Output schema (strict order): [date, store, item, y_hat, model]

Run:
    python -m src.classical_models --top-n 10                      # last 28 days only
    python -m src.classical_models --top-n 10 --backtest-windows 3 # 3 rolling origins
"""
import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from src.data import time_split          # mandatory: leakage-safe split
from src.metrics import wape, rmse       # reuse shared metrics

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

DATE_COL, STORE_COL, ITEM_COL, TARGET_COL = "date", "store", "item", "sales"
OUT_COLS = ["date", "store", "item", "y_hat", "model"]
HORIZON = 28


def select_top_series(df: pd.DataFrame, top_n: int) -> list[tuple]:
    """Fixed subset: the top-N (store, item) series by total sales (deterministic)."""
    totals = (
        df.groupby([STORE_COL, ITEM_COL])[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return list(totals.index)


def _format(dates, y_hat, store, item, model) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "store": store,
            "item": item,
            "y_hat": pd.Series(y_hat).clip(lower=0).values,  # sales can't be negative
            "model": model,
        }
    )
    return out[OUT_COLS]


def forecast_sarima(train_s: pd.Series, test_dates, store, item) -> pd.DataFrame:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train_s.asfreq("D").interpolate()
    fit = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),  # weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    pred = fit.forecast(steps=len(test_dates))
    return _format(test_dates, pred.values, store, item, "sarima")


def forecast_prophet(train_s: pd.Series, test_dates, store, item) -> pd.DataFrame:
    from prophet import Prophet

    d = train_s.reset_index()
    d.columns = ["ds", "y"]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(d)
    pred = m.predict(pd.DataFrame({"ds": test_dates}))
    return _format(test_dates, pred["yhat"].values, store, item, "prophet")


def _forecast_all(train: pd.DataFrame, val: pd.DataFrame, series: list[tuple]) -> list[pd.DataFrame]:
    """Fit SARIMA + Prophet per series on `train`, forecast the dates present in `val`."""
    preds = []
    for store, item in series:
        tr = (
            train[(train[STORE_COL] == store) & (train[ITEM_COL] == item)]
            .sort_values(DATE_COL)
            .set_index(DATE_COL)[TARGET_COL]
        )
        te_dates = (
            val[(val[STORE_COL] == store) & (val[ITEM_COL] == item)]
            .sort_values(DATE_COL)[DATE_COL]
            .values
        )
        for fn in (forecast_sarima, forecast_prophet):
            try:
                preds.append(fn(tr, te_dates, store, item))
            except Exception as e:  # one bad series shouldn't kill the run
                print(f"  ! {fn.__name__} failed for store={store} item={item}: {e}")
    return preds


def window_bounds(df: pd.DataFrame, horizon: int = HORIZON, n_windows: int = 3) -> list[tuple]:
    """(val_start, val_end) per rolling-origin window, chronological order.

    Same arithmetic as the backtest: non-overlapping `horizon`-day blocks counted
    back from the last date. The most recent window equals time_split(df, horizon)'s
    validation period.
    """
    all_dates = sorted(df[DATE_COL].unique())
    total = len(all_dates)
    bounds = []
    for i in range(n_windows):
        end_idx = total - 1 - i * horizon
        start_idx = end_idx - horizon + 1
        if start_idx < 0:
            break
        bounds.append((pd.Timestamp(all_dates[start_idx]), pd.Timestamp(all_dates[end_idx])))
    return bounds[::-1]


def run_backtest_windows(data_path: str, top_n: int, n_windows: int, out_path: str) -> pd.DataFrame:
    """Refit SARIMA + Prophet at every rolling origin (train strictly before val_start).

    The top-N series are picked once from data before the earliest window (no validation
    data is used for selection) and held fixed across windows. Output keeps the 5-column
    schema; windows are identified by their dates.
    """
    df = pd.read_csv(data_path, parse_dates=[DATE_COL])
    bounds = window_bounds(df, HORIZON, n_windows)

    # Leakage-safe, fixed cohort: rank series using only data before the EARLIEST
    # window's val_start, so no window's validation period influences which series
    # are evaluated, and the same series are used in every window.
    series = select_top_series(df[df[DATE_COL] < bounds[0][0]], top_n)

    parts = []
    for val_start, val_end in bounds:
        train = df[df[DATE_COL] < val_start]
        val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)]
        print(f"Window {val_start.date()} -> {val_end.date()}: {len(series)} series x 2 models")
        parts.extend(_forecast_all(train, val, series))

    forecasts = pd.concat(parts, ignore_index=True)[OUT_COLS]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_path, index=False)

    merged = forecasts.merge(
        df.rename(columns={TARGET_COL: "y_true"})[[DATE_COL, STORE_COL, ITEM_COL, "y_true"]],
        on=[DATE_COL, STORE_COL, ITEM_COL],
    )
    for model, g in merged.groupby("model"):
        print(f"{model:8s} WAPE={wape(g['y_true'], g['y_hat']):.4f}  RMSE={rmse(g['y_true'], g['y_hat']):.4f}  rows={len(g)}")
    print(f"Saved {len(forecasts)} rows -> {out_path}")
    return forecasts


def run(data_path: str, top_n: int, out_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, parse_dates=[DATE_COL])
    train, test = time_split(df, horizon=HORIZON)

    series = select_top_series(train, top_n)
    print(f"Forecasting {len(series)} series x 2 models, horizon={HORIZON}")

    preds = _forecast_all(train, test, series)

    forecasts = pd.concat(preds, ignore_index=True)[OUT_COLS]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_path, index=False)

    # quick sanity metrics against the held-out 28 days
    merged = forecasts.merge(
        test.rename(columns={TARGET_COL: "y_true"})[[DATE_COL, STORE_COL, ITEM_COL, "y_true"]],
        on=[DATE_COL, STORE_COL, ITEM_COL],
    )
    for model, g in merged.groupby("model"):
        print(f"{model:8s} WAPE={wape(g['y_true'], g['y_hat']):.4f}  RMSE={rmse(g['y_true'], g['y_hat']):.4f}")

    print(f"Saved {len(forecasts)} rows -> {out_path}")
    return forecasts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/train.csv")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out", default="data/processed/forecasts_classical.csv")
    ap.add_argument("--backtest-windows", type=int, default=0,
                    help="if > 0, refit at this many rolling origins and write --backtest-out")
    ap.add_argument("--backtest-out", default="data/processed/forecasts_classical_backtest.csv")
    a = ap.parse_args()
    if a.backtest_windows > 0:
        run_backtest_windows(a.data, a.top_n, a.backtest_windows, a.backtest_out)
    else:
        run(a.data, a.top_n, a.out)