import pandas as pd
import numpy as np
from src.schema import DATE_COL, STORE_COL, ITEM_COL, SALES_COL
from src.metrics import wape, rmse, mape, bias


def rolling_origin_windows(df, horizon=28, n_windows=3):
    """
    Generates n_windows (train, val) splits rolling forward.
    Window 1 = earliest val period, window n = latest (most recent).
    """
    all_dates = sorted(df[DATE_COL].unique())
    total_days = len(all_dates)

    windows = []
    for i in range(n_windows):
        val_end_idx   = total_days - 1 - (i * horizon)
        val_start_idx = val_end_idx - horizon + 1
        if val_start_idx < 0:
            break
        val_end   = all_dates[val_end_idx]
        val_start = all_dates[val_start_idx]

        train_df = df[df[DATE_COL] < val_start].copy()
        val_df   = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] <= val_end)].copy()

        windows.append({
            'window':    i + 1,
            'train':     train_df,
            'val':       val_df,
            'val_start': val_start,
            'val_end':   val_end
        })

    # Return in chronological order (window 1 = earliest val period)
    windows.reverse()
    return windows


def evaluate_forecast(forecast_df, val_df, window_num):
    """
    Scores forecast_df against val_df per model.
    Only rows with matching (date, store, item) keys are scored.
    Returns one row per model with wape, rmse, mape, bias, n_rows.
    """
    merged = val_df[[DATE_COL, STORE_COL, ITEM_COL, SALES_COL]].merge(
        forecast_df[[DATE_COL, STORE_COL, ITEM_COL, 'y_hat', 'model']],
        on=[DATE_COL, STORE_COL, ITEM_COL],
        how='inner'
    )

    results = []
    for model_name, group in merged.groupby('model'):
        results.append({
            'window': window_num,
            'model':  model_name,
            'wape':   wape(group[SALES_COL], group['y_hat']),
            'rmse':   rmse(group[SALES_COL], group['y_hat']),
            'mape':   mape(group[SALES_COL], group['y_hat']),
            'bias':   bias(group[SALES_COL], group['y_hat']),
            'n_rows': len(group)
        })
    return pd.DataFrame(results)


def run_backtest(forecast_fn, df, n_windows=3, horizon=28):
    """
    Runs forecast_fn on each rolling window and aggregates metrics.
    forecast_fn(train_df, val_df) -> forecast_df with columns:
        date, store, item, y_hat, model
    """
    windows = rolling_origin_windows(df, horizon=horizon, n_windows=n_windows)
    all_results = []

    for w in windows:
        forecast_df = forecast_fn(w['train'], w['val'])
        result      = evaluate_forecast(forecast_df, w['val'], w['window'])
        all_results.append(result)

    all_results = pd.concat(all_results, ignore_index=True)

    # Average metrics across windows, report how many windows each model appeared in
    comparison = (
        all_results
        .groupby('model')
        .agg(
            wape=('wape', 'mean'),
            rmse=('rmse', 'mean'),
            mape=('mape', 'mean'),
            bias=('bias', 'mean'),
            n_rows=('n_rows', 'sum'),
            n_windows=('window', 'count')
        )
        .round(4)
        .sort_values('wape')
        .reset_index()
    )
    comparison['rank'] = range(1, len(comparison) + 1)

    return comparison, all_results