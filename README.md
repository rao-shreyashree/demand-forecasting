# Demand Forecasting & Inventory Planning System

## Dataset
Kaggle Store Item Demand Forecasting — 10 stores x 50 items, 5 years daily sales.
Download train.csv from https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
Place at data/raw/train.csv.

## Forecast Output Contract
Every model CSV must have exactly these columns in this order:
date, store, item, y_hat, model

Share your forecast CSV with Shryes directly (WhatsApp/Drive).

## Metrics
Primary: WAPE | Secondary: RMSE, MAPE, Bias
Validation: rolling-origin backtest, 3 windows of 28 days each.
Scores are averaged across windows. n_windows and n_rows are reported
so readers know how many windows each model was evaluated on.

## Leakage rule
Lag features must be >= 28 days. Rolling features must be shifted by 28
before the window. See notebooks/eda_features_shryes.ipynb.
