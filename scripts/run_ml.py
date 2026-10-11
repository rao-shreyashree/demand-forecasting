"""Train the LightGBM forecaster on the TRAIN split and write its forecasts.

Run from the repo root:
    python -m scripts.run_ml
    python -m scripts.run_ml --data data/raw/train.csv --out data/processed/forecasts_ml.csv

Output columns: date, store, item, y_hat, model (the forecast contract).
"""
import argparse
from pathlib import Path

from src.data import clean_data, load_data, time_split
from src.ml_models import forecast_lightgbm

DEFAULT_DATA = "data/raw/train.csv"
DEFAULT_OUT = "data/processed/forecasts_ml.csv"


def run(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT, horizon=28, params=None):
    """Fit on train only (never the validation window) and save the forecast."""
    df = clean_data(load_data(data_path))
    train, _ = time_split(df, horizon=horizon)
    forecast = forecast_lightgbm(train, horizon=horizon, params=params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out_path, index=False)
    return forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    forecast = run(args.data, args.out)
    print(f"Saved {len(forecast)} rows to {args.out}")
    print(f"Dates: {forecast['date'].min().date()} to {forecast['date'].max().date()}")


if __name__ == "__main__":
    main()