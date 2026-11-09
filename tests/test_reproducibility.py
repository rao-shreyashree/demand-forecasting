"""DF-NF-003: the same input and seeds must give identical outputs, run after run."""
import numpy as np
import pandas as pd
import pytest

from src.baselines import forecast_by_group
from src.data import time_split
from src.inventory import simulate_inventory
from src.ml_models import forecast_lightgbm
from src.schema import DATE_COL, ITEM_COL, SALES_COL, STORE_COL
from src.selection import build_selected_plan, combine_forecasts

FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5, "learning_rate": 0.2}


def _sales():
    """Small deterministic dataset: 3 series, weekly pattern plus fixed-seed noise."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2017-01-01", periods=200)
    rows = []
    for store, item in [(1, 1), (1, 2), (2, 1)]:
        for d in dates:
            level = 10 + 5 * store + item + (15 if d.dayofweek >= 5 else 0)
            rows.append((d, store, item, int(level + rng.integers(-3, 4))))
    return pd.DataFrame(rows, columns=[DATE_COL, STORE_COL, ITEM_COL, SALES_COL])


def _pipeline():
    """data -> baselines + LightGBM -> model selection -> inventory plan."""
    train, valid = time_split(_sales(), horizon=28)
    pool = combine_forecasts(
        forecast_by_group(train, "naive"),
        forecast_by_group(train, "moving_average"),
        forecast_lightgbm(train, horizon=28, params=FAST),
    )
    return build_selected_plan(valid, pool, simulate_inventory(train))


def test_full_pipeline_twice_gives_identical_plan():
    pd.testing.assert_frame_equal(_pipeline(), _pipeline())


def test_pipeline_ignores_global_numpy_random_state():
    np.random.seed(1)
    first = _pipeline()
    np.random.seed(999)
    second = _pipeline()
    pd.testing.assert_frame_equal(first, second)


def test_baselines_are_deterministic():
    train, _ = time_split(_sales(), horizon=28)
    for method in ("naive", "seasonal_naive", "moving_average"):
        pd.testing.assert_frame_equal(
            forecast_by_group(train, method), forecast_by_group(train, method)
        )


def test_lightgbm_same_seed_same_forecast():
    train, _ = time_split(_sales(), horizon=28)
    a = forecast_lightgbm(train, 28, seed=7, params=FAST)
    b = forecast_lightgbm(train, 28, seed=7, params=FAST)
    pd.testing.assert_frame_equal(a, b)


def test_simulated_inventory_same_seed_same_table_and_seed_matters():
    train, _ = time_split(_sales(), horizon=28)
    pd.testing.assert_frame_equal(simulate_inventory(train, seed=5), simulate_inventory(train, seed=5))
    assert not simulate_inventory(train, seed=5).equals(simulate_inventory(train, seed=6))