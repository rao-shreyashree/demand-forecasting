"""End-to-end integration test (issue #63):
data -> clean -> split -> forecasts -> metrics -> model selection -> inventory plan.

Runs on a small synthetic dataset: fast, deterministic, CI-safe (no train.csv, no Prophet/SARIMA/LightGBM).
Those models have their own unit tests; here we check that the stages fit together.
"""
import numpy as np
import pandas as pd
import pytest

from src import data, inventory, selection
from src.baselines import forecast_by_group
from src.schema import CHOSEN_MODEL_COL, MODEL_COL, SELECTED_MODEL, WAPE_COL

HORIZON = 28
STORES = [1, 2]
ITEMS = [1, 2, 3]
N_DAYS = 420
METHODS = ["naive", "seasonal_naive", "moving_average"]  # method= values accepted by forecast_by_group
FORECAST_COLS = ["date", "store", "item", "y_hat", "model"]
N_SERIES = len(STORES) * len(ITEMS)


# synthetic data
@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", periods=N_DAYS, freq="D")
    frames = []
    for s in STORES:
        for i in ITEMS:
            base = 20 + 5 * s + 3 * i
            weekly = np.where(dates.day_of_week >= 5, 8, 0)
            sales = rng.poisson(base + weekly).astype(int)
            frames.append(pd.DataFrame({"date": dates, "store": s, "item": i, "sales": sales}))
    return pd.concat(frames, ignore_index=True)


# pipeline (built once)
@pytest.fixture(scope="module")
def pipeline(raw):
    df = data.clean_data(raw)
    train, val = data.time_split(df, HORIZON)

    frames = [forecast_by_group(train, method=m, horizon=HORIZON) for m in METHODS]
    fc = selection.combine_forecasts(*frames)

    scores = selection.score_models(val, fc)
    best = selection.select_best_model(scores)
    selected = selection.apply_selection(fc, best)
    return {"df": df, "train": train, "val": val, "fc": fc,
            "scores": scores, "best": best, "selected": selected}


# data + split
def test_clean_keeps_all_series(pipeline):
    df = pipeline["df"]
    assert df[["store", "item"]].drop_duplicates().shape[0] == N_SERIES
    assert not df.duplicated(["date", "store", "item"]).any()
    assert (df["sales"] >= 0).all()


def test_split_is_chronological_and_28_days(pipeline):
    train, val = pipeline["train"], pipeline["val"]
    assert train["date"].max() < val["date"].min()
    assert val["date"].nunique() == HORIZON


# forecasts
def test_forecast_contract_and_no_leakage(pipeline):
    fc, train = pipeline["fc"], pipeline["train"]
    assert list(fc.columns) == FORECAST_COLS
    assert not fc["y_hat"].isna().any()
    assert fc["date"].min() > train["date"].max()


def test_every_model_covers_every_series_for_full_horizon(pipeline):
    fc = pipeline["fc"]
    assert fc["model"].nunique() >= len(METHODS)
    for model, g in fc.groupby("model"):
        assert g[["store", "item"]].drop_duplicates().shape[0] == N_SERIES, model
        assert g.groupby(["store", "item"]).size().eq(HORIZON).all(), model


# metrics + selection
def test_scores_are_valid(pipeline):
    scores = pipeline["scores"]
    assert WAPE_COL in scores.columns
    assert np.isfinite(scores[WAPE_COL]).all()
    assert (scores[WAPE_COL] >= 0).all()


def test_one_best_model_per_series(pipeline):
    best, fc = pipeline["best"], pipeline["fc"]
    assert len(best) == N_SERIES
    assert not best.duplicated(["store", "item"]).any()
    assert set(best[MODEL_COL]) <= set(fc[MODEL_COL])


def test_best_model_is_never_worse_than_any_candidate(pipeline):
    scores, best = pipeline["scores"], pipeline["best"]
    merged = scores.merge(best, on=["store", "item"], suffixes=("_cand", "_best"))
    chosen = merged[merged[f"{MODEL_COL}_cand"] == merged[f"{MODEL_COL}_best"]]
    worst_allowed = merged.groupby(["store", "item"])[f"{WAPE_COL}_cand"].min()
    got = chosen.set_index(["store", "item"])[f"{WAPE_COL}_best"]
    assert (got <= worst_allowed + 1e-9).all()


def test_selected_forecast_is_complete(pipeline):
    sel = pipeline["selected"]
    assert list(sel.columns) == FORECAST_COLS
    assert set(sel["model"].unique()) == {SELECTED_MODEL}
    assert sel.groupby(["store", "item"]).size().eq(HORIZON).all()
    assert not sel["y_hat"].isna().any()


# inventory plan
def test_simulated_inventory_has_no_gaps(pipeline):
    inv = inventory.simulate_inventory(pipeline["df"])
    assert len(inv) > 0
    assert not inv.isna().any().any()
    # deterministic: same seed -> same output
    pd.testing.assert_frame_equal(inv, inventory.simulate_inventory(pipeline["df"]))


def _plan(pipeline):
    """ADAPT when there is a change: build_selected_plan's real arguments (see selection.py line 99)."""
    inv = inventory.simulate_inventory(pipeline["train"])
    return selection.build_selected_plan(pipeline["val"], pipeline["fc"], inv)


def test_inventory_plan_one_row_per_series(pipeline):
    plan = _plan(pipeline)
    assert len(plan) == N_SERIES
    assert not plan.duplicated(["store", "item"]).any()
    assert not plan[["store", "item"]].isna().any().any()
    # build_selected_plan renames MODEL_COL -> CHOSEN_MODEL_COL (selection.py),
    # so this is the real end-to-end chosen-model check.
    assert CHOSEN_MODEL_COL in plan.columns
    assert not plan[CHOSEN_MODEL_COL].isna().any()
    assert set(plan[CHOSEN_MODEL_COL]) <= set(pipeline["fc"][MODEL_COL])