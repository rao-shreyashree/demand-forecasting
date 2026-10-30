import numpy as np
import pandas as pd
import pytest

from src.inventory import simulate_inventory
from src.metrics import wape
from src.schema import (
    CHOSEN_MODEL_COL, DATE_COL, FLAG_COL, ITEM_COL, MODEL_COL, SALES_COL,
    SELECTED_MODEL, STORE_COL, WAPE_COL, Y_HAT_COL,
)
from src.selection import (
    apply_selection, build_selected_plan, combine_forecasts, plan_summary,
    score_models, select_best_model, top_risks,
)

DATES = pd.date_range("2017-12-04", periods=4)
KEYS = [STORE_COL, ITEM_COL]


def _actual(extra=None):
    data = {(1, 1): [10, 12, 14, 16], (1, 2): [20, 20, 20, 20]}
    data.update(extra or {})
    rows = [(d, s, i, v) for (s, i), vals in data.items() for d, v in zip(DATES, vals)]
    return pd.DataFrame(rows, columns=[DATE_COL, STORE_COL, ITEM_COL, SALES_COL])


def _fc(actual, model, offset, series=None):
    if series is None:
        mask = np.ones(len(actual), dtype=bool)
    else:
        mask = actual.set_index(KEYS).index.isin(series)
    f = actual[mask].copy()
    f[Y_HAT_COL] = f.pop(SALES_COL) + offset
    f[MODEL_COL] = model
    return f[[DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL]]


def test_picks_lowest_wape_per_series():
    a = _actual()
    fc = combine_forecasts(_fc(a, "good", 1), _fc(a, "bad", 5))
    best = select_best_model(score_models(a, fc))
    assert len(best) == 2
    assert (best[MODEL_COL] == "good").all()


def test_series_without_classical_fall_back_to_baseline():
    a = _actual()
    fc = combine_forecasts(
        _fc(a, "naive", 3),
        _fc(a, "sarima", 0, series=[(1, 1)]),
    )
    best = select_best_model(score_models(a, fc)).set_index(KEYS)[MODEL_COL]
    assert best[(1, 1)] == "sarima"
    assert best[(1, 2)] == "naive"


def test_score_matches_metrics_wape():
    a = _actual()
    f = _fc(a, "m", 0)
    f[Y_HAT_COL] = f[Y_HAT_COL] + np.tile([1, -2, 3, -4], 2)
    got = score_models(a, f).set_index(KEYS)[WAPE_COL]
    for s, i in [(1, 1), (1, 2)]:
        mask_a = (a[STORE_COL] == s) & (a[ITEM_COL] == i)
        mask_f = (f[STORE_COL] == s) & (f[ITEM_COL] == i)
        assert got[(s, i)] == pytest.approx(wape(a[mask_a][SALES_COL], f[mask_f][Y_HAT_COL]))


def test_apply_selection_one_model_no_duplicates():
    a = _actual()
    fc = combine_forecasts(_fc(a, "good", 1), _fc(a, "bad", 5))
    best = select_best_model(score_models(a, fc))
    sel = apply_selection(fc, best)
    assert (sel[MODEL_COL] == SELECTED_MODEL).all()
    assert not sel.duplicated([DATE_COL, STORE_COL, ITEM_COL]).any()
    assert len(sel) == 2 * len(DATES)
    assert list(sel.columns) == [DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL]


def test_combine_rejects_duplicate_rows():
    a = _actual()
    with pytest.raises(ValueError, match="duplicate"):
        combine_forecasts(_fc(a, "naive", 1), _fc(a, "naive", 2))


def test_combine_rejects_missing_columns():
    a = _actual()
    with pytest.raises(ValueError, match="missing columns"):
        combine_forecasts(_fc(a, "naive", 1).drop(columns=[MODEL_COL]))


def test_zero_actuals_do_not_crash_and_tie_break_by_name():
    a = _actual(extra={(3, 3): [0, 0, 0, 0]})
    fc = combine_forecasts(_fc(a, "b", 1), _fc(a, "a", 1))
    scores = score_models(a, fc)
    assert scores[scores[STORE_COL] == 3][WAPE_COL].isna().all()
    best = select_best_model(scores).set_index(KEYS)[MODEL_COL]
    assert best[(3, 3)] == "a"


def test_models_with_different_coverage_raise():
    a = _actual()
    short = _fc(a, "short", 1).groupby(KEYS).head(3)
    fc = combine_forecasts(_fc(a, "full", 1), short)
    with pytest.raises(ValueError, match="different numbers of days"):
        score_models(a, fc)


def test_incomplete_validation_window_raises():
    a = _actual()
    fc = _fc(a, "m", 1)
    with pytest.raises(ValueError, match="incomplete"):
        score_models(a.iloc[1:], fc)


def test_build_selected_plan_end_to_end():
    a = _actual()
    noisy = _fc(a, "noisy", 0)
    noisy[Y_HAT_COL] = noisy[Y_HAT_COL] + np.tile([2, -1, 3, -2], 2)
    fc = combine_forecasts(noisy, _fc(a, "sarima", 1, series=[(1, 1)]))
    inv = simulate_inventory(a)
    plan = build_selected_plan(a, fc, inv)

    assert len(plan) == 2
    assert not plan.duplicated(KEYS).any()
    assert {CHOSEN_MODEL_COL, WAPE_COL, FLAG_COL} <= set(plan.columns)
    chosen = plan.set_index(KEYS)[CHOSEN_MODEL_COL]
    assert chosen[(1, 1)] == "sarima"
    assert chosen[(1, 2)] == "noisy"
    assert plan[FLAG_COL].isin(["stockout_risk", "ok", "overstock"]).all()


def test_summary_and_top_risks():
    a = _actual()
    fc = combine_forecasts(_fc(a, "naive", 2), _fc(a, "prophet", 1, series=[(1, 1)]))
    plan = build_selected_plan(a, fc, simulate_inventory(a))
    summ = plan_summary(plan)
    assert summ.loc["All", "All"] == len(plan)
    assert len(top_risks(plan, n=1)) <= 1

def test_summary_counts_ml_winner_as_ml():
    a = _actual()
    fc = combine_forecasts(_fc(a, "naive", 3), _fc(a, "lightgbm", 0, series=[(1, 1)]))
    plan = build_selected_plan(a, fc, simulate_inventory(a))
    summ = plan_summary(plan)
    assert summ.loc["ml", "All"] == 1
    assert summ.loc["baseline", "All"] == 1
    assert "classical" not in summ.index