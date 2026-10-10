"""Per-series model selection and forecast-to-plan glue (Phase 2).

Candidates = baselines (all series) + classical models (top-N series).
Each (store, item) keeps the candidate with the lowest WAPE on the validation
window. Ties / undefined WAPE (all-zero actuals) are broken by model name, so
the result is deterministic.

build_inventory_plan() accepts ONE model, so the winners are stitched into a
single forecast labelled SELECTED_MODEL; the winning model name per series is
carried through as CHOSEN_MODEL_COL on the plan.
"""
import pandas as pd

from src.inventory import DEFAULT_SERVICE_LEVEL, STOCKOUT_RISK, build_inventory_plan
from src.metrics import forecast_errors
from src.schema import (
    CHOSEN_MODEL_COL,
    DATE_COL,
    FLAG_COL,
    ITEM_COL,
    MODEL_COL,
    ORDER_QTY_COL,
    SALES_COL,
    SELECTED_MODEL,
    STORE_COL,
    WAPE_COL,
    Y_HAT_COL,
    validate_schema,
)

CLASSICAL_MODELS = {"sarima", "prophet"}

_KEYS = [STORE_COL, ITEM_COL]
_ROW_KEYS = [DATE_COL, STORE_COL, ITEM_COL]
_FORECAST_COLS = [DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL]


def load_forecasts(path: str) -> pd.DataFrame:
    """Read a forecast CSV (date, store, item, y_hat, model) with correct dtypes."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    if list(df.columns) != _FORECAST_COLS:
        raise ValueError(f"{path}: expected columns {_FORECAST_COLS}, got {list(df.columns)}")
    return df


def combine_forecasts(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stack forecast tables (baselines, classical, later ML) into one candidate pool."""
    if not frames:
        raise ValueError("no forecast tables given")
    for i, f in enumerate(frames):
        missing = [c for c in _FORECAST_COLS if c not in f.columns]
        if missing:
            raise ValueError(f"forecast table {i} is missing columns: {missing}")
    out = pd.concat([f[_FORECAST_COLS] for f in frames], ignore_index=True)
    if out.duplicated(_ROW_KEYS + [MODEL_COL]).any():
        raise ValueError("duplicate (date, store, item, model) rows across forecast tables")
    return out.sort_values([STORE_COL, ITEM_COL, DATE_COL, MODEL_COL]).reset_index(drop=True)


def score_models(actual: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """WAPE per (store, item, model) on the validation window.

    Same formula as metrics.wape (sum|y - y_hat| / sum|y|), vectorised.
    WAPE is NaN where the series has zero actual sales.
    """
    validate_schema(actual)
    if actual.duplicated(_ROW_KEYS).any():
        raise ValueError("actual has duplicate (date, store, item) rows")
    m = forecasts.merge(actual[_ROW_KEYS + [SALES_COL]], on=_ROW_KEYS, how="left")
    if m[SALES_COL].isna().any():
        raise ValueError("some forecast rows have no actual; validation window is incomplete")
    m["abs_err"] = (m[SALES_COL] - m[Y_HAT_COL]).abs()
    m["abs_act"] = m[SALES_COL].abs()
    g = m.groupby(_KEYS + [MODEL_COL]).agg(
        num=("abs_err", "sum"), den=("abs_act", "sum"), days=(DATE_COL, "nunique")
    )
    if g["days"].nunique() != 1:
        raise ValueError("models cover different numbers of days; scores are not comparable")
    scores = g.reset_index()
    scores[WAPE_COL] = scores["num"] / scores["den"].where(scores["den"] != 0)
    return scores[_KEYS + [MODEL_COL, WAPE_COL]]


def select_best_model(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (store, item): lowest WAPE (NaN last, then model name)."""
    ranked = scores.sort_values(_KEYS + [WAPE_COL, MODEL_COL])
    return ranked.drop_duplicates(_KEYS, keep="first").reset_index(drop=True)


def apply_selection(forecasts: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Keep only each series' winning forecast, relabelled as one model."""
    sel = forecasts.merge(best[_KEYS + [MODEL_COL]], on=_KEYS + [MODEL_COL], how="inner")
    sel[MODEL_COL] = SELECTED_MODEL
    return sel.sort_values([STORE_COL, ITEM_COL, DATE_COL]).reset_index(drop=True)


def build_selected_plan(
    actual: pd.DataFrame,
    forecasts: pd.DataFrame,
    inventory: pd.DataFrame,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> pd.DataFrame:
    """Score candidates, pick per-series winners, build the inventory plan.

    actual    : validation slice from time_split (date, store, item, sales)
    forecasts : combine_forecasts(...) output (candidate pool)
    inventory : simulate_inventory() output

    Returns the standard plan plus CHOSEN_MODEL_COL and WAPE_COL.
    """
    scores = score_models(actual, forecasts)
    best = select_best_model(scores)
    sel = apply_selection(forecasts, best)
    errors = forecast_errors(actual, sel)
    plan = build_inventory_plan(sel, errors, inventory, service_level)
    chosen = best.rename(columns={MODEL_COL: CHOSEN_MODEL_COL})
    return plan.merge(chosen, on=_KEYS, how="left")


def plan_summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Flag counts split by whether the winner was a classical model or a baseline."""
    source = plan[CHOSEN_MODEL_COL].map(
        lambda m: "classical" if m in CLASSICAL_MODELS else "baseline"
    ).rename("source")
    return pd.crosstab(source, plan[FLAG_COL], margins=True)


def top_risks(plan: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n stockout-risk series with the largest recommended order."""
    return plan[plan[FLAG_COL] == STOCKOUT_RISK].nlargest(n, ORDER_QTY_COL).reset_index(drop=True)