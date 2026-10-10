"""Inventory policy: safety stock, reorder point, EOQ, stock flags,
simulated inventory, and a DataFrame wrapper that builds the plan.

Rules (single source of truth, flag and order agree):
    stock <= ROP          -> stockout_risk, order up to ROP + EOQ
    stock >  ROP + EOQ    -> overstock
    otherwise             -> ok
"""
from math import ceil, isfinite, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.schema import (
    EOQ_COL,
    ERROR_COL,
    FLAG_COL,
    HOLDING_RATE_COL,
    ITEM_COL,
    LEAD_TIME_COL,
    MODEL_COL,
    ORDER_COST_COL,
    ORDER_QTY_COL,
    REORDER_POINT_COL,
    SAFETY_STOCK_COL,
    SALES_COL,
    STOCK_ON_HAND_COL,
    STORE_COL,
    UNIT_COST_COL,
    Y_HAT_COL,
    validate_schema,
)

DEFAULT_SERVICE_LEVEL = 0.95
RANDOM_SEED = 42
DAYS_PER_YEAR = 365

STOCKOUT_RISK = "stockout_risk"
OK = "ok"
OVERSTOCK = "overstock"

_KEYS = [STORE_COL, ITEM_COL]


def _require(name: str, value: float, positive: bool = False) -> None:
    """Reject NaN/inf/negative (and zero if positive=True)."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}, got {value}")


def _check_columns(name: str, df: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def z_score(service_level: float = DEFAULT_SERVICE_LEVEL) -> float:
    """Normal z-value for a cycle service level, e.g. 0.95 -> ~1.645."""
    if not 0 < service_level < 1:
        raise ValueError("service_level must be in (0, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(
    sigma_daily: float,
    lead_time_days: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """z * sigma_daily * sqrt(lead_time). sigma_daily = std of daily forecast error."""
    _require("sigma_daily", sigma_daily)
    _require("lead_time_days", lead_time_days)
    return z_score(service_level) * sigma_daily * sqrt(lead_time_days)


def reorder_point(
    mean_daily_demand: float, lead_time_days: float, safety_stock_units: float
) -> float:
    """Expected demand during lead time + safety stock."""
    _require("mean_daily_demand", mean_daily_demand)
    _require("lead_time_days", lead_time_days)
    _require("safety_stock_units", safety_stock_units)
    return mean_daily_demand * lead_time_days + safety_stock_units


def eoq(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_cost_rate: float,
) -> float:
    """sqrt(2 * D * S / H), H = unit_cost * holding_cost_rate."""
    _require("annual_demand", annual_demand)
    _require("order_cost", order_cost)
    _require("unit_cost", unit_cost)
    _require("holding_cost_rate", holding_cost_rate)
    holding_cost = unit_cost * holding_cost_rate
    if holding_cost <= 0:
        raise ValueError("unit_cost * holding_cost_rate must be > 0")
    return sqrt(2 * annual_demand * order_cost / holding_cost)


def max_stock_level(reorder_point_units: float, eoq_units: float) -> float:
    """Order-up-to level: ROP + EOQ."""
    return reorder_point_units + eoq_units


def stock_flag(
    stock_on_hand: float, reorder_point_units: float, eoq_units: float
) -> str:
    _require("stock_on_hand", stock_on_hand)
    _require("reorder_point_units", reorder_point_units)
    _require("eoq_units", eoq_units)
    # stockout_risk only if an order would actually be placed (qty > 0);
    # with zero demand ROP = EOQ = 0, so stock 0 is ok, not a risk.
    if (
        stock_on_hand <= reorder_point_units
        and max_stock_level(reorder_point_units, eoq_units) > stock_on_hand
    ):
        return STOCKOUT_RISK
    if stock_on_hand > max_stock_level(reorder_point_units, eoq_units):
        return OVERSTOCK
    return OK


def recommended_order_qty(
    stock_on_hand: float, reorder_point_units: float, eoq_units: float
) -> int:
    """If stock <= ROP, order up to ROP + EOQ (>= one EOQ). Else 0."""
    _require("stock_on_hand", stock_on_hand)
    _require("reorder_point_units", reorder_point_units)
    _require("eoq_units", eoq_units)
    if stock_on_hand <= reorder_point_units:
        return int(ceil(max_stock_level(reorder_point_units, eoq_units) - stock_on_hand))
    return 0


def simulate_inventory(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simulated inventory per (store, item). Deterministic for a given seed.

    Input: cleaned sales df (validated against schema).
    Output cols: store, item, lead_time_days, unit_cost, order_cost,
                 holding_cost_rate, stock_on_hand  (sorted by store, item)

    Assumptions:
      lead_time_days     ~ int uniform [2, 14]
      unit_cost          ~ uniform [5, 50], 2 dp
      order_cost         ~ uniform [20, 100], 2 dp
      holding_cost_rate  ~ uniform [0.15, 0.30], 3 dp (fraction of unit_cost / year)
      stock_on_hand      = mean daily sales * days of cover, cover ~ uniform [2, 30]
    """
    validate_schema(df)
    rng = np.random.default_rng(seed)
    mean_daily = df.groupby(_KEYS)[SALES_COL].mean().sort_index()
    n = len(mean_daily)

    out = mean_daily.index.to_frame(index=False)
    out[LEAD_TIME_COL] = rng.integers(2, 15, n)
    out[UNIT_COST_COL] = rng.uniform(5, 50, n).round(2)
    out[ORDER_COST_COL] = rng.uniform(20, 100, n).round(2)
    out[HOLDING_RATE_COL] = rng.uniform(0.15, 0.30, n).round(3)
    cover_days = rng.uniform(2, 30, n)
    out[STOCK_ON_HAND_COL] = np.round(mean_daily.to_numpy() * cover_days).astype(int)
    return out


def build_inventory_plan(
    forecast: pd.DataFrame,
    errors: pd.DataFrame,
    inventory: pd.DataFrame,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> pd.DataFrame:
    """Combine forecast + validation errors + simulated inventory into a plan.

    forecast : date, store, item, y_hat, model  (ONE model only)
    errors   : store, item, error  (daily forecast error on validation window)
    inventory: output of simulate_inventory()

    sigma_daily = std of errors per series. Series with < 2 error points get
    NaN std; those are filled with the median sigma of the other series.
    
    NOTE: std measures spread around the mean error, so a biased forecast
    looks safer than it is (RMSE would be more conservative). We follow the
    standard formula; revisit if the chosen model is clearly biased.

    If `errors` has a `model` column, only rows matching the forecast's
    model are used.
    """
    _check_columns("forecast", forecast, _KEYS + [Y_HAT_COL, MODEL_COL])
    _check_columns("errors", errors, _KEYS + [ERROR_COL])
    _check_columns(
        "inventory",
        inventory,
        _KEYS + [LEAD_TIME_COL, UNIT_COST_COL, ORDER_COST_COL,
                 HOLDING_RATE_COL, STOCK_ON_HAND_COL],
    )
    if forecast.empty:
        raise ValueError("forecast is empty")
    if forecast[MODEL_COL].nunique() > 1:
        raise ValueError("forecast must contain a single model; filter first")
    if MODEL_COL in errors.columns:
        errors = errors[errors[MODEL_COL] == forecast[MODEL_COL].iloc[0]]

    demand = (
        forecast.assign(**{Y_HAT_COL: forecast[Y_HAT_COL].clip(lower=0)})
        .groupby(_KEYS)[Y_HAT_COL]
        .mean()
        .rename("mean_daily_demand")
        .reset_index()
    )
    sigma = errors.groupby(_KEYS)[ERROR_COL].std().rename("sigma_daily").reset_index()

    plan = inventory.merge(demand, on=_KEYS, how="left").merge(sigma, on=_KEYS, how="left")
    if plan["mean_daily_demand"].isna().any():
        raise ValueError("some (store, item) series have no forecast")
    if plan["sigma_daily"].isna().all():
        raise ValueError("no usable forecast errors to estimate sigma")
    plan["sigma_daily"] = plan["sigma_daily"].fillna(plan["sigma_daily"].median())

    rows = []
    for r in plan.to_dict("records"):
        ss = safety_stock(r["sigma_daily"], r[LEAD_TIME_COL], service_level)
        rop = reorder_point(r["mean_daily_demand"], r[LEAD_TIME_COL], ss)
        q = eoq(
            r["mean_daily_demand"] * DAYS_PER_YEAR,
            r[ORDER_COST_COL],
            r[UNIT_COST_COL],
            r[HOLDING_RATE_COL],
        )
        stock = r[STOCK_ON_HAND_COL]
        rows.append(
            {
                STORE_COL: r[STORE_COL],
                ITEM_COL: r[ITEM_COL],
                STOCK_ON_HAND_COL: stock,
                SAFETY_STOCK_COL: ss,
                REORDER_POINT_COL: rop,
                EOQ_COL: q,
                FLAG_COL: stock_flag(stock, rop, q),
                ORDER_QTY_COL: recommended_order_qty(stock, rop, q),
            }
        )
    return pd.DataFrame(rows).sort_values(_KEYS).reset_index(drop=True)