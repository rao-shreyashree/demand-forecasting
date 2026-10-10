"""Inventory policy math: safety stock, reorder point, EOQ, stock flags.
"""
from math import ceil, sqrt
from statistics import NormalDist

DEFAULT_SERVICE_LEVEL = 0.95

STOCKOUT_RISK = "stockout_risk"
OK = "ok"
OVERSTOCK = "overstock"


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
    """z * sigma_daily * sqrt(lead_time).

    sigma_daily = std of daily forecast error (from validation errors).
    """
    if sigma_daily < 0 or lead_time_days < 0:
        raise ValueError("sigma_daily and lead_time_days must be >= 0")
    return z_score(service_level) * sigma_daily * sqrt(lead_time_days)


def reorder_point(
    mean_daily_demand: float, lead_time_days: float, safety_stock_units: float
) -> float:
    """Expected demand during lead time + safety stock."""
    if mean_daily_demand < 0 or lead_time_days < 0 or safety_stock_units < 0:
        raise ValueError("inputs must be >= 0")
    return mean_daily_demand * lead_time_days + safety_stock_units


def eoq(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_cost_rate: float,
) -> float:
    """Economic order quantity: sqrt(2 * D * S / H), H = unit_cost * holding_cost_rate."""
    if annual_demand < 0 or order_cost < 0:
        raise ValueError("annual_demand and order_cost must be >= 0")
    holding_cost = unit_cost * holding_cost_rate
    if holding_cost <= 0:
        raise ValueError("unit_cost * holding_cost_rate must be > 0")
    return sqrt(2 * annual_demand * order_cost / holding_cost)


def max_stock_level(reorder_point_units: float, eoq_units: float) -> float:
    """Upper bound of healthy stock: right after a reorder lands on top of ROP."""
    return reorder_point_units + eoq_units


def stock_flag(
    stock_on_hand: float, reorder_point_units: float, eoq_units: float
) -> str:
    """Classify current stock.

    stock_on_hand <  ROP            -> stockout_risk
    stock_on_hand >  ROP + EOQ      -> overstock
    otherwise                       -> ok
    """
    if stock_on_hand < reorder_point_units:
        return STOCKOUT_RISK
    if stock_on_hand > max_stock_level(reorder_point_units, eoq_units):
        return OVERSTOCK
    return OK


def recommended_order_qty(
    stock_on_hand: float, reorder_point_units: float, eoq_units: float
) -> int:
    """Order one EOQ (rounded up) if at/below ROP, else 0."""
    if stock_on_hand <= reorder_point_units:
        return int(ceil(eoq_units))
    return 0