from math import ceil, nan

import numpy as np
import pandas as pd
import pytest

from src.inventory import (
    OK,
    OVERSTOCK,
    STOCKOUT_RISK,
    build_inventory_plan,
    eoq,
    max_stock_level,
    recommended_order_qty,
    reorder_point,
    safety_stock,
    simulate_inventory,
    stock_flag,
    z_score,
)
from src.schema import (
    ERROR_COL,
    FLAG_COL,
    HOLDING_RATE_COL,
    ITEM_COL,
    LEAD_TIME_COL,
    ORDER_COST_COL,
    ORDER_QTY_COL,
    REORDER_POINT_COL,
    SAFETY_STOCK_COL,
    EOQ_COL,
    SALES_COL,
    DATE_COL,
    STOCK_ON_HAND_COL,
    STORE_COL,
    UNIT_COST_COL,
    Y_HAT_COL,
    MODEL_COL,
)


def test_z_score_95():
    assert z_score(0.95) == pytest.approx(1.6448536, rel=1e-6)


@pytest.mark.parametrize("bad", [0, 1.0, -0.1, 1.5])
def test_z_score_invalid(bad):
    with pytest.raises(ValueError):
        z_score(bad)


def test_safety_stock_hand_computed():
    # 1.6448536 * 10 * sqrt(4)
    assert safety_stock(10, 4, 0.95) == pytest.approx(32.897072, rel=1e-6)


def test_safety_stock_default_is_95():
    assert safety_stock(10, 4) == pytest.approx(safety_stock(10, 4, 0.95))


def test_safety_stock_zero_sigma():
    assert safety_stock(0, 7) == 0


@pytest.mark.parametrize("sigma", [-1, nan, float("inf")])
def test_safety_stock_rejects_bad_sigma(sigma):
    with pytest.raises(ValueError):
        safety_stock(sigma, 4)


def test_reorder_point():
    # 20 * 5 + 30 = 130
    assert reorder_point(20, 5, 30) == 130


def test_eoq_hand_computed():
    # H = 10 * 0.2 = 2 ; sqrt(2 * 1000 * 50 / 2)
    assert eoq(1000, 50, 10, 0.2) == pytest.approx(223.6068, rel=1e-6)


def test_eoq_zero_holding_cost_raises():
    with pytest.raises(ValueError):
        eoq(1000, 50, 10, 0)


def test_max_stock_level():
    assert max_stock_level(100, 200) == 300


@pytest.mark.parametrize(
    "stock, expected",
    [
        (0, STOCKOUT_RISK),
        (99, STOCKOUT_RISK),
        (100, STOCKOUT_RISK),  # at ROP counts as reorder
        (101, OK),
        (300, OK),  # exactly at max
        (301, OVERSTOCK),
    ],
)
def test_stock_flag(stock, expected):
    assert stock_flag(stock, 100, 200) == expected


@pytest.mark.parametrize("stock", [-1, nan])
def test_stock_validation(stock):
    with pytest.raises(ValueError):
        stock_flag(stock, 100, 200)
    with pytest.raises(ValueError):
        recommended_order_qty(stock, 100, 200)


def test_order_qty_orders_up_to_rop_plus_eoq():
    # ROP 130, EOQ 50, stock 0 -> 180 (one EOQ alone would leave 50 < ROP)
    assert recommended_order_qty(0, 130, 50) == 180
    assert recommended_order_qty(130, 130, 50) == 50
    assert recommended_order_qty(131, 130, 50) == 0


@pytest.mark.parametrize("stock", [0, 50, 100, 101, 250, 301])
def test_flag_and_order_agree(stock):
    flag = stock_flag(stock, 100, 200)
    qty = recommended_order_qty(stock, 100, 200)
    assert (flag == STOCKOUT_RISK) == (qty > 0)


def _sales_df():
    dates = pd.date_range("2020-01-01", periods=10)
    rows = [
        (d, s, i, 10 * s + i + k)
        for s in (2, 1)
        for i in (3, 1, 2)
        for k, d in enumerate(dates)
    ]
    return pd.DataFrame(rows, columns=[DATE_COL, STORE_COL, ITEM_COL, SALES_COL])


def test_simulate_reproducible_and_sorted():
    a = simulate_inventory(_sales_df(), seed=7)
    b = simulate_inventory(_sales_df(), seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 6
    assert list(a[[STORE_COL, ITEM_COL]].itertuples(index=False, name=None)) == sorted(
        zip(a[STORE_COL], a[ITEM_COL])
    )


def test_simulate_different_seed_differs():
    a = simulate_inventory(_sales_df(), seed=1)
    b = simulate_inventory(_sales_df(), seed=2)
    assert not a.equals(b)


def test_simulate_columns_ranges_types():
    out = simulate_inventory(_sales_df())
    assert list(out.columns) == [
        STORE_COL, ITEM_COL, LEAD_TIME_COL, UNIT_COST_COL,
        ORDER_COST_COL, HOLDING_RATE_COL, STOCK_ON_HAND_COL,
    ]
    assert pd.api.types.is_integer_dtype(out[STORE_COL])
    assert pd.api.types.is_integer_dtype(out[ITEM_COL])
    assert out[LEAD_TIME_COL].between(2, 14).all()
    assert out[UNIT_COST_COL].between(5, 50).all()
    assert out[HOLDING_RATE_COL].between(0.15, 0.30).all()
    assert (out[STOCK_ON_HAND_COL] >= 0).all()


def test_simulate_validates_schema():
    with pytest.raises(ValueError):
        simulate_inventory(_sales_df().drop(columns=[SALES_COL]))


def _plan_inputs(error_points_item2=(3.0, -3.0, 4.0, -4.0)):
    dates = pd.date_range("2021-01-01", periods=28)
    forecast = pd.DataFrame(
        [(d, 1, i, 20.0, "naive") for i in (1, 2) for d in dates],
        columns=[DATE_COL, STORE_COL, ITEM_COL, Y_HAT_COL, MODEL_COL],
    )
    errors = pd.DataFrame(
        [(1, 1, 5.0), (1, 1, -5.0)] + [(1, 2, e) for e in error_points_item2],
        columns=[STORE_COL, ITEM_COL, ERROR_COL],
    )
    inventory = pd.DataFrame(
        {
            STORE_COL: [1, 1],
            ITEM_COL: [1, 2],
            LEAD_TIME_COL: [4, 4],
            UNIT_COST_COL: [10.0, 10.0],
            ORDER_COST_COL: [50.0, 50.0],
            HOLDING_RATE_COL: [0.2, 0.2],
            STOCK_ON_HAND_COL: [30, 100000],
        }
    )
    return forecast, errors, inventory


def test_plan_values_and_flags():
    plan = build_inventory_plan(*_plan_inputs())
    r = plan.iloc[0]
    sigma = np.std([5.0, -5.0], ddof=1)
    ss = safety_stock(sigma, 4)
    rop = reorder_point(20.0, 4, ss)
    q = eoq(20.0 * 365, 50.0, 10.0, 0.2)
    assert r[SAFETY_STOCK_COL] == pytest.approx(ss)
    assert r[REORDER_POINT_COL] == pytest.approx(rop)
    assert r[EOQ_COL] == pytest.approx(q)
    assert r[FLAG_COL] == STOCKOUT_RISK
    assert r[ORDER_QTY_COL] == ceil(rop + q - 30)
    assert plan.iloc[1][FLAG_COL] == OVERSTOCK
    assert plan.iloc[1][ORDER_QTY_COL] == 0


def test_plan_fills_nan_sigma_for_single_error_point():
    forecast, errors, inventory = _plan_inputs(error_points_item2=(3.0,))
    plan = build_inventory_plan(forecast, errors, inventory)
    assert plan[SAFETY_STOCK_COL].notna().all()
    # item 2 sigma filled from the only other series (item 1)
    assert plan.iloc[1][SAFETY_STOCK_COL] == pytest.approx(safety_stock(np.std([5.0, -5.0], ddof=1), 4))


def test_plan_rejects_multiple_models():
    forecast, errors, inventory = _plan_inputs()
    forecast.loc[0, MODEL_COL] = "ma"
    with pytest.raises(ValueError):
        build_inventory_plan(forecast, errors, inventory)


def test_plan_rejects_missing_forecast():
    forecast, errors, inventory = _plan_inputs()
    forecast = forecast[forecast[ITEM_COL] == 1]
    with pytest.raises(ValueError):
        build_inventory_plan(forecast, errors, inventory)