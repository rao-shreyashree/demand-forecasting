import pytest

from src.inventory import (
    OK,
    OVERSTOCK,
    STOCKOUT_RISK,
    eoq,
    max_stock_level,
    recommended_order_qty,
    reorder_point,
    safety_stock,
    stock_flag,
    z_score,
)


def test_z_score_95():
    assert z_score(0.95) == pytest.approx(1.6448536, rel=1e-6)


def test_z_score_invalid():
    with pytest.raises(ValueError):
        z_score(1.0)
    with pytest.raises(ValueError):
        z_score(0)


def test_safety_stock_hand_computed():
    # 1.6448536 * 10 * sqrt(4) = 32.897072
    assert safety_stock(10, 4, 0.95) == pytest.approx(32.897072, rel=1e-6)


def test_safety_stock_default_is_95():
    assert safety_stock(10, 4) == pytest.approx(safety_stock(10, 4, 0.95))


def test_safety_stock_zero_sigma():
    assert safety_stock(0, 7) == 0


def test_safety_stock_negative_raises():
    with pytest.raises(ValueError):
        safety_stock(-1, 4)


def test_reorder_point():
    # 20 * 5 + 30 = 130
    assert reorder_point(20, 5, 30) == 130


def test_eoq_hand_computed():
    # H = 10 * 0.2 = 2 ; sqrt(2 * 1000 * 50 / 2) = sqrt(50000)
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
        (100, OK),  # exactly at ROP is not below it
        (300, OK),  # exactly at max is not above it
        (301, OVERSTOCK),
    ],
)
def test_stock_flag(stock, expected):
    assert stock_flag(stock, 100, 200) == expected


def test_recommended_order_qty():
    assert recommended_order_qty(100, 100, 223.6) == 224  # at ROP -> order
    assert recommended_order_qty(50, 100, 223.6) == 224
    assert recommended_order_qty(101, 100, 223.6) == 0