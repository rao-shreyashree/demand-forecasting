import numpy as np
import pandas as pd
import pytest

from dashboard import helpers as h
from src.schema import (
    DATE_COL, EOQ_COL, FLAG_COL, ITEM_COL, MODEL_COL, ORDER_QTY_COL,
    REORDER_POINT_COL, SAFETY_STOCK_COL, SALES_COL, STOCK_ON_HAND_COL,
    STORE_COL, Y_HAT_COL,
)


@pytest.fixture
def actuals():
    d = pd.date_range("2017-11-01", "2017-12-31")
    return pd.DataFrame({DATE_COL: d, STORE_COL: 1, ITEM_COL: 1, SALES_COL: range(len(d))})


@pytest.fixture
def forecasts():
    d = pd.date_range("2017-12-04", "2017-12-31")
    return pd.concat(
        [pd.DataFrame({DATE_COL: d, STORE_COL: 1, ITEM_COL: 1, Y_HAT_COL: 10.0, MODEL_COL: m})
         for m in ("sarima", "prophet")],
        ignore_index=True,
    )


@pytest.fixture
def plan():
    return pd.DataFrame({
        STORE_COL: [1, 1, 2, 2],
        ITEM_COL: [1, 2, 1, 2],
        STOCK_ON_HAND_COL: [5, 100, 50, 10],
        SAFETY_STOCK_COL: [10, 10, 10, 10],
        REORDER_POINT_COL: [30, 30, 40, 30],
        EOQ_COL: [100, 100, 80, 60],
        FLAG_COL: ["A", "B", "C", "A"],
        ORDER_QTY_COL: [100, 0, 0, 60],
    })


def test_forecast_vs_actual_columns_and_history(forecasts, actuals):
    out = h.forecast_vs_actual(forecasts, actuals, 1, 1, history_days=28)
    assert set(out.columns) == {"actual", "sarima", "prophet"}
    assert out.index.min() == pd.Timestamp("2017-11-06")
    assert out.index.max() == pd.Timestamp("2017-12-31")


def test_forecast_vs_actual_model_filter(forecasts, actuals):
    out = h.forecast_vs_actual(forecasts, actuals, 1, 1, models=["sarima"])
    assert list(out.columns) == ["actual", "sarima"]


def test_forecast_vs_actual_unknown_series_is_empty(forecasts, actuals):
    assert h.forecast_vs_actual(forecasts, actuals, 9, 9).empty


def test_flag_summary_counts(plan):
    s = h.flag_summary(plan)
    assert dict(zip(s[FLAG_COL], s["series"])) == {"A": 2, "B": 1, "C": 1}


def test_reorder_table_sorted_and_filtered(plan):
    t = h.reorder_table(plan)
    assert list(zip(t[STORE_COL], t[ITEM_COL])) == [(1, 1), (2, 2)]
    assert list(t["shortfall"]) == [25, 20]
    assert list(t[ORDER_QTY_COL]) == [100, 60]


def test_reorder_table_flag_filter(plan):
    t = h.reorder_table(plan, flags=["C"])
    assert t.empty


def test_load_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        h.load_plan(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        h.load_actuals(tmp_path / "nope.csv")


def test_forecast_metrics_values(forecasts, actuals):
    m = h.forecast_metrics(forecasts, actuals)
    assert set(m[MODEL_COL]) == {"sarima", "prophet"}
    r = m.iloc[0]
    # window 2017-12-04..31 -> actual sales 33..60 (sum 1302), forecast 10/day (sum 280)
    s_ = np.arange(33, 61)
    assert r["n"] == 28
    assert r["actual_total"] == 1302
    assert r["wape"] == pytest.approx(1022 / 1302)
    assert r["bias"] == pytest.approx(-1022 / 28)  # mean(y_hat - actual): under-forecast
    assert r["rmse"] == pytest.approx(np.sqrt(np.mean((10 - s_) ** 2)))


def test_forecast_metrics_by_store_and_empty(forecasts, actuals):
    m = h.forecast_metrics(forecasts, actuals, by=[STORE_COL])
    assert set(m[STORE_COL]) == {1}
    assert h.forecast_metrics(forecasts, actuals, store=9, item=9).empty


def test_error_series(forecasts, actuals):
    e = h.error_series(forecasts, actuals, 1, 1, models=["sarima"])
    assert set(e[MODEL_COL]) == {"sarima"}
    assert len(e) == 28
    assert e["error"].iloc[0] == 10 - 33


def test_trim_actuals(forecasts, actuals):
    t = h.trim_actuals(actuals, forecasts, days=10)
    assert t[DATE_COL].min() == pd.Timestamp("2017-11-24")


def test_series_plan_and_kpis(plan):
    assert h.series_plan(plan, 1, 1)[ORDER_QTY_COL] == 100
    assert h.series_plan(plan, 9, 9) is None
    assert h.plan_kpis(plan) == {"series": 4, "to_order": 2, "order_units": 160.0}


def test_flag_by_store(plan):
    f = h.flag_by_store(plan)
    assert f["series"].sum() == 4


def test_reorder_table_store_filter_and_pct(plan):
    t = h.reorder_table(plan, stores=[2])
    assert list(zip(t[STORE_COL], t[ITEM_COL])) == [(2, 2)]
    assert t.loc[0, "stock_vs_rop_pct"] == pytest.approx(10 / 30 * 100)