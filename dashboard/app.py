"""Dashboard: forecast vs actual, model comparison, stock flags, reorder table.

Run from repo root:  streamlit run dashboard/app.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import helpers as h
from src.schema import (
    DATE_COL, EOQ_COL, FLAG_COL, ITEM_COL, MODEL_COL, ORDER_QTY_COL,
    REORDER_POINT_COL, SAFETY_STOCK_COL, STOCK_ON_HAND_COL, STORE_COL,
)

_T0 = time.perf_counter()
st.set_page_config(page_title="Demand Forecasting & Inventory Planning", page_icon="📦", layout="wide")
st.markdown("<style>.block-container{padding-top:2.2rem;}</style>", unsafe_allow_html=True)

ACTUAL_COLOR = "#0ea5e9"
PALETTE = ["#f97316", "#a855f7", "#22c55e", "#eab308", "#ec4899", "#14b8a6"]
GRID = "#64748b"
SIM_NOTE = (
    "Stock on hand, lead times and costs are **simulated** (fixed random seed), not real inventory data. "
    "Forecasts are real model outputs."
)


# ---------- helpers ----------
def show_df(df, **kw):
    """st.dataframe at full width across streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def color_map(models):
    return {"actual": ACTUAL_COLOR, **{m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}}


def fmt_date(d):
    return pd.Timestamp(d).strftime("%d %b %Y")


@st.cache_data(show_spinner="Loading data…")
def load_all():
    forecasts = h.load_forecasts()
    actuals = h.trim_actuals(h.load_actuals(), forecasts, days=120)
    plan = h.load_plan()
    return forecasts, actuals, plan


@st.cache_data(show_spinner="Computing model metrics…")
def global_metrics(common_only: bool):
    fc, ac, _ = load_all()
    if common_only:
        fc = h.restrict_to_common_series(fc)
    return h.forecast_metrics(fc, ac), h.forecast_metrics(fc, ac, by=[STORE_COL])


def forecast_chart(df, cmap):
    """Actual + model lines, shaded forecast window, hover crosshair with all values."""
    series = list(df.columns)
    scale = alt.Scale(domain=series, range=[cmap[s] for s in series])
    long = df.reset_index().melt(DATE_COL, var_name="series", value_name="value").dropna(subset=["value"])
    wide = df.reset_index()

    fc_dates = df.drop(columns="actual", errors="ignore").dropna(how="all").index
    window = alt.Chart(pd.DataFrame({"s": [fc_dates.min()], "e": [fc_dates.max()]})).mark_rect(
        opacity=0.08, color="#94a3b8"
    ).encode(x="s:T", x2="e:T")

    x = alt.X(f"{DATE_COL}:T", title=None, axis=alt.Axis(format="%d %b", labelOverlap=True))
    base = alt.Chart(long).encode(
        x=x,
        y=alt.Y("value:Q", title="Units sold", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
    )
    lines = base.mark_line(strokeWidth=2.2).encode(
        strokeDash=alt.condition(alt.datum.series == "actual", alt.value([1, 0]), alt.value([6, 3]))
    )
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=[DATE_COL], empty=False)
    selectors = alt.Chart(wide).mark_point(opacity=0).encode(x=f"{DATE_COL}:T").add_params(nearest)
    points = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    tips = [alt.Tooltip(f"{DATE_COL}:T", title="Date", format="%a, %d %b %Y")] + [
        alt.Tooltip(f"{c}:Q", format=".1f") for c in series
    ]
    rules = alt.Chart(wide).mark_rule(color=GRID).encode(x=f"{DATE_COL}:T", tooltip=tips).transform_filter(nearest)
    return alt.layer(window, lines, selectors, points, rules).properties(height=370, width="container")


def error_chart(err, cmap):
    models = sorted(err[MODEL_COL].unique())
    scale = alt.Scale(domain=models, range=[cmap[m] for m in models])
    lines = alt.Chart(err).mark_line(point=True, strokeWidth=1.6).encode(
        x=alt.X(f"{DATE_COL}:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("error:Q", title="Forecast − actual (units)"),
        color=alt.Color(f"{MODEL_COL}:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip(f"{DATE_COL}:T", format="%a, %d %b"), f"{MODEL_COL}:N", alt.Tooltip("error:Q", format="+.1f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GRID, strokeDash=[4, 4]).encode(y="y:Q")
    return (zero + lines).properties(height=260, width="container")


# ---------- load ----------
st.title("📦 Demand Forecasting & Inventory Planning")
try:
    forecasts, actuals, plan = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

all_models = sorted(forecasts[MODEL_COL].unique())
cmap = color_map(all_models)
series_df = h.available_series(forecasts)
st.caption(
    f"{len(series_df)} series · {len(all_models)} models ({', '.join(all_models)}) · "
    f"forecast window {fmt_date(forecasts[DATE_COL].min())} → {fmt_date(forecasts[DATE_COL].max())}"
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("Filters")
    store = st.selectbox("Store", sorted(series_df[STORE_COL].unique()))
    item = st.selectbox("Item", sorted(series_df.loc[series_df[STORE_COL] == store, ITEM_COL].unique()))
    avail = sorted(forecasts.loc[(forecasts[STORE_COL] == store) & (forecasts[ITEM_COL] == item), MODEL_COL].unique())
    models = st.multiselect("Models", avail, default=avail)
    history_days = st.slider("History shown before forecast (days)", 7, 90, 28, step=7)
    st.caption("Filters apply to the *Forecast vs actual* tab. Other tabs cover all series.")
    show_time = st.checkbox("Show render time")

tab_fc, tab_models, tab_flags, tab_reorder = st.tabs(
    ["📈 Forecast vs actual", "🏁 Model comparison", "🚦 Stock flags", "🛒 Reorder table"]
)

# ---------- tab 1: forecast vs actual ----------
with tab_fc:
    st.subheader(f"Store {store} · Item {item}")
    if not models:
        st.info("Pick at least one model in the sidebar.")
    else:
        chart_df = h.forecast_vs_actual(forecasts, actuals, store, item, models, history_days)
        met = h.forecast_metrics(forecasts, actuals, store, item, models)

        if not met.empty:
            cols = st.columns(len(met) + 1)
            with cols[0].container(border=True):
                st.metric("Actual demand (window)", f"{met['actual_total'].iloc[0]:,.0f} units")
            best = met.iloc[0][MODEL_COL]
            for col, (_, r) in zip(cols[1:], met.iterrows()):
                with col.container(border=True):
                    st.metric(
                        f"{'🏆 ' if r[MODEL_COL] == best else ''}{r[MODEL_COL]} · WAPE",
                        f"{r['wape']:.1%}",
                        delta=f"bias {r['bias']:+.1f}/day · RMSE {r['rmse']:.1f}",
                        delta_color="off",
                    )

        with st.container(border=True):
            st.altair_chart(forecast_chart(chart_df, cmap))
            st.caption("Shaded area = forecast window. Hover for exact values. Dashed = model forecasts.")

        left, right = st.columns([3, 2])
        with left.container(border=True):
            st.markdown("**Forecast error by day**")
            err = h.error_series(forecasts, actuals, store, item, models)
            if err.empty:
                st.caption("No overlapping actuals for this window.")
            else:
                st.altair_chart(error_chart(err, cmap))
                st.caption("Above 0 = over-forecast (overstock risk) · below 0 = under-forecast (stockout risk).")
        with right.container(border=True):
            st.markdown("**Inventory plan for this series**")
            row = h.series_plan(plan, store, item)
            if row is None:
                st.caption("This series isn't in the inventory plan.")
            else:
                pm, src = h.plan_model(row, forecasts, actuals, store, item)
                if pm:
                    st.caption(f"Plan based on: **{pm}** ({src})")
                a, b = st.columns(2)
                a.metric("Flag", str(row[FLAG_COL]))
                b.metric("Recommended order", f"{row[ORDER_QTY_COL]:,.0f}")
                a.metric("On hand (simulated)", f"{row[STOCK_ON_HAND_COL]:,.0f}")
                b.metric("Reorder point", f"{row[REORDER_POINT_COL]:,.0f}")
                a.metric("Safety stock", f"{row[SAFETY_STOCK_COL]:,.0f}")
                b.metric("EOQ", f"{row[EOQ_COL]:,.0f}")
                st.caption("Stock, lead time and costs are simulated.")

        with st.expander("Data"):
            d = chart_df.round(1).reset_index()
            d[DATE_COL] = d[DATE_COL].dt.strftime("%Y-%m-%d")
            shown = d.copy()
            for c in chart_df.columns:  # blank cells (history rows have no forecast) -> "–"
                shown[c] = d[c].map(lambda v: "–" if pd.isna(v) else f"{v:,.1f}")
            show_df(shown)
            st.download_button("Download CSV", d.to_csv(index=False), f"forecast_vs_actual_s{store}_i{item}.csv", "text/csv")

# ---------- tab 2: model comparison ----------
with tab_models:
    st.subheader("Model comparison — all series, forecast window")
    st.caption("Quick look on the shared forecast window. The official comparison is the rolling-origin backtest.")
    common_only = st.toggle(
        "Compare only on series covered by every model",
        value=True,
        help="Models cover different numbers of series (e.g. classical = top-N, ML = all). Comparing on shared series keeps it fair.",
    )
    overall, by_store = global_metrics(common_only)
    if overall.empty:
        st.info("No overlap between forecasts and actuals (or no series covered by every model).")
    else:
        disp = overall.assign(wape=overall["wape"] * 100)
        show_df(
            disp[[MODEL_COL, "wape", "bias", "rmse", "n_series", "n"]],
            column_config={
                MODEL_COL: "Model",
                "wape": st.column_config.NumberColumn("WAPE (%)", format="%.2f"),
                "bias": st.column_config.NumberColumn("Bias (units/day)", format="%+.2f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "n_series": st.column_config.NumberColumn("Series", format="%d"),
                "n": st.column_config.NumberColumn("Rows", format="%d"),
            },
        )
        base = alt.Chart(by_store).encode(
            x=alt.X(f"{MODEL_COL}:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{STORE_COL}:O", title="Store"),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("wape:Q", title="WAPE", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
            tooltip=[STORE_COL, MODEL_COL, alt.Tooltip("wape:Q", format=".1%"), alt.Tooltip("bias:Q", format="+.1f")],
        )
        text = base.mark_text(fontSize=11).encode(text=alt.Text("wape:Q", format=".1%"), color=alt.value("black"))
        with st.container(border=True):
            st.markdown("**WAPE by store**")
            st.altair_chart((heat + text).properties(height=max(200, 34 * by_store[STORE_COL].nunique()), width="container"))

# ---------- tab 3: stock flags ----------
with tab_flags:
    st.subheader("Stock flags")
    st.info(SIM_NOTE)
    k = h.plan_kpis(plan)
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.metric("Series in plan", f"{k['series']:,}")
    with c2.container(border=True):
        st.metric("Need a reorder", f"{k['to_order']:,}", delta=f"{k['to_order'] / max(k['series'], 1):.0%} of series", delta_color="off")
    with c3.container(border=True):
        st.metric("Units to order", f"{k['order_units']:,.0f}")

    flag_scale = alt.Scale(scheme="tableau10")
    summary = h.flag_summary(plan)
    a, b = st.columns([1, 2])
    with a.container(border=True):
        st.markdown("**Share of series by flag**")
        donut = alt.Chart(summary).mark_arc(innerRadius=55).encode(
            theta="series:Q",
            color=alt.Color(f"{FLAG_COL}:N", scale=flag_scale, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[FLAG_COL, "series"],
        ).properties(height=260)
        st.altair_chart(donut)
    with b.container(border=True):
        st.markdown("**Flags by store**")
        bars = alt.Chart(h.flag_by_store(plan)).mark_bar().encode(
            x=alt.X(f"{STORE_COL}:O", title="Store", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("series:Q", title="Series"),
            color=alt.Color(f"{FLAG_COL}:N", scale=flag_scale, legend=None),
            tooltip=[STORE_COL, FLAG_COL, "series"],
        ).properties(height=260, width="container")
        st.altair_chart(bars)

    with st.container(border=True):
        st.markdown("**On hand (simulated) vs reorder point**")
        lim = float(max(plan[STOCK_ON_HAND_COL].max(), plan[REORDER_POINT_COL].max()))
        pts = alt.Chart(plan).mark_circle(size=55, opacity=0.75).encode(
            x=alt.X(f"{REORDER_POINT_COL}:Q", title="Reorder point"),
            y=alt.Y(f"{STOCK_ON_HAND_COL}:Q", title="On hand (simulated)"),
            color=alt.Color(f"{FLAG_COL}:N", scale=flag_scale, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[STORE_COL, ITEM_COL, FLAG_COL, STOCK_ON_HAND_COL, REORDER_POINT_COL, ORDER_QTY_COL],
        )
        diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(color=GRID, strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", title="Reorder point"), y=alt.Y("y:Q", title="On hand (simulated)")
        )
        st.altair_chart((diag + pts).properties(height=340, width="container"))
        st.caption("Points below the dashed line are under their reorder point.")

    with st.expander("Full plan"):
        show_df(plan)

# ---------- tab 4: reorder table ----------
with tab_reorder:
    st.subheader("Reorder table")
    st.info(SIM_NOTE)
    f1, f2, f3 = st.columns(3)
    flag_opts = sorted(plan.loc[plan[ORDER_QTY_COL] > 0, FLAG_COL].unique())
    chosen_flags = f1.multiselect("Flags", flag_opts, default=flag_opts)
    store_opts = sorted(plan[STORE_COL].unique())
    chosen_stores = f2.multiselect("Stores", store_opts, default=store_opts)
    item_opts = sorted(plan[ITEM_COL].unique())
    chosen_items = f3.multiselect("Items (empty = all)", item_opts)
    table = h.reorder_table(plan, flags=chosen_flags, stores=chosen_stores, items=chosen_items or None)

    m1, m2, m3 = st.columns(3)
    with m1.container(border=True):
        st.metric("Series to order", f"{len(table):,}")
    with m2.container(border=True):
        st.metric("Units to order", f"{table[ORDER_QTY_COL].sum():,.0f}")
    with m3.container(border=True):
        top = f"S{table.iloc[0][STORE_COL]} · I{table.iloc[0][ITEM_COL]}" if len(table) else "—"
        st.metric("Most urgent", top)

    show_df(
        table,
        column_config={
            STORE_COL: "Store",
            ITEM_COL: "Item",
            STOCK_ON_HAND_COL: st.column_config.NumberColumn("On hand (sim.)", format="%d"),
            REORDER_POINT_COL: st.column_config.NumberColumn("Reorder pt", format="%d"),
            "stock_vs_rop_pct": st.column_config.ProgressColumn("Stock / ROP", min_value=0, max_value=100, format="%d%%"),
            SAFETY_STOCK_COL: st.column_config.NumberColumn("Safety stock", format="%d"),
            "shortfall": st.column_config.NumberColumn("Shortfall", format="%d"),
            EOQ_COL: st.column_config.NumberColumn("EOQ", format="%d"),
            ORDER_QTY_COL: st.column_config.NumberColumn("Order qty", format="%d"),
            FLAG_COL: "Flag",
        },
    )
    st.download_button("Download CSV", table.to_csv(index=False), "reorder_table.csv", "text/csv")

if show_time:
    st.sidebar.caption(f"Rendered in {time.perf_counter() - _T0:.2f}s")
