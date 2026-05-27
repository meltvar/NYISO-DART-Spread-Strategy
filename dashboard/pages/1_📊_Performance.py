"""Performance page — yearly / monthly / zone drill-down."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import load_cumulative_pnl, load_trades, load_zone_summary, load_precision_recall
from lib.metrics import compute_stats, format_money, format_pct
from lib.plots import (
    monthly_pnl_heatmap, zone_attribution_chart,
    precision_recall_scatter, trade_payoff_distribution, yearly_pnl_bar,
)
from lib.theme import (
    apply_theme, kpi_grid, kpi_tile, status_strip, section_header,
)


st.set_page_config(page_title="Performance · NYISO DART", page_icon="📊", layout="wide")
apply_theme()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>Performance · Drill-Down</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    'Year, month, zone, and trade-level breakdowns of out-of-sample performance.'
    '</div>',
    unsafe_allow_html=True,
)


trades = load_trades()
cum_pnl = load_cumulative_pnl()
zone_summary = load_zone_summary()
pr = load_precision_recall()
stats = compute_stats(trades, cum_pnl.index)


# ── Status / KPI row ────────────────────────────────────────────────────────
status_strip([
    ("PERIOD", "2022-01-01 → 2025-12-31", "green"),
    ("TRADES", f"{stats.n_trades:,}", "green"),
    ("AVG/TRADE", format_money(stats.total_pnl / max(stats.n_trades, 1), 2), "green"),
])

section_header("HEADLINE STATISTICS")
kpi_grid([
    kpi_tile("TOTAL P&L", format_money(stats.total_pnl, 0), color="green", accent="green"),
    kpi_tile("ANN. P&L", format_money(stats.annualized_pnl, 0), accent="green"),
    kpi_tile("SHARPE (D)", f"{stats.sharpe_daily:.2f}", accent="dim"),
    kpi_tile("SORTINO", f"{stats.sortino:.2f}", color="green", accent="green"),
    kpi_tile("CALMAR", f"{stats.calmar:.2f}", color="green", accent="green"),
    kpi_tile("WIN RATE", format_pct(stats.win_rate, 1), color="green", accent="green"),
    kpi_tile("PROFIT FACTOR", f"{stats.profit_factor:.2f}x", accent="green"),
    kpi_tile("MAX DD", format_money(stats.max_drawdown, 0), color="red", accent="red"),
])


# ── Yearly ──────────────────────────────────────────────────────────────────
section_header("BY YEAR")
col_a, col_b = st.columns([2, 3])

with col_a:
    yearly = (
        trades.assign(year=pd.DatetimeIndex(trades["interval_start_local"]).year)
        .groupby("year")
        .agg(
            total_pnl=("payoff", "sum"),
            n_trades=("payoff", "size"),
            win_rate=("payoff", lambda x: (x > 0).mean()),
            avg_trade=("payoff", "mean"),
        )
        .round(2)
    )
    st.dataframe(
        yearly.style.format({
            "total_pnl": lambda x: format_money(x, 0),
            "n_trades":  "{:,.0f}",
            "win_rate":  lambda x: f"{x:.1%}",
            "avg_trade": lambda x: format_money(x, 2),
        }),
        use_container_width=True,
        height=210,
    )

with col_b:
    st.plotly_chart(
        yearly_pnl_bar(trades, title=""),
        use_container_width=True, config={"displayModeBar": False},
    )


# ── Monthly heatmap ─────────────────────────────────────────────────────────
section_header("MONTHLY HEATMAP", "year × month, USD")
st.plotly_chart(
    monthly_pnl_heatmap(trades, title=""),
    use_container_width=True, config={"displayModeBar": False},
)


# ── Zone attribution + precision/recall ─────────────────────────────────────
section_header("ZONE × SIDE")
left, right = st.columns(2)

with left:
    st.plotly_chart(
        zone_attribution_chart(zone_summary, title="P&L ATTRIBUTION"),
        use_container_width=True, config={"displayModeBar": False},
    )

with right:
    if not pr.empty:
        st.plotly_chart(
            precision_recall_scatter(pr, title="PRECISION × RECALL"),
            use_container_width=True, config={"displayModeBar": False},
        )
        st.markdown(
            '<div style="font-size:0.72rem; color:#586069;">'
            'Bubble size = total predictions (TP+FP). DEC trades (green) cluster at high '
            'precision; INC (amber) are rarer but show similar precision in strong zones.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("precision_recall.csv not found. Run `python -m nyiso_dart.backtest.report`.")


# ── Trade payoff distribution ───────────────────────────────────────────────
section_header("PAYOFF DISTRIBUTION")
st.plotly_chart(
    trade_payoff_distribution(trades, title=""),
    use_container_width=True, config={"displayModeBar": False},
)

kpi_grid([
    kpi_tile("AVG WIN", format_money(stats.avg_win, 2), color="green", accent="green"),
    kpi_tile("AVG LOSS", format_money(stats.avg_loss, 2), color="red", accent="red"),
    kpi_tile("BEST TRADE", format_money(stats.best_trade, 0), color="green", accent="green"),
    kpi_tile("WORST TRADE", format_money(stats.worst_trade, 0), color="red", accent="red"),
])

st.markdown(
    '<div style="font-size:0.78rem; color:#1a1a1a; margin-top:14px; padding:12px; '
    'background:#fafafa; border-left:2px solid #f57c00; border-radius:2px;">'
    'Right-skewed payoff: small wins dominate the median, large winners dominate the total. '
    'This is by design — the model targets the tail of the DART distribution, not the center.'
    '</div>',
    unsafe_allow_html=True,
)


# ── Full table ──────────────────────────────────────────────────────────────
section_header("FULL ATTRIBUTION", "sorted by P&L, descending")
display = zone_summary.copy().sort_values("total_pnl", ascending=False)
display["side"] = display["side"].str.upper()
st.dataframe(
    display.style.format({
        "n_trades":  "{:,.0f}",
        "n_spikes":  "{:,.0f}",
        "precision": "{:.1%}",
        "total_pnl": lambda x: format_money(x, 2),
        "avg_pnl":   lambda x: format_money(x, 2),
    }),
    use_container_width=True,
    hide_index=True,
)
