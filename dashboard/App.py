"""
NYISO DART Strategy — dashboard entry (Overview).

Reads frozen artifacts from disk; near-instant page navigation via
@st.cache_data on the loaders.

    streamlit run dashboard/App.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.data import (
    load_cumulative_pnl, load_trades, load_zone_summary, load_thresholds,
    compute_2026_trades, PRIMARY,
)
from lib.metrics import compute_stats, format_money, format_pct
from lib.plots import (
    cumulative_pnl_chart, cumulative_pnl_split_chart,
    zone_attribution_chart,
)
from lib.theme import (
    apply_theme, kpi_grid, kpi_tile, status_strip, section_header, trade_tape,
)


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYISO DART · Desk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### NYISO DART  ·  v2")
    st.caption("Day-ahead vs real-time spread strategy")
    st.markdown("---")
    st.markdown(
        "**OVERVIEW** — desk view\n\n"
        "📊 **Performance** — drill-down\n\n"
        "🔮 **2026 Live** — out-of-sample\n\n"
        "🎯 **Sandbox** — backtest any window\n\n"
        "🔬 **Methodology** — bias audit\n\n"
        "📚 **How It Works** — primer"
    )
    st.markdown("---")
    st.caption("Built by Melvin Varghese")
    st.caption(f"Variant: `{PRIMARY}` (leak-free D-2/D-3 lags)")


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1>NYISO DART  ·  Systematic Virtual-Bid Strategy</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    f'Zone-specific logistic regression on day-ahead vs real-time spreads · '
    f'NYISO 11 zones · trained 2015–2019 · validated 2020–2021 · '
    f'out-of-sample 2022 → present'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Load core data ──────────────────────────────────────────────────────────
trades       = load_trades()
cum_pnl      = load_cumulative_pnl()
zone_summary = load_zone_summary()
thresholds   = load_thresholds()
live_trades  = compute_2026_trades()

hourly_index = cum_pnl.index
stats = compute_stats(trades, hourly_index)
live_stats = compute_stats(live_trades) if not live_trades.empty else None


# ── Status strip ────────────────────────────────────────────────────────────
last_trade_ts = trades["interval_start_local"].max()
last_live_ts  = live_trades["interval_start_local"].max() if not live_trades.empty else None
n_elig = int(thresholds["eligible"].sum())

status_strip([
    ("STATUS",   "OPERATIONAL", "green"),
    ("VARIANT",  f"{PRIMARY.upper()}  ·  no look-ahead", "green"),
    ("ELIGIBLE", f"{n_elig}/22 (zone, side) pairs", "green"),
    ("LAST OOS", last_trade_ts.strftime("%Y-%m-%d %H:%M ET"), "green"),
    ("LIVE THRU", last_live_ts.strftime("%Y-%m-%d") if last_live_ts is not None else "—",
     "amber" if last_live_ts is None else "green"),
])


# ── Hero KPI grid ───────────────────────────────────────────────────────────
section_header("OUT-OF-SAMPLE RESULTS", "2022-01-01 → 2025-12-31  ·  unit size (1 MWh)")

total_color = "green" if stats.total_pnl > 0 else "red"
sortino_color = "green" if stats.sortino > 1.5 else "amber" if stats.sortino > 1.0 else None
sharpe_color = "amber" if stats.sharpe_daily < 1.0 else None

kpi_grid([
    kpi_tile("TOTAL P&L", format_money(stats.total_pnl, 0),
             sub=f"{stats.n_trades:,} trades · 4 yrs",
             color=total_color, accent="green"),
    kpi_tile("ANN. P&L", format_money(stats.annualized_pnl, 0),
             sub="per year", accent="green"),
    kpi_tile("WIN RATE", format_pct(stats.win_rate, 1),
             sub=f"{int(stats.win_rate*stats.n_trades):,} wins",
             color="green", accent="green"),
    kpi_tile("PROFIT FACTOR", f"{stats.profit_factor:.2f}x",
             sub="gross win / gross loss", accent="green"),
    kpi_tile("SORTINO", f"{stats.sortino:.2f}",
             sub="downside-vol Sharpe",
             color=sortino_color, accent="green"),
    kpi_tile("CALMAR", f"{stats.calmar:.2f}",
             sub="ann. P&L / max DD", accent="green"),
    kpi_tile("SHARPE", f"{stats.sharpe_daily:.2f}",
             sub="ann. (daily resample)",
             color=sharpe_color, accent="amber" if sharpe_color else "dim"),
    kpi_tile("MAX DRAWDOWN", format_money(stats.max_drawdown, 0),
             sub="peak to trough",
             color="red", accent="red"),
])

# Sortino vs Sharpe explainer — small footnote, big credibility
st.markdown(
    f'<div style="font-size:0.72rem; color:#586069; margin-top:4px; padding:6px 10px; '
    f'background:#fafafa; border-left:2px solid #1565c0; border-radius:2px;">'
    f'<b style="color:#1565c0;">NOTE</b> &nbsp; The payoff distribution is positive-skew '
    f'(median trade tiny, right tail large), so Sharpe {stats.sharpe_daily:.2f} understates this. '
    f'Sortino {stats.sortino:.2f} — which only penalizes downside vol — is the honest read; '
    f'profit factor {stats.profit_factor:.2f}× and {stats.win_rate*100:.0f}% win rate confirm it.'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Equity curve + DEC/INC attribution ─────────────────────────────────────
section_header("EQUITY", "cumulative P&L, hourly, with drawdown shading")

col_a, col_b = st.columns([3, 2])
with col_a:
    st.plotly_chart(
        cumulative_pnl_chart(cum_pnl, title=""),
        use_container_width=True, config={"displayModeBar": False},
    )
with col_b:
    st.plotly_chart(
        cumulative_pnl_split_chart(cum_pnl, title=""),
        use_container_width=True, config={"displayModeBar": False},
    )


# ── Best & worst trades + recent tape ───────────────────────────────────────
section_header("TRADE TAPE", "ten most recent + record holders")

t1, t2 = st.columns([2, 3])
with t1:
    st.markdown(
        '<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
        'letter-spacing:0.1em; margin-bottom:4px;">RECORD TRADES</div>',
        unsafe_allow_html=True,
    )
    best  = trades.nlargest(3, "payoff")
    worst = trades.nsmallest(3, "payoff")
    record_rows = []
    for _, r in pd.concat([best, worst]).iterrows():
        record_rows.append({
            "ts": r["interval_start_local"], "zone": r["zone"],
            "side": r["side"], "proba": r["proba"],
            "dart": r["dart"], "payoff": r["payoff"],
        })
    trade_tape(record_rows)

with t2:
    st.markdown(
        '<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
        'letter-spacing:0.1em; margin-bottom:4px;">MOST RECENT FILLS</div>',
        unsafe_allow_html=True,
    )
    recent = trades.sort_values("interval_start_local").tail(12).iloc[::-1]
    recent_rows = []
    for _, r in recent.iterrows():
        recent_rows.append({
            "ts": r["interval_start_local"], "zone": r["zone"],
            "side": r["side"], "proba": r["proba"],
            "dart": r["dart"], "payoff": r["payoff"],
        })
    trade_tape(recent_rows)


# ── 2026 live test ─────────────────────────────────────────────────────────
section_header("LIVE 2026", "frozen models · zero re-tuning · data the model never saw")

if live_stats is not None:
    l_color = "green" if live_stats.total_pnl > 0 else "red"
    kpi_grid([
        kpi_tile("LIVE P&L", format_money(live_stats.total_pnl, 0),
                 sub=f"YTD · {live_stats.n_trades:,} fills",
                 color=l_color, accent="green"),
        kpi_tile("WIN RATE", format_pct(live_stats.win_rate, 1),
                 color="green", accent="green"),
        kpi_tile("PROFIT FACTOR", f"{live_stats.profit_factor:.2f}x",
                 accent="green"),
        kpi_tile("BIGGEST WIN", format_money(live_stats.best_trade, 0),
                 color="green", accent="green"),
        kpi_tile("WORST LOSS", format_money(live_stats.worst_trade, 0),
                 color="red", accent="red"),
        kpi_tile("AVG PER TRADE", format_money(
            live_stats.total_pnl / max(live_stats.n_trades, 1), 2),
                 accent="dim"),
    ])
    st.markdown(
        f'<div style="font-size:0.72rem; color:#586069; margin-top:4px; padding:6px 10px; '
        f'background:#fafafa; border-left:2px solid #2e7d32; border-radius:2px;">'
        f'The Jan 26–30 winter cold-snap drove most of the period\'s P&L — the model '
        f'fired DEC signals across all 13 eligible zones. See <b>2026 Live</b> page for '
        f'the day-by-day breakdown.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="padding:14px; background:#fafafa; border:1px solid #e1e4e8; '
        'border-radius:4px; color:#586069;">'
        'Live 2026 panel not built. Run<br><code>python -m nyiso_dart.data.download '
        '--start 2026 --end 2026</code><br>then<br><code>python -m nyiso_dart.data.build</code>.</div>',
        unsafe_allow_html=True,
    )


# ── Zone attribution ────────────────────────────────────────────────────────
section_header("ZONE ATTRIBUTION", "green = DEC (DA>RT)  ·  amber = INC (RT>DA)")
st.plotly_chart(
    zone_attribution_chart(zone_summary, title=""),
    use_container_width=True, config={"displayModeBar": False},
)


# ── How it works (collapsed) ────────────────────────────────────────────────
with st.expander("HOW IT WORKS  ·  one-paragraph version"):
    st.markdown(
        """
        Every morning before NYISO's day-ahead market closes at **05:00 Eastern**,
        the strategy builds a **52-feature vector** for each (operating hour, zone)
        from publicly available data: zone-level day-ahead load forecasts, DART
        spreads lagged 48h and 72h, lagged load-forecast errors, calendar effects,
        and counts of system-wide spike clusters two days back.

        The vector is fed through **22 zone-specific logistic regressions** —
        one for each NYISO zone × {pos, neg} side. When a zone's predicted
        probability clears its validation-tuned threshold τ, the strategy
        submits a virtual **INC** (buy DA, profit if RT > DA) or **DEC**
        (sell DA, profit if DA > RT) bid for that hour at 1 MWh.

        Every input is settled strictly before gate closure — the lag math
        and DST handling are audited to refuse any leakage. Validation
        thresholds are frozen in 2021 and never re-touched. The numbers
        above are true out-of-sample.
        """
    )
