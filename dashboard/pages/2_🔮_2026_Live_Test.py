"""2026 live test page — the showpiece. True out-of-sample on post-deployment data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import compute_2026_trades, load_panel_2026, load_trades
from lib.metrics import compute_stats, format_money, format_pct
from lib.plots import yoy_comparison_chart, _terminal_layout
from lib.theme import (
    apply_theme, kpi_grid, kpi_tile, status_strip, section_header, trade_tape,
    GREEN, GREEN_DIM, RED, AMBER, PURPLE, TEXT, TEXT_DIM, TEXT_FAINT, BG, BG_PANEL,
)


st.set_page_config(page_title="2026 Live · NYISO DART", page_icon="🔮", layout="wide")
apply_theme()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>2026 Live  ·  True Out-of-Sample</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    'Models trained 2015–2019 · thresholds locked 2020–2021 · applied to NYISO data '
    'the model never saw.  Zero re-fitting, zero re-tuning.'
    '</div>',
    unsafe_allow_html=True,
)


# ── Load ────────────────────────────────────────────────────────────────────
live_trades = compute_2026_trades()
hist_trades = load_trades()
panel_2026  = load_panel_2026()

if live_trades.empty or panel_2026 is None:
    st.markdown(
        '<div style="padding:18px; background:#fafafa; border:1px solid #e1e4e8; '
        'border-radius:4px; color:#586069;">'
        '2026 live panel not built. Run:<br><br>'
        '<code>python -m nyiso_dart.data.download --start 2026 --end 2026</code><br>'
        '<code>python -m nyiso_dart.data.build</code>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

live_stats = compute_stats(live_trades)


# ── Status strip ────────────────────────────────────────────────────────────
last_live_ts = live_trades["interval_start_local"].max()
status_strip([
    ("STATUS",   "LIVE", "green"),
    ("WINDOW",   f"2026-01-01 → {last_live_ts.strftime('%Y-%m-%d')}", "green"),
    ("MODEL FROZEN AT", "2021-12-31", "green"),
    ("VINTAGE",  "trained 2015–2019", "green"),
])


# ── Hero KPI grid ───────────────────────────────────────────────────────────
section_header("LIVE 2026 RESULTS")
ytd_color = "green" if live_stats.total_pnl > 0 else "red"
kpi_grid([
    kpi_tile("YTD P&L", format_money(live_stats.total_pnl, 0),
             sub=f"{live_stats.n_trades:,} fills", color=ytd_color, accent="green"),
    kpi_tile("WIN RATE", format_pct(live_stats.win_rate, 1),
             color="green", accent="green"),
    kpi_tile("PROFIT FACTOR", f"{live_stats.profit_factor:.2f}x", accent="green"),
    kpi_tile("PRECISION", format_pct(live_stats.precision, 1),
             sub="trade ↔ realized spike", accent="green"),
    kpi_tile("BEST TRADE", format_money(live_stats.best_trade, 0),
             color="green", accent="green"),
    kpi_tile("WORST TRADE", format_money(live_stats.worst_trade, 0),
             color="red", accent="red"),
    kpi_tile("DEC P&L", format_money(live_stats.pos_pnl, 0),
             sub=f"{live_stats.pos_trades:,} fills", color="green", accent="green"),
    kpi_tile("INC P&L", format_money(live_stats.neg_pnl, 0),
             sub=f"{live_stats.neg_trades:,} fills",
             color="green" if live_stats.neg_pnl >= 0 else "red", accent="amber"),
])


# ── 2026 equity curve ───────────────────────────────────────────────────────
section_header("2026 EQUITY CURVE", "hourly cumulative, drawdown shaded")

panel_2026["interval_start_local"] = pd.DatetimeIndex(panel_2026["interval_start_local"])
hourly_idx = (
    panel_2026[panel_2026["interval_start_local"].dt.year == 2026]
    ["interval_start_local"].drop_duplicates().sort_values()
)
hourly_pnl_total = (
    live_trades.groupby("interval_start_local")["payoff"].sum()
    .reindex(hourly_idx).fillna(0.0)
)
cum = hourly_pnl_total.cumsum()
running_max = cum.cummax()
underwater = cum.where(cum < running_max, running_max)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cum.index, y=running_max, mode="lines",
    line=dict(color=TEXT_FAINT, width=0.6, dash="dot"),
    hoverinfo="skip", showlegend=False,
))
fig.add_trace(go.Scatter(
    x=cum.index, y=underwater, mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(255, 77, 109, 0.10)",
    hoverinfo="skip", showlegend=False,
))
fig.add_trace(go.Scatter(
    x=cum.index, y=cum.values, mode="lines",
    line=dict(color=PURPLE, width=2.2),
    hovertemplate="%{x|%b %d %H:%M}<br>$%{y:,.0f}<extra></extra>",
    showlegend=False,
))
fig.add_hline(y=0, line=dict(color=TEXT_FAINT, width=0.5))
fig = _terminal_layout(fig, height=380, title=None)
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Year-over-year ──────────────────────────────────────────────────────────
section_header("YEAR-OVER-YEAR", "same calendar window across all years")
st.plotly_chart(
    yoy_comparison_chart(hist_trades, live_trades, cap_doy=135, title=""),
    use_container_width=True, config={"displayModeBar": False},
)


# ── January cold-snap deep dive ─────────────────────────────────────────────
section_header("THE JAN 26–30 EVENT", "where most of 2026's P&L came from")

jan_event = panel_2026[
    (panel_2026["interval_start_local"].dt.year == 2026)
    & (panel_2026["interval_start_local"].dt.month == 1)
    & (panel_2026["interval_start_local"].dt.day.between(26, 30))
].copy()

daily_event = (
    jan_event.assign(date=lambda d: d["interval_start_local"].dt.date)
    .groupby("date")["dart"].agg(["mean", "max", "min"]).round(1)
)

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_event.index, y=daily_event["max"],
        mode="lines+markers", name="MAX DART",
        line=dict(color=GREEN, width=1.5, dash="dot"),
        marker=dict(color=GREEN, size=8),
        hovertemplate="%{x}<br>Max: $%{y}/MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily_event.index, y=daily_event["mean"],
        mode="lines+markers", name="MEAN DART",
        line=dict(color=AMBER, width=2),
        marker=dict(color=AMBER, size=10),
        hovertemplate="%{x}<br>Mean: $%{y}/MWh<extra></extra>",
    ))
    fig = _terminal_layout(fig, height=320, title=None)
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with c2:
    event_trades = live_trades[
        live_trades["interval_start_local"].dt.day.between(26, 30) &
        (live_trades["interval_start_local"].dt.month == 1)
    ]
    event_pnl = event_trades["payoff"].sum()
    kpi_grid([
        kpi_tile("EVENT P&L", format_money(event_pnl, 0),
                 sub="5 days · Jan 26–30",
                 color="green", accent="green"),
        kpi_tile("EVENT TRADES", f"{len(event_trades):,}",
                 sub=f"avg ${event_pnl/max(len(event_trades),1):,.0f}/trade",
                 accent="green"),
        kpi_tile("% OF YTD", f"{event_pnl/max(live_stats.total_pnl,1)*100:.0f}%",
                 sub="of 2026 YTD P&L",
                 color="amber", accent="amber"),
    ])
    st.dataframe(
        daily_event.style.format({"mean": "${:,.1f}", "max": "${:,.1f}", "min": "${:,.1f}"}),
        use_container_width=True, height=210,
    )


# ── Trade tape for the event ────────────────────────────────────────────────
section_header("EVENT FILLS", "top 12 by P&L from Jan 26–30")
top_event = event_trades.nlargest(12, "payoff")
rows = []
for _, r in top_event.iterrows():
    rows.append({
        "ts": r["interval_start_local"], "zone": r["zone"], "side": r["side"],
        "proba": r["proba"], "dart": r["dart"], "payoff": r["payoff"],
    })
trade_tape(rows)


# ── Monthly breakdown ───────────────────────────────────────────────────────
section_header("MONTHLY BREAKDOWN")

month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May"}
monthly = (
    live_trades.assign(month=lambda d: d["interval_start_local"].dt.month)
    .groupby("month")
    .agg(total_pnl=("payoff", "sum"),
         n_trades=("payoff", "size"),
         avg_pnl=("payoff", "mean"),
         win_rate=("payoff", lambda x: (x > 0).mean()))
)

col_a, col_b = st.columns([2, 1])
with col_a:
    fig = go.Figure(go.Bar(
        x=[month_names[m] for m in monthly.index],
        y=monthly["total_pnl"],
        marker=dict(color=[GREEN if v >= 0 else RED for v in monthly["total_pnl"]],
                    line=dict(color=BG, width=0.5)),
        text=[f"${v:,.0f}" for v in monthly["total_pnl"]],
        textfont=dict(color=TEXT, size=11),
        textposition="outside",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig = _terminal_layout(fig, height=300, title=None)
    fig.add_hline(y=0, line=dict(color=TEXT_FAINT, width=0.5))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_b:
    display_monthly = monthly.copy()
    display_monthly.index = [month_names[m] for m in display_monthly.index]
    st.dataframe(
        display_monthly.style.format({
            "total_pnl": lambda x: format_money(x, 0),
            "n_trades":  "{:,.0f}",
            "avg_pnl":   lambda x: format_money(x, 2),
            "win_rate":  "{:.1%}",
        }),
        use_container_width=True,
    )


# ── Per-zone attribution ────────────────────────────────────────────────────
section_header("2026 ZONE × SIDE")
zone_attr = (
    live_trades.groupby(["zone", "side"])
    .agg(n_trades=("payoff", "size"),
         total_pnl=("payoff", "sum"),
         precision=("correct", "mean"),
         avg_pnl=("payoff", "mean"))
    .sort_values("total_pnl", ascending=False)
    .reset_index()
)
zone_attr["side"] = zone_attr["side"].str.upper()
st.dataframe(
    zone_attr.style.format({
        "n_trades":  "{:,.0f}",
        "total_pnl": lambda x: format_money(x, 2),
        "avg_pnl":   lambda x: format_money(x, 2),
        "precision": "{:.1%}",
    }),
    use_container_width=True, hide_index=True,
)


# ── Honest framing ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.78rem; color:#1a1a1a; margin-top:18px; padding:14px; '
    'background:#fafafa; border-left:2px solid #1565c0; border-radius:2px;">'
    '<b style="color:#1565c0;">HOW TO READ THIS</b><br><br>'
    'Jan 2026 saw a five-day extreme cold event. The DAM priced aggressively for '
    'worst-case demand; the RTM cleared lower as actual generation met it. '
    'The model identified this regime correctly across all 13 eligible zones. '
    '<br><br>Feb–May is essentially flat, consistent with shoulder-season behavior. '
    'The honest framing: 2026 proves the model <b>survives unseen regimes</b> — '
    'not that it earns this much every five months.'
    '</div>',
    unsafe_allow_html=True,
)
