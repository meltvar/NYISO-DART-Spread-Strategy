"""How It Works — DART/INC/DEC explainer and one-trade trace."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import load_trades, load_panel, ZONES
from lib.plots import _terminal_layout
from lib.theme import (
    apply_theme, section_header, kpi_grid, kpi_tile, trade_tape,
    GREEN, GREEN_DIM, RED, AMBER, BLUE, PURPLE, TEXT, TEXT_DIM, TEXT_FAINT, BG, BG_PANEL,
)


st.set_page_config(page_title="How It Works · NYISO DART", page_icon="📚", layout="wide")
apply_theme()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>How It Works  ·  Primer</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    'From a market mechanic to a one-trade trace. What the model actually does.'
    '</div>',
    unsafe_allow_html=True,
)


# ── Market structure ────────────────────────────────────────────────────────
section_header("THE MARKET IN ONE MINUTE")

st.markdown(
    """
    NYISO clears electricity through a **two-settlement system**:

    | Market | When | What clears |
    |---|---|---|
    | **Day-Ahead (DAM)** | Bids submitted by **05:00 ET on day D−1** | Hourly LMPs for every operating hour of day D |
    | **Real-Time (RTM)** | Every 5 minutes during day D operations | Actual dispatch prices |

    The gap between these — **DART = DA price − RT price** — is a persistent risk
    factor driven by forecasting errors, transmission congestion, and unit-commitment
    decisions. Financial-only **virtual bids** let traders take positions on DART
    without physical delivery:
    """
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div style="border-left: 3px solid {GREEN}; padding: 10px 16px;
                    background:{BG_PANEL}; border-radius:3px;">
        <div style="color:{GREEN}; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.15em; font-weight:600;">DEC · VIRTUAL SUPPLY</div>
        <div style="color:{TEXT}; margin-top:8px; line-height:1.7;">
        Submit an offer to <b>SELL</b> 1 MWh at DA price.<br>
        Auto-buy back at RT price.<br>
        <b style="color:{GREEN};">Payoff = DA − RT = +DART</b>.<br>
        Profit when DA &gt; RT (DA overpriced).
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div style="border-left: 3px solid {AMBER}; padding: 10px 16px;
                    background:{BG_PANEL}; border-radius:3px;">
        <div style="color:{AMBER}; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.15em; font-weight:600;">INC · VIRTUAL DEMAND</div>
        <div style="color:{TEXT}; margin-top:8px; line-height:1.7;">
        Submit a bid to <b>BUY</b> 1 MWh at DA price.<br>
        Auto-sell back at RT price.<br>
        <b style="color:{AMBER};">Payoff = RT − DA = −DART</b>.<br>
        Profit when RT &gt; DA (DA underpriced).
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── The 11 zones ────────────────────────────────────────────────────────────
section_header("THE 11 NYISO ZONES")

st.markdown(
    """
    NYISO is divided into 11 load zones, each with its own LMP. The price difference
    between zones is driven by **transmission congestion** — when wires connecting
    zones hit capacity, prices split.

    | Zone | Region | Role |
    |---|---|---|
    | A · WEST | Western NY (Buffalo) | Generation-heavy, often low LMP |
    | B · GENESE | Rochester | Upstate |
    | C · CENTRL | Syracuse | Upstate, large generation |
    | D · NORTH | North country | Near Canadian border |
    | E · MHKVL | Mohawk Valley | Transmission corridor |
    | F · CAPITL | Albany | Mid-state |
    | G · HUDVL | Hudson Valley | Downstate-adjacent |
    | H · MILLWD | Millwood | Small downstate zone |
    | I · DUNWOD | Dunwoodie | Near NYC |
    | **J · NYC** | New York City | Largest load, heavily congested |
    | **K · LONGIL** | Long Island | Import-constrained load pocket |

    LONGIL and NYC are the most predictive — both are import-constrained, regularly
    experiencing congestion that creates persistent DART patterns.
    """
)


# ── DART distribution ───────────────────────────────────────────────────────
section_header("WHAT DART ACTUALLY LOOKS LIKE", "2015–2025, three representative zones, clipped ±$100")

panel = load_panel()
fig = go.Figure()
zone_colors = [("LONGIL", GREEN), ("NYC", PURPLE), ("WEST", BLUE)]
for zone, color in zone_colors:
    darts = panel[panel["zone"] == zone]["dart"].clip(-100, 100)
    fig.add_trace(go.Histogram(
        x=darts, name=zone, opacity=0.55, nbinsx=200,
        marker=dict(color=color, line=dict(color=BG, width=0.2)),
        histnorm="probability density",
    ))

fig.add_vline(x=5,  line=dict(color=GREEN, width=1.2, dash="dash"),
              annotation=dict(text="γ_pos = +$5", font=dict(color=GREEN, size=10)),
              annotation_position="top")
fig.add_vline(x=-30, line=dict(color=AMBER, width=1.2, dash="dash"),
              annotation=dict(text="γ_neg = −$30", font=dict(color=AMBER, size=10)),
              annotation_position="top")
fig = _terminal_layout(fig, height=380, title=None)
fig.update_layout(barmode="overlay")
fig.update_xaxes(title="DART ($/MWh)")
fig.update_yaxes(title="DENSITY")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    '<div style="font-size:0.72rem; color:#586069;">'
    'LONGIL has the heaviest tails — import constraints create the most extreme spreads. '
    'The dashed lines mark the spike thresholds where labels are defined.'
    '</div>',
    unsafe_allow_html=True,
)


# ── One trade, end to end ───────────────────────────────────────────────────
section_header("ONE TRADE, END TO END")

timeline = [
    ("D-1 04:59 ET", "Feature vector for hour t built from settled data: zone load forecasts, "
                     "DART lag-48h and lag-72h, lag-48h forecast errors, calendar, past-spike counts. 52 numbers."),
    ("D-1 04:59 ET", "StandardScaler (fitted on 2015–2019 only) normalizes. Out-of-distribution "
                     "values (e.g. polar-vortex DART) become large z-scores."),
    ("D-1 04:59 ET", "All 22 logistic regression models run in parallel. Each outputs one probability."),
    ("D-1 04:59 ET", "Decision: for each eligible (zone, side), if p ≥ τ (validation-tuned), prepare a bid."),
    ("D-1 05:00 ET", "NYISO DAM gate closes. Our bids are locked in."),
    ("D-1 ~11:00 ET", "NYISO publishes DA LMPs for day D. Virtual bids clear at the zonal DA price."),
    ("D  hour t", "NYISO dispatches generators every 5 minutes. Hourly avg RT LMP publishes shortly after."),
    ("D  ~t+1:05", "Settlement: P&L = (DA − RT) × ±1 MWh.  Done."),
]

timeline_html = '<div style="background:' + BG_PANEL + '; border:1px solid #e1e4e8; ' \
                'border-radius:4px; padding:14px 18px; font-family: JetBrains Mono, monospace;">'
for i, (ts, desc) in enumerate(timeline):
    dot_color = GREEN if i < 5 else AMBER if i < 7 else BLUE
    timeline_html += (
        f'<div style="display:grid; grid-template-columns:140px 12px 1fr; gap:12px; '
        f'padding:8px 0; border-bottom:1px dotted #e1e4e8;">'
        f'<span style="color:{TEXT_DIM}; font-size:0.78rem;">{ts}</span>'
        f'<span style="display:inline-block; width:8px; height:8px; border-radius:50%; '
        f'background:{dot_color}; box-shadow:0 0 6px {dot_color}; margin-top:6px;"></span>'
        f'<span style="color:{TEXT}; font-size:0.82rem; line-height:1.5;">{desc}</span>'
        f'</div>'
    )
timeline_html += '</div>'
st.markdown(timeline_html, unsafe_allow_html=True)


# ── Largest trades ──────────────────────────────────────────────────────────
section_header("LARGEST HISTORICAL TRADES", "top 10 winners and worst 10 losers, side by side")

trades = load_trades()
left, right = st.columns(2)

with left:
    st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                'letter-spacing:0.1em; margin-bottom:4px;">TOP 10 WINNERS</div>',
                unsafe_allow_html=True)
    top = trades.nlargest(10, "payoff")
    rows = [{"ts": r["interval_start_local"], "zone": r["zone"], "side": r["side"],
             "proba": r["proba"], "dart": r["dart"], "payoff": r["payoff"]}
            for _, r in top.iterrows()]
    trade_tape(rows)

with right:
    st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                'letter-spacing:0.1em; margin-bottom:4px;">WORST 10 LOSERS</div>',
                unsafe_allow_html=True)
    worst = trades.nsmallest(10, "payoff")
    rows = [{"ts": r["interval_start_local"], "zone": r["zone"], "side": r["side"],
             "proba": r["proba"], "dart": r["dart"], "payoff": r["payoff"]}
            for _, r in worst.iterrows()]
    trade_tape(rows)

st.markdown(
    '<div style="font-size:0.72rem; color:#586069; margin-top:10px;">'
    'Even the largest losers come from genuine market events — RT prices spiking '
    'above DA when the model predicted a DEC. Right-skewed payoff is the design: '
    'capped left tail, uncapped right tail.'
    '</div>',
    unsafe_allow_html=True,
)
