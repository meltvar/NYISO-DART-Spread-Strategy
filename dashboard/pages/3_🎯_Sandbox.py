"""Sandbox page — backtest any date range interactively."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import (
    PROJECT, MODELS, FEATS, PROC, ZONES, PRIMARY,
    load_predictions, load_thresholds, load_panel, load_panel_2026,
)
from lib.metrics import compute_stats, format_money, format_pct
from lib.plots import _terminal_layout
from lib.theme import (
    apply_theme, kpi_grid, kpi_tile, status_strip, section_header,
    GREEN, GREEN_DIM, RED, AMBER, TEXT, TEXT_DIM, TEXT_FAINT, BG,
)


st.set_page_config(page_title="Sandbox · NYISO DART", page_icon="🎯", layout="wide")
apply_theme()

st.markdown("<h1>Sandbox  ·  Replay Any Window</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    'Run the locked policy over an arbitrary date range. Models and thresholds never change — '
    'this is the same strategy applied to a window of your choice.'
    '</div>',
    unsafe_allow_html=True,
)


# ── User input ───────────────────────────────────────────────────────────────
TZ = "America/New_York"
HIST_END = pd.Timestamp("2025-12-31 23:00:00", tz=TZ)
LIVE_END = pd.Timestamp("2026-05-15 23:00:00", tz=TZ)
MIN_DATE = pd.Timestamp("2015-01-03", tz=TZ)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    start_date = st.date_input(
        "Start date",
        value=pd.Timestamp("2024-01-01").date(),
        min_value=MIN_DATE.date(),
        max_value=LIVE_END.date(),
    )
with col2:
    end_date = st.date_input(
        "End date",
        value=pd.Timestamp("2024-12-31").date(),
        min_value=MIN_DATE.date(),
        max_value=LIVE_END.date(),
    )
with col3:
    st.write("")
    st.write("")
    run_btn = st.button("▶  Run backtest", type="primary", use_container_width=True)


t_start = pd.Timestamp(f"{start_date} 00:00:00", tz=TZ)
t_end   = pd.Timestamp(f"{end_date}   23:00:00", tz=TZ)

if t_start >= t_end:
    st.error("Start date must be before end date.")
    st.stop()


# ── Run only when button is pressed (and cache the result via session state) ──
if "sandbox_result" not in st.session_state or run_btn:
    with st.spinner(f"Running backtest from {start_date} to {end_date}..."):
        # Load predictions for the window
        parts_p, parts_d = [], []
        hist_start, hist_end = t_start, min(t_end, HIST_END)
        if hist_start <= hist_end:
            preds_h = load_predictions()
            preds_h = preds_h[(preds_h.index >= hist_start) & (preds_h.index <= hist_end)]
            panel_h = load_panel()
            dart_h = panel_h.pivot(
                index="interval_start_local", columns="zone", values="dart"
            )[ZONES]
            dart_h = dart_h[(dart_h.index >= hist_start) & (dart_h.index <= hist_end)]
            parts_p.append(preds_h)
            parts_d.append(dart_h)

        # 2026 portion (if requested)
        live_start = max(t_start, HIST_END + pd.Timedelta(hours=1))
        live_end_clip = min(t_end, LIVE_END)
        if live_start <= live_end_clip:
            panel_l = load_panel_2026()
            if panel_l is not None:
                import joblib
                sys.path.insert(0, str(PROJECT))
                from nyiso_dart.features.build import build_features
                arts = build_features(panel_l)
                X_l = arts[f"X_{PRIMARY}"]
                X_l = X_l[(X_l.index >= live_start) & (X_l.index <= live_end_clip)]
                preds_l = pd.DataFrame(index=X_l.index)
                valid = X_l.notna().all(axis=1)
                for zone in ZONES:
                    for side in ("pos", "neg"):
                        pipe = joblib.load(MODELS / PRIMARY / f"{zone}_{side}.joblib")
                        col = pd.Series(np.nan, index=X_l.index)
                        if valid.any():
                            col[valid] = pipe.predict_proba(X_l[valid].values)[:, 1]
                        preds_l[f"{zone}_{side}"] = col
                dart_l = panel_l.pivot(
                    index="interval_start_local", columns="zone", values="dart"
                )[ZONES]
                dart_l = dart_l[(dart_l.index >= live_start) & (dart_l.index <= live_end_clip)]
                parts_p.append(preds_l)
                parts_d.append(dart_l)

        if not parts_p:
            st.error("No data available for the selected range.")
            st.stop()

        preds = pd.concat(parts_p).sort_index()
        dart_wide = pd.concat(parts_d).sort_index()
        thr = load_thresholds()
        elig = thr[thr["eligible"]].copy()

        rows = []
        for _, r in elig.iterrows():
            zone, side, tau = r["zone"], r["side"], float(r["best_tau"])
            lbl = f"{zone}_{side}"
            if lbl not in preds.columns:
                continue
            p = preds[lbl]
            d = dart_wide[zone] if zone in dart_wide.columns else pd.Series(dtype=float)
            ok = p.notna() & d.notna() & (p >= tau)
            if not ok.any():
                continue
            payoff = d[ok] if side == "pos" else -d[ok]
            for t, pv, dv, py in zip(ok[ok].index, p[ok], d[ok], payoff):
                rows.append({
                    "interval_start_local": t, "zone": zone, "side": side,
                    "proba": float(pv), "dart": float(dv), "payoff": float(py),
                    "correct": (float(dv) >= 5 if side == "pos" else float(dv) <= -30),
                })

        trades = (pd.DataFrame(rows).sort_values("interval_start_local").reset_index(drop=True)
                  if rows else pd.DataFrame())
        st.session_state["sandbox_result"] = {
            "trades": trades, "preds": preds, "start": t_start, "end": t_end,
        }

result = st.session_state["sandbox_result"]
trades = result["trades"]
preds  = result["preds"]


# ── Display ─────────────────────────────────────────────────────────────────
if trades.empty:
    st.warning("No trades fired in this window. Try a wider range.")
    st.stop()

stats = compute_stats(trades, preds.index)
n_days = (result["end"] - result["start"]).days + 1

status_strip([
    ("WINDOW", f"{start_date} → {end_date}", "green"),
    ("DAYS", f"{n_days:,}", "green"),
    ("TRADES", f"{stats.n_trades:,}", "green"),
])

section_header("RESULT", f"{start_date} → {end_date}")
pnl_color = "green" if stats.total_pnl > 0 else "red"
kpi_grid([
    kpi_tile("TOTAL P&L", format_money(stats.total_pnl, 0),
             color=pnl_color, accent="green"),
    kpi_tile("ANN. P&L", format_money(stats.annualized_pnl, 0), accent="green"),
    kpi_tile("WIN RATE", format_pct(stats.win_rate, 1),
             color="green", accent="green"),
    kpi_tile("SORTINO", f"{stats.sortino:.2f}",
             color="green" if stats.sortino > 1.0 else None, accent="green"),
    kpi_tile("SHARPE", f"{stats.sharpe_daily:.2f}", accent="dim"),
    kpi_tile("PROFIT FACTOR", f"{stats.profit_factor:.2f}x", accent="green"),
    kpi_tile("MAX DD", format_money(stats.max_drawdown, 0),
             color="red", accent="red"),
    kpi_tile("PRECISION", format_pct(stats.precision, 1), accent="green"),
])


# ── Cumulative P&L plot ─────────────────────────────────────────────────────
section_header("EQUITY CURVE")
hourly_pnl = trades.groupby("interval_start_local")["payoff"].sum().reindex(preds.index).fillna(0.0)
cum = hourly_pnl.cumsum()
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
    line=dict(color=GREEN, width=1.8),
    fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
    hovertemplate="%{x|%b %d, %Y %H:%M}<br>$%{y:,.0f}<extra></extra>",
    showlegend=False,
))
fig.add_hline(y=0, line=dict(color=TEXT_FAINT, width=0.5))
fig = _terminal_layout(fig, height=380, title=None)
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Side-by-side: zone + monthly ────────────────────────────────────────────
section_header("BREAKDOWN")
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                'letter-spacing:0.1em; margin-bottom:4px;">P&L BY ZONE × SIDE</div>',
                unsafe_allow_html=True)
    zsum = (
        trades.groupby(["zone", "side"])
        .agg(n_trades=("payoff", "size"),
             total_pnl=("payoff", "sum"),
             precision=("correct", "mean"))
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )
    zsum["side"] = zsum["side"].str.upper()
    st.dataframe(
        zsum.style.format({
            "n_trades": "{:,.0f}",
            "total_pnl": lambda x: format_money(x, 0),
            "precision": "{:.1%}",
        }),
        use_container_width=True, hide_index=True, height=320,
    )

with col_r:
    st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                'letter-spacing:0.1em; margin-bottom:4px;">MONTHLY P&L</div>',
                unsafe_allow_html=True)
    monthly = trades.groupby(pd.Grouper(key="interval_start_local", freq="ME"))["payoff"].sum()
    fig = go.Figure(go.Bar(
        x=monthly.index, y=monthly.values,
        marker=dict(color=[GREEN if v >= 0 else RED for v in monthly.values],
                    line=dict(color=BG, width=0.5)),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig = _terminal_layout(fig, height=320, title=None)
    fig.add_hline(y=0, line=dict(color=TEXT_FAINT, width=0.5))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Top trades log ──────────────────────────────────────────────────────────
section_header("TOP 15 TRADES (BY |PAYOFF|)")
top = trades.reindex(trades["payoff"].abs().sort_values(ascending=False).index).head(15)
top_display = top[["interval_start_local", "zone", "side", "proba", "dart", "payoff", "correct"]].copy()
top_display["side"] = top_display["side"].str.upper()
top_display.columns = ["Time", "Zone", "Side", "Probability", "DART ($/MWh)", "Payoff ($)", "Spike hit"]
st.dataframe(
    top_display.style.format({
        "Probability":  "{:.1%}",
        "DART ($/MWh)": "{:,.2f}",
        "Payoff ($)":   lambda x: format_money(x, 2),
    }),
    use_container_width=True, hide_index=True,
)
