"""
Plotly figure builders — light theme.

All charts share a consistent palette: DEC (positive DART side) = green,
INC (negative DART side) = amber, total/equity = primary blue. Drawdowns
shaded in muted red. Same function signatures throughout so pages don't change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.theme import (
    GREEN, GREEN_DIM, RED, RED_DIM, AMBER, BLUE, PURPLE, PRIMARY_BLUE,
    TEXT, TEXT_DIM, TEXT_FAINT, GRID, BG, BG_PANEL,
)

# Legacy palette names for compatibility
PRIMARY   = PRIMARY_BLUE
DEC_COLOR = GREEN
INC_COLOR = AMBER          # amber for INC so green/red is reserved for win/loss
GREY      = TEXT_FAINT
DRAWDOWN  = RED
ACCENT    = PURPLE


def _light_layout(fig: go.Figure, height: int = 380,
                  title: str | None = None) -> go.Figure:
    """Common light-theme layout."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color=TEXT_DIM),
            x=0, xanchor="left", y=0.97,
        ) if title else None,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=11),
        margin=dict(l=8, r=12, t=36 if title else 12, b=8),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=BG_PANEL, bordercolor=TEXT_FAINT,
            font=dict(color=TEXT, size=11),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=TEXT_DIM, size=10),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=GRID,
        linecolor=TEXT_FAINT, tickfont=dict(color=TEXT_DIM, size=10),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=TEXT_FAINT, zerolinewidth=1,
        linecolor=TEXT_FAINT, tickfont=dict(color=TEXT_DIM, size=10),
    )
    return fig


# Backwards-compat alias used by older pages
_terminal_layout = _light_layout


def cumulative_pnl_chart(
    cum_pnl: pd.DataFrame,
    title: str = "Equity curve",
    show_drawdown: bool = True,
) -> go.Figure:
    """Cumulative P&L with drawdown shading."""
    fig = go.Figure()

    if show_drawdown:
        running_max = cum_pnl["pnl_total"].cummax()
        fig.add_trace(go.Scatter(
            x=cum_pnl.index, y=running_max,
            mode="lines", line=dict(color=TEXT_FAINT, width=0.8, dash="dot"),
            name="Peak", hoverinfo="skip", showlegend=False,
        ))
        underwater = cum_pnl["pnl_total"].where(
            cum_pnl["pnl_total"] < running_max, running_max
        )
        fig.add_trace(go.Scatter(
            x=cum_pnl.index, y=underwater,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(239, 83, 80, 0.15)",
            name="Drawdown", hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl["pnl_total"],
        mode="lines", line=dict(color=PRIMARY_BLUE, width=2.2),
        name="P&L",
        hovertemplate="%{x|%b %d, %Y %H:%M}<br>$%{y:,.0f}<extra></extra>",
    ))

    fig = _light_layout(fig, height=420, title=title)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(showlegend=False)
    return fig


def cumulative_pnl_split_chart(cum_pnl: pd.DataFrame,
                               title: str = "DEC vs INC attribution") -> go.Figure:
    """Two lines: DEC (pos) and INC (neg) cumulative attribution."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl["pnl_pos"],
        mode="lines", line=dict(color=GREEN, width=1.8),
        name="DEC (DA > RT)",
        hovertemplate="%{x|%b %d, %Y}<br>DEC: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl["pnl_neg"],
        mode="lines", line=dict(color=AMBER, width=1.8),
        name="INC (RT > DA)",
        hovertemplate="%{x|%b %d, %Y}<br>INC: $%{y:,.0f}<extra></extra>",
    ))
    fig = _light_layout(fig, height=320, title=title)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def zone_attribution_chart(zone_summary: pd.DataFrame,
                           title: str = "P&L by zone / side") -> go.Figure:
    """Horizontal bar chart of P&L by (zone, side)."""
    zs = zone_summary.sort_values("total_pnl", ascending=True).copy()
    labels = zs["zone"] + " · " + zs["side"].str.upper()
    colors = [GREEN if s == "pos" else AMBER for s in zs["side"]]

    fig = go.Figure(go.Bar(
        x=zs["total_pnl"], y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        hovertemplate="%{y}<br>$%{x:,.0f}<br>%{customdata[0]:,} trades<extra></extra>",
        customdata=zs[["n_trades"]].values,
    ))
    fig = _light_layout(fig, height=max(360, 22 * len(labels) + 40), title=title)
    fig.add_vline(x=0, line=dict(color=TEXT_DIM, width=1))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f", title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(showlegend=False)
    return fig


def monthly_pnl_heatmap(trades: pd.DataFrame,
                        title: str = "Monthly P&L heatmap") -> go.Figure:
    """Year × month heatmap of P&L."""
    df = trades.copy()
    df["ts"] = pd.DatetimeIndex(df["interval_start_local"])
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    pivot = df.pivot_table(index="year", columns="month", values="payoff",
                           aggfunc="sum", fill_value=0)
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot[sorted(pivot.columns)]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=pivot.index,
        colorscale=[
            [0.0, "#b71c1c"], [0.35, "#ef5350"], [0.5, "#ffffff"],
            [0.65, "#81c784"], [1.0, "#1b5e20"],
        ],
        zmid=0,
        colorbar=dict(
            title=dict(text="USD", font=dict(color=TEXT_DIM, size=10)),
            tickfont=dict(color=TEXT_DIM, size=10),
            outlinewidth=0,
        ),
        hovertemplate="%{y} %{x}<br>$%{z:,.0f}<extra></extra>",
    ))
    fig = _light_layout(fig, height=300, title=title)
    fig.update_yaxes(tickmode="array", tickvals=list(pivot.index), autorange="reversed")
    return fig


def precision_recall_scatter(pr_df: pd.DataFrame,
                             title: str = "Precision vs Recall by zone") -> go.Figure:
    """Bubble chart of precision against recall, sized by support."""
    df = pr_df[pr_df["eligible"] == True].dropna(subset=["precision"]).copy()
    df["label"] = df["zone"] + "/" + df["side"]
    df["color"] = df["side"].map({"pos": GREEN, "neg": AMBER})

    fig = go.Figure(go.Scatter(
        x=df["recall"], y=df["precision"],
        mode="markers+text",
        marker=dict(
            size=np.sqrt(df["TP"] + df["FP"]) * 2.5,
            color=df["color"],
            opacity=0.75,
            line=dict(color="white", width=1.5),
        ),
        text=df["label"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_DIM),
        hovertemplate="%{text}<br>Precision: %{y:.1%}<br>Recall: %{x:.1%}<extra></extra>",
    ))
    fig = _light_layout(fig, height=420, title=title)
    fig.update_xaxes(tickformat=".0%", title="Recall")
    fig.update_yaxes(tickformat=".0%", title="Precision")
    return fig


def yoy_comparison_chart(
    hist_trades: pd.DataFrame,
    live_trades: pd.DataFrame | None,
    cap_doy: int = 135,
    title: str = "Year-over-year · Jan 1 → May 15 cumulative P&L",
) -> go.Figure:
    """Year-over-year Jan→May 15 cumulative P&L comparison."""
    colors_yr = {
        2022: "#e57373", 2023: "#ffb74d", 2024: "#aed581",
        2025: "#64b5f6", 2026: PURPLE,
    }
    fig = go.Figure()

    for year in [2022, 2023, 2024, 2025]:
        y_tr = hist_trades[hist_trades["interval_start_local"].dt.year == year].copy()
        if y_tr.empty:
            continue
        y_tr["doy"] = y_tr["interval_start_local"].dt.day_of_year
        cum = y_tr.groupby("doy")["payoff"].sum().cumsum()
        cum = cum[cum.index <= cap_doy]
        final = float(cum.iloc[-1]) if len(cum) else 0
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", line=dict(color=colors_yr[year], width=1.5),
            name=f"{year}  ${final:,.0f}",
            hovertemplate=f"{year} · Day %{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))

    if live_trades is not None and not live_trades.empty:
        lt = live_trades.copy()
        lt["doy"] = lt["interval_start_local"].dt.day_of_year
        cum26 = lt.groupby("doy")["payoff"].sum().cumsum()
        final = float(cum26.iloc[-1])
        fig.add_trace(go.Scatter(
            x=cum26.index, y=cum26.values,
            mode="lines", line=dict(color=PURPLE, width=3.0),
            name=f"2026 LIVE  ${final:,.0f}",
            hovertemplate="2026 LIVE · Day %{x}<br>$%{y:,.0f}<extra></extra>",
        ))

    fig.add_vline(x=cap_doy, line=dict(color=TEXT_FAINT, width=0.8, dash="dash"),
                  annotation=dict(text="May 15", font=dict(color=TEXT_DIM, size=10)),
                  annotation_position="top")
    fig.add_hline(y=0, line=dict(color=TEXT_FAINT, width=0.6))
    fig = _light_layout(fig, height=420, title=title)
    fig.update_xaxes(title="Day of year")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def trade_payoff_distribution(trades: pd.DataFrame,
                              title: str = "Trade payoff distribution") -> go.Figure:
    """Histogram of trade payoffs with win/loss color split."""
    payoffs = trades["payoff"].clip(-500, 500)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=payoffs[payoffs >= 0], name="Winning trades",
        marker=dict(color=GREEN, line=dict(color="white", width=0.3)),
        opacity=0.8, nbinsx=80,
    ))
    fig.add_trace(go.Histogram(
        x=payoffs[payoffs < 0], name="Losing trades",
        marker=dict(color=RED, line=dict(color="white", width=0.3)),
        opacity=0.8, nbinsx=40,
    ))
    fig.add_vline(x=0, line=dict(color=TEXT_DIM, width=1))
    fig = _light_layout(fig, height=320, title=title)
    fig.update_xaxes(title="Payoff per trade (USD, clipped ±$500)")
    fig.update_yaxes(title="Count")
    fig.update_layout(bargap=0.04, barmode="overlay")
    return fig


def yearly_pnl_bar(trades: pd.DataFrame, title: str = "P&L by year") -> go.Figure:
    """Year-over-year bars."""
    yr = trades.groupby(trades["interval_start_local"].dt.year)["payoff"].agg(
        ["sum", "count", lambda s: (s > 0).mean()]
    )
    yr.columns = ["pnl", "n", "wr"]
    yr = yr.reset_index().rename(columns={"interval_start_local": "year"})
    colors = [GREEN if v >= 0 else RED for v in yr["pnl"]]
    fig = go.Figure(go.Bar(
        x=yr["year"], y=yr["pnl"],
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"${v:,.0f}" for v in yr["pnl"]],
        textfont=dict(color=TEXT, size=11),
        textposition="outside",
        customdata=yr[["n", "wr"]].values,
        hovertemplate="%{x}<br>P&L: $%{y:,.0f}<br>Trades: %{customdata[0]:,}<br>Win rate: %{customdata[1]:.1%}<extra></extra>",
    ))
    fig = _light_layout(fig, height=300, title=title)
    fig.add_hline(y=0, line=dict(color=TEXT_DIM, width=0.5))
    fig.update_xaxes(title=None, tickmode="array",
                     tickvals=yr["year"].tolist())
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title=None)
    fig.update_layout(showlegend=False)
    return fig
