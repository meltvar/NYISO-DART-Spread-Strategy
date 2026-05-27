"""
Statistics calculations shared across pages.

Pure functions only — no Streamlit dependencies, fully testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeStats:
    n_trades: int
    total_pnl: float
    annualized_pnl: float
    win_rate: float
    profit_factor: float
    precision: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    sharpe_daily: float
    sharpe_monthly: float
    sortino: float
    max_drawdown: float
    calmar: float
    pos_pnl: float
    neg_pnl: float
    pos_trades: int
    neg_trades: int

    def as_dict(self) -> dict:
        return self.__dict__


def compute_stats(trades: pd.DataFrame, hourly_index: pd.DatetimeIndex | None = None) -> TradeStats:
    """Compute full statistics from a trades dataframe.

    `trades` must have columns: interval_start_local, payoff, side.
    `hourly_index` is optional; if given, used as the time axis for the
    cumulative P&L (filling zero on hours with no trade) so the Sharpe
    properly reflects idle hours.
    """
    if len(trades) == 0:
        return TradeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    n = len(trades)
    total = float(trades["payoff"].sum())
    wins = trades[trades["payoff"] > 0]["payoff"]
    losses = trades[trades["payoff"] < 0]["payoff"]
    win_rate = len(wins) / n
    pf = wins.sum() / abs(losses.sum()) if len(losses) and abs(losses.sum()) > 0 else float("inf")
    precision = float(trades["correct"].mean()) if "correct" in trades.columns else float("nan")

    if hourly_index is not None:
        hourly_pnl = (
            trades.groupby("interval_start_local")["payoff"]
            .sum().reindex(hourly_index).fillna(0.0)
        )
        cum_pnl = hourly_pnl.cumsum()
        daily_pnl = hourly_pnl.resample("D").sum()
    else:
        # Fall back to trade-day resampling (less accurate for Sharpe)
        trades_idx = pd.DatetimeIndex(trades["interval_start_local"])
        hourly_pnl = trades.set_index("interval_start_local")["payoff"]
        cum_pnl = hourly_pnl.sort_index().cumsum()
        daily_pnl = hourly_pnl.resample("D").sum()

    sharpe_d = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
                if daily_pnl.std() > 0 else 0.0)

    monthly_pnl = cum_pnl.resample("ME").last().diff().fillna(0)
    sharpe_m = (monthly_pnl.mean() / monthly_pnl.std() * np.sqrt(12)
                if monthly_pnl.std() > 0 else 0.0)

    # Sortino: canonical definition — downside deviation against target=0,
    # i.e. sqrt(mean(min(0, R)^2)) across ALL days (positive days contribute 0).
    # For positive-skew tail strategies this is more honest than Sharpe.
    downside_sq = np.minimum(daily_pnl.values, 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean())) if len(daily_pnl) else 0.0
    sortino = (daily_pnl.mean() / downside_dev * np.sqrt(365)
               if downside_dev > 0 else 0.0)

    running_max = cum_pnl.cummax()
    max_dd = float((cum_pnl - running_max).min())

    # Annualized P&L over the actual time span
    span_days = (cum_pnl.index.max() - cum_pnl.index.min()).days
    years = max(span_days / 365.25, 1/52)
    annualized = total / years
    calmar = abs(annualized / max_dd) if max_dd != 0 else float("inf")

    pos_pnl = float(trades.loc[trades["side"] == "pos", "payoff"].sum())
    neg_pnl = float(trades.loc[trades["side"] == "neg", "payoff"].sum())

    return TradeStats(
        n_trades=n,
        total_pnl=total,
        annualized_pnl=annualized,
        win_rate=win_rate,
        profit_factor=pf,
        precision=precision,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        best_trade=float(trades["payoff"].max()),
        worst_trade=float(trades["payoff"].min()),
        sharpe_daily=sharpe_d,
        sharpe_monthly=sharpe_m,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        pos_pnl=pos_pnl,
        neg_pnl=neg_pnl,
        pos_trades=int((trades["side"] == "pos").sum()),
        neg_trades=int((trades["side"] == "neg").sum()),
    )


def format_money(x: float, decimals: int = 0) -> str:
    """Format dollars with sign and commas."""
    if pd.isna(x):
        return "—"
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def format_pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.{decimals}f}%"
