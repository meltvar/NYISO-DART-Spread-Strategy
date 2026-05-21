"""
Out-of-sample backtest on the test period 2022-2025.

This module locks all policy parameters from earlier stages and applies them
to test rows. It does NOT recompute thresholds, eligibility, or any model
coefficients. Re-running it gives identical results.

Policy applied per (variant, zone, side):
  - If (zone, side) is INELIGIBLE per validation tuning -> no trades placed.
  - Else trigger trade when predicted_proba >= best_tau (from validation).
  - Trade size: 1 MWh per triggered signal (unit-size benchmark).
  - Payoff per trade:
      pos side (DEC trade)  =  +DART[t, z]
      neg side (INC trade)  =  -DART[t, z]

Outputs
-------
results/<variant>/trades.parquet
    One row per executed trade, columns: interval_start_local, zone, side,
    proba, dart, payoff, year, month, season.

results/<variant>/zone_summary.parquet
    Per (zone, side, year) totals: trades, total_pnl, avg_pnl,
    precision (= fraction of trades that coincide with a realized spike).

results/<variant>/cumulative_pnl.parquet
    Hourly cumulative P&L for total and per-side, indexed by
    interval_start_local. Used by reporting to draw the equity curves.

results/<variant>/metrics.json
    Aggregate test-period numbers: total P&L, P&L by year, by side,
    precision/recall vectors.

Bias guarantee
--------------
The backtest reads thresholds_<variant>.parquet (validation-tuned) and applies
them to the test period only. It never re-fits, never re-tunes, never inspects
multiple thresholds on test data.

Usage
-----
    python -m nyiso_dart.backtest.run
    python -m nyiso_dart.backtest.run --variant safe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nyiso_dart.config import (
    FEATURES_DIR,
    GAMMA_NEG,
    GAMMA_POS,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ZONES,
    season_of,
)
from nyiso_dart.models.splits import test_mask

log = logging.getLogger(__name__)

VARIANTS = ("safe", "naive")
SIDES = ("pos", "neg")


def _load(variant: str):
    preds = pd.read_parquet(FEATURES_DIR / f"predictions_{variant}.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    thresholds = pd.read_parquet(MODELS_DIR / f"thresholds_{variant}.parquet")
    dart_wide = panel.pivot(
        index="interval_start_local", columns="zone", values="dart"
    )[ZONES]
    return preds, dart_wide, thresholds


def _collect_trades(
    preds: pd.DataFrame,
    dart_wide: pd.DataFrame,
    thresholds: pd.DataFrame,
    te: pd.Series,
) -> pd.DataFrame:
    rows = []
    elig = thresholds[thresholds["eligible"] == True]  # noqa: E712

    for _, row in elig.iterrows():
        zone, side, tau = row["zone"], row["side"], float(row["best_tau"])
        label = f"{zone}_{side}"
        p = preds[label]
        d = dart_wide[zone]

        ok = te.values & p.notna().values & d.notna().values
        trigger = (p >= tau).values & ok
        if not trigger.any():
            continue
        idx = preds.index[trigger]
        payoff = d.loc[idx] if side == "pos" else -d.loc[idx]
        rows.append(
            pd.DataFrame({
                "interval_start_local": idx,
                "zone": zone,
                "side": side,
                "proba": p.loc[idx].values,
                "dart": d.loc[idx].values,
                "payoff": payoff.values,
            })
        )

    if not rows:
        return pd.DataFrame(
            columns=["interval_start_local", "zone", "side", "proba", "dart", "payoff"]
        )
    trades = pd.concat(rows, ignore_index=True)
    trades["year"] = trades["interval_start_local"].dt.year
    trades["month"] = trades["interval_start_local"].dt.month
    trades["season"] = trades["month"].map(season_of)
    trades["hour"] = trades["interval_start_local"].dt.hour
    return trades.sort_values("interval_start_local").reset_index(drop=True)


def _precision(trades: pd.DataFrame) -> pd.DataFrame:
    """For each (zone, side) compute the realized-spike precision.

    A pos-side trade is "correct" if dart >= GAMMA_POS.
    A neg-side trade is "correct" if dart <= -GAMMA_NEG.
    """
    if trades.empty:
        return pd.DataFrame(columns=["zone", "side", "n_trades", "n_spikes",
                                     "precision", "total_pnl", "avg_pnl"])
    pos_mask = trades["side"] == "pos"
    correct = np.where(
        pos_mask,
        trades["dart"] >= GAMMA_POS,
        trades["dart"] <= -GAMMA_NEG,
    )
    trades = trades.assign(correct=correct.astype(bool))
    g = trades.groupby(["zone", "side"], as_index=False).agg(
        n_trades=("payoff", "size"),
        n_spikes=("correct", "sum"),
        total_pnl=("payoff", "sum"),
        avg_pnl=("payoff", "mean"),
    )
    g["precision"] = g["n_spikes"] / g["n_trades"]
    return g[["zone", "side", "n_trades", "n_spikes", "precision",
              "total_pnl", "avg_pnl"]]


def _zone_year_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby(["zone", "side", "year"], as_index=False).agg(
        n_trades=("payoff", "size"),
        total_pnl=("payoff", "sum"),
        avg_pnl=("payoff", "mean"),
    )


def _cumulative(trades: pd.DataFrame, all_test_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly cumulative P&L for total and per-side."""
    cum = pd.DataFrame(index=all_test_idx)
    cum["pnl_total"] = 0.0
    cum["pnl_pos"] = 0.0
    cum["pnl_neg"] = 0.0
    if trades.empty:
        return cum.cumsum()

    pnl_by_hour = trades.groupby(["interval_start_local", "side"])["payoff"].sum().unstack(
        "side", fill_value=0.0
    )
    pnl_by_hour = pnl_by_hour.reindex(all_test_idx).fillna(0.0)
    cum["pnl_pos"] = pnl_by_hour.get("pos", 0.0)
    cum["pnl_neg"] = pnl_by_hour.get("neg", 0.0)
    cum["pnl_total"] = cum["pnl_pos"] + cum["pnl_neg"]
    return cum.cumsum()


def run_variant(variant: str) -> dict:
    preds, dart_wide, thresholds = _load(variant)
    if not preds.index.equals(dart_wide.index):
        common = preds.index.intersection(dart_wide.index)
        preds = preds.loc[common]
        dart_wide = dart_wide.loc[common]
    te = test_mask(preds.index)
    log.info("[%s] test rows: %d", variant, int(te.sum()))

    trades = _collect_trades(preds, dart_wide, thresholds, te)
    log.info("[%s] executed trades: %d", variant, len(trades))

    prec = _precision(trades)
    yr = _zone_year_summary(trades)
    cum = _cumulative(trades, preds.index[te])

    out_dir = RESULTS_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_dir / "trades.parquet", index=False)
    prec.to_parquet(out_dir / "zone_summary.parquet", index=False)
    if not yr.empty:
        yr.to_parquet(out_dir / "zone_year_summary.parquet", index=False)
    cum.to_parquet(out_dir / "cumulative_pnl.parquet")

    total_pnl = float(trades["payoff"].sum()) if not trades.empty else 0.0
    pos_pnl = float(trades.loc[trades["side"] == "pos", "payoff"].sum()) if not trades.empty else 0.0
    neg_pnl = float(trades.loc[trades["side"] == "neg", "payoff"].sum()) if not trades.empty else 0.0

    metrics = {
        "variant": variant,
        "test_rows": int(te.sum()),
        "n_trades": int(len(trades)),
        "total_pnl": total_pnl,
        "pos_pnl": pos_pnl,
        "neg_pnl": neg_pnl,
        "by_year": (
            trades.groupby("year")["payoff"].sum().to_dict() if not trades.empty else {}
        ),
        "by_zone_side": (
            trades.groupby(["zone", "side"])["payoff"].sum().reset_index().to_dict("records")
            if not trades.empty else []
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def _print_summary(variant: str, metrics: dict) -> None:
    print(f"\n=== {variant} -- TEST 2022-2025 ===")
    print(f"  trades         : {metrics['n_trades']:>10,}")
    print(f"  total P&L      : ${metrics['total_pnl']:>14,.2f}")
    print(f"  pos-side (DEC) : ${metrics['pos_pnl']:>14,.2f}")
    print(f"  neg-side (INC) : ${metrics['neg_pnl']:>14,.2f}")
    by_year = metrics.get("by_year", {})
    if by_year:
        print("  by year:")
        for y, v in sorted(by_year.items()):
            print(f"    {y}: ${float(v):>14,.2f}")


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Out-of-sample backtest on test period")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    summary: dict[str, dict] = {}
    for v in args.variants:
        summary[v] = run_variant(v)
        _print_summary(v, summary[v])

    (RESULTS_DIR / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nSummary written to {RESULTS_DIR / 'backtest_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
