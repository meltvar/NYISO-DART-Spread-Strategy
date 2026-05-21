"""
Validation-set threshold tuning.

For each (variant, zone, side) we choose the probability cutoff `tau` that
maximises cumulative unit-size P&L on the VALIDATION period (2020-2021).

We also apply a zone-eligibility rule: if the best-tau validation P&L per
trade is negative or the total validation P&L is non-positive, the (zone,
side) is marked ineligible and will not trade in the test period. This
prevents zones that don't generalise well from contaminating live results.

This module is the ONLY place tau and eligibility are decided. The backtest
module loads the chosen taus and the eligibility mask from disk; it is
forbidden to recompute either against test data.

P&L convention
--------------
For a row at operating hour t in zone z, with predicted spike probability
p_{t,z,side} and tau cutoff tau_{z,side}:

  side = pos  ->  DEC trade triggers when p >= tau.  Payoff = +DART[t,z].
  side = neg  ->  INC trade triggers when p >= tau.  Payoff = -DART[t,z].

Unit size: 1 MWh per triggered signal (the benchmark before any position
sizing).

Usage
-----
    python -m nyiso_dart.models.thresholds
    python -m nyiso_dart.models.thresholds --variant safe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from nyiso_dart.config import (
    FEATURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TAU_GRID,
    ZONES,
)
from nyiso_dart.models.splits import val_mask

log = logging.getLogger(__name__)

VARIANTS = ("safe", "naive")
SIDES = ("pos", "neg")


def _load(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (predictions, dart_wide, val_mask_series)."""
    preds = pd.read_parquet(FEATURES_DIR / f"predictions_{variant}.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    dart_wide = panel.pivot(
        index="interval_start_local", columns="zone", values="dart"
    )[ZONES]
    # Align indices
    if not preds.index.equals(dart_wide.index):
        common = preds.index.intersection(dart_wide.index)
        preds = preds.loc[common]
        dart_wide = dart_wide.loc[common]
    va = val_mask(preds.index)
    return preds, dart_wide, va


def _pnl_for_tau(
    proba: pd.Series,
    dart: pd.Series,
    side: str,
    tau: float,
) -> tuple[float, int, float]:
    """Total P&L, trade count, average per-trade P&L for a single (zone, side, tau)."""
    trade = proba >= tau
    n = int(trade.sum())
    if n == 0:
        return 0.0, 0, 0.0
    if side == "pos":
        payoff = dart[trade]
    else:  # neg
        payoff = -dart[trade]
    total = float(payoff.sum())
    avg = float(payoff.mean())
    return total, n, avg


def tune(variant: str) -> dict:
    preds, dart_wide, va = _load(variant)
    log.info("[%s] preds shape=%s val rows=%d", variant, preds.shape, int(va.sum()))

    results = []
    for zone in ZONES:
        for side in SIDES:
            label = f"{zone}_{side}"
            p_full = preds[label]
            d_full = dart_wide[zone]
            # Restrict to validation period AND rows with non-null p and dart.
            row_ok = va.values & p_full.notna().values & d_full.notna().values
            p_val = p_full[row_ok]
            d_val = d_full[row_ok]
            if len(p_val) == 0:
                results.append(
                    {"zone": zone, "side": side, "best_tau": None,
                     "val_pnl": 0.0, "val_trades": 0, "val_avg": 0.0,
                     "eligible": False, "reason": "no validation rows"}
                )
                continue

            best = (-np.inf, None, 0, 0.0)
            for tau in TAU_GRID:
                total, n, avg = _pnl_for_tau(p_val, d_val, side, float(tau))
                if total > best[0]:
                    best = (total, float(tau), n, avg)
            total, tau, n, avg = best
            eligible = (tau is not None) and (n > 0) and (total > 0) and (avg > 0)
            reason = "ok" if eligible else (
                "no trades" if n == 0
                else "negative total P&L" if total <= 0
                else "negative avg P&L"
            )
            results.append(
                {"zone": zone, "side": side, "best_tau": tau,
                 "val_pnl": total, "val_trades": n, "val_avg": avg,
                 "eligible": eligible, "reason": reason}
            )

    df = pd.DataFrame(results)
    return {"variant": variant, "table": df}


def _print_table(variant: str, df: pd.DataFrame) -> None:
    print(f"\n=== {variant} -- validation-tuned thresholds ===")
    cols = ["zone", "side", "best_tau", "val_pnl", "val_trades", "val_avg",
            "eligible", "reason"]
    fmt = df[cols].copy()
    fmt["val_pnl"] = fmt["val_pnl"].map(lambda x: f"{x:>12,.2f}")
    fmt["val_avg"] = fmt["val_avg"].map(lambda x: f"{x:>8,.2f}")
    fmt["best_tau"] = fmt["best_tau"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "-"
    )
    print(fmt.to_string(index=False))
    n_elig = int(df["eligible"].sum())
    print(f"\nEligible (zone, side) pairs: {n_elig}/22")


def _save(variant: str, df: pd.DataFrame) -> None:
    path = MODELS_DIR / f"thresholds_{variant}.parquet"
    df.to_parquet(path, index=False)
    log.info("[%s] wrote %s", variant, path)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Tune per-(zone,side) tau on validation set")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    summary: dict[str, list[dict]] = {}
    for v in args.variants:
        out = tune(v)
        _save(v, out["table"])
        _print_table(v, out["table"])
        summary[v] = out["table"].to_dict("records")

    (MODELS_DIR / "thresholds_manifest.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
