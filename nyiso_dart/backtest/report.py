"""
Reporting: precision/recall tables, cumulative P&L plots, safe-vs-naive comparison.

Reads:
  results/<variant>/trades.parquet
  results/<variant>/zone_summary.parquet
  results/<variant>/cumulative_pnl.parquet
  data/features/predictions_<variant>.parquet  (for full precision/recall)
  data/features/y.parquet                       (for full precision/recall)
  models/thresholds_<variant>.parquet           (for eligibility)
  data/processed/panel.parquet                  (for DART)

Writes:
  results/<variant>/precision_recall.csv        per-(zone,side) classification metrics
  results/<variant>/yearly_pnl.csv              per-(zone,side,year) P&L grid
  results/<variant>/figures/cumulative_pnl.png  (total, pos, neg)
  results/<variant>/figures/zone_pnl_bar.png
  results/comparison_safe_vs_naive.csv          (the bias-impact headline)
  results/figures/comparison_safe_vs_naive.png  (overlay of cumulative curves)

Usage
-----
    python -m nyiso_dart.backtest.report
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
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
)
from nyiso_dart.models.splits import test_mask

log = logging.getLogger(__name__)

VARIANTS = ("safe", "naive")


# ---------------------------------------------------------------------------
# Precision / recall over the test period, per (zone, side)
# ---------------------------------------------------------------------------
def precision_recall_table(variant: str) -> pd.DataFrame:
    """For each (zone, side): precision, recall, F1, TP/FP/FN/TN on test period."""
    preds = pd.read_parquet(FEATURES_DIR / f"predictions_{variant}.parquet")
    y = pd.read_parquet(FEATURES_DIR / "y.parquet")
    thr = pd.read_parquet(MODELS_DIR / f"thresholds_{variant}.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    dart_wide = panel.pivot(
        index="interval_start_local", columns="zone", values="dart"
    )[ZONES]

    if not preds.index.equals(y.index):
        common = preds.index.intersection(y.index)
        preds = preds.loc[common]; y = y.loc[common]
    te = test_mask(preds.index)

    rows = []
    for _, t in thr.iterrows():
        zone, side = t["zone"], t["side"]
        if not bool(t["eligible"]) or pd.isna(t["best_tau"]):
            tau = None
        else:
            tau = float(t["best_tau"])
        label = f"{zone}_{side}"

        p = preds[label]
        y_true = y[label]

        ok = te.values & p.notna().values & y_true.notna().values
        p_te, y_te = p[ok], y_true[ok].astype(int)
        n = int(len(p_te))
        support_pos = int(y_te.sum())
        if tau is None:
            rows.append({"zone": zone, "side": side, "tau": np.nan, "eligible": False,
                         "TP": 0, "FP": 0, "FN": support_pos, "TN": n - support_pos,
                         "precision": np.nan, "recall": 0.0, "f1": 0.0,
                         "support_pos": support_pos, "n": n})
            continue

        pred = (p_te >= tau).astype(int)
        tp = int(((pred == 1) & (y_te == 1)).sum())
        fp = int(((pred == 1) & (y_te == 0)).sum())
        fn = int(((pred == 0) & (y_te == 1)).sum())
        tn = int(((pred == 0) & (y_te == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and not np.isnan(precision) else 0.0)
        rows.append({"zone": zone, "side": side, "tau": tau, "eligible": True,
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": precision, "recall": recall, "f1": f1,
                     "support_pos": support_pos, "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Yearly P&L table, per (zone, side, year)
# ---------------------------------------------------------------------------
def yearly_pnl_table(variant: str) -> pd.DataFrame:
    p = RESULTS_DIR / variant / "zone_year_summary.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    pivot = df.pivot_table(
        index=["zone", "side"],
        columns="year",
        values="total_pnl",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    return pivot.sort_values("TOTAL", ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_cumulative(variant: str) -> Path:
    cum = pd.read_parquet(RESULTS_DIR / variant / "cumulative_pnl.parquet")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum.index, cum["pnl_total"], label="Total", linewidth=1.6, color="#1f4e79")
    ax.plot(cum.index, cum["pnl_pos"], label="DEC (pos side)", linewidth=1.0,
            color="#2e7d32", alpha=0.85)
    ax.plot(cum.index, cum["pnl_neg"], label="INC (neg side)", linewidth=1.0,
            color="#c62828", alpha=0.85)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"Cumulative P&L on test 2022-2025  --  variant: {variant}")
    ax.set_xlabel("operating hour")
    ax.set_ylabel("USD (unit-size, 1 MWh per trade)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = RESULTS_DIR / variant / "figures" / "cumulative_pnl.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_zone_pnl_bar(variant: str) -> Path:
    zs = pd.read_parquet(RESULTS_DIR / variant / "zone_summary.parquet")
    zs = zs.sort_values("total_pnl", ascending=True)
    labels = zs["zone"] + " / " + zs["side"]
    colors = ["#2e7d32" if s == "pos" else "#c62828" for s in zs["side"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, zs["total_pnl"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(f"Per-(zone, side) total test P&L  --  variant: {variant}")
    ax.set_xlabel("USD over 2022-2025 (unit size)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / variant / "figures" / "zone_pnl_bar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_safe_vs_naive() -> Path:
    safe = pd.read_parquet(RESULTS_DIR / "safe" / "cumulative_pnl.parquet")
    naive = pd.read_parquet(RESULTS_DIR / "naive" / "cumulative_pnl.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax = axes[0]
    ax.plot(safe.index, safe["pnl_total"], label="SAFE (no leak)",
            linewidth=1.6, color="#1f4e79")
    ax.plot(naive.index, naive["pnl_total"], label="NAIVE (literal lag, 83% leak)",
            linewidth=1.6, color="#b71c1c")
    ax.set_title("Total cumulative P&L: bias-resistant vs literal-lag replication")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3); ax.legend(loc="upper left")

    delta = naive["pnl_total"] - safe["pnl_total"]
    ax2 = axes[1]
    ax2.plot(delta.index, delta, color="#6a1b9a", linewidth=1.4)
    ax2.fill_between(delta.index, 0, delta, alpha=0.15, color="#6a1b9a")
    ax2.set_title("Look-ahead inflation: NAIVE minus SAFE cumulative P&L")
    ax2.set_ylabel("USD inflation"); ax2.set_xlabel("operating hour")
    ax2.grid(alpha=0.3); ax2.axhline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    out = RESULTS_DIR / "figures" / "comparison_safe_vs_naive.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def comparison_table() -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        zs = pd.read_parquet(RESULTS_DIR / v / "zone_summary.parquet")
        total = zs["total_pnl"].sum()
        n_trades = zs["n_trades"].sum()
        n_eligible = int((zs["n_trades"] > 0).sum())
        pos_pnl = float(zs.loc[zs["side"] == "pos", "total_pnl"].sum())
        neg_pnl = float(zs.loc[zs["side"] == "neg", "total_pnl"].sum())
        rows.append({"variant": v, "total_pnl": float(total), "n_trades": int(n_trades),
                     "eligible_pairs_traded": n_eligible,
                     "pos_pnl": pos_pnl, "neg_pnl": neg_pnl})
    df = pd.DataFrame(rows)
    safe_total = df.loc[df["variant"] == "safe", "total_pnl"].iloc[0]
    naive_total = df.loc[df["variant"] == "naive", "total_pnl"].iloc[0]
    df["pct_vs_safe"] = df["total_pnl"] / safe_total
    df.loc[df["variant"] == "naive", "leak_inflation_x"] = naive_total / safe_total
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description="Generate report tables and plots")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    for v in args.variants:
        log.info("[%s] building precision/recall table...", v)
        pr = precision_recall_table(v)
        pr_path = RESULTS_DIR / v / "precision_recall.csv"
        pr.to_csv(pr_path, index=False)
        log.info("  wrote %s", pr_path)

        log.info("[%s] building yearly P&L table...", v)
        yr = yearly_pnl_table(v)
        if not yr.empty:
            yr_path = RESULTS_DIR / v / "yearly_pnl.csv"
            yr.to_csv(yr_path)
            log.info("  wrote %s", yr_path)

        log.info("[%s] plotting cumulative P&L...", v)
        out = plot_cumulative(v); log.info("  wrote %s", out)
        out = plot_zone_pnl_bar(v); log.info("  wrote %s", out)

    log.info("Building safe vs naive comparison...")
    comp = comparison_table()
    comp_path = RESULTS_DIR / "comparison_safe_vs_naive.csv"
    comp.to_csv(comp_path, index=False)
    log.info("  wrote %s", comp_path)
    out = plot_safe_vs_naive(); log.info("  wrote %s", out)

    print("\n=== SAFE vs NAIVE -- the audit-risk headline ===")
    print(comp.to_string(index=False))

    print("\n=== SAFE precision/recall (test 2022-2025) ===")
    pr_safe = pd.read_csv(RESULTS_DIR / "safe" / "precision_recall.csv")
    pr_safe = pr_safe.sort_values("precision", ascending=False, na_position="last")
    fmt_cols = ["zone", "side", "tau", "eligible", "TP", "FP", "FN",
                "precision", "recall", "f1", "support_pos", "n"]
    print(pr_safe[fmt_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== SAFE yearly P&L by (zone, side) ===")
    yr_safe = pd.read_csv(RESULTS_DIR / "safe" / "yearly_pnl.csv")
    print(yr_safe.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
