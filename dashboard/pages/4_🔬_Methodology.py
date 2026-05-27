"""Methodology page — bias prevention, discipline, audit trail."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import load_trades, load_cumulative_pnl
from lib.metrics import compute_stats, format_money
from lib.plots import _terminal_layout
from lib.theme import (
    apply_theme, kpi_grid, kpi_tile, status_strip, section_header,
    GREEN, RED, AMBER, TEXT, TEXT_DIM, TEXT_FAINT, BG,
)


st.set_page_config(page_title="Methodology · NYISO DART", page_icon="🔬", layout="wide")
apply_theme()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1>Methodology  ·  Bias Audit</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#586069; font-size:0.85rem; margin-bottom:14px;">'
    'Every backtest claim comes with discipline. What was done — and what was not.'
    '</div>',
    unsafe_allow_html=True,
)


# ── Split discipline ────────────────────────────────────────────────────────
section_header("TRAIN / VALIDATION / TEST SPLIT", "locked by calendar date")

st.markdown(
    """
    | Period | Window | Role |
    |---|---|---|
    | **Train** | 2015-01-01 → 2019-12-31 | Fits the 22 logistic regression coefficients |
    | **Validation** | 2020-01-01 → 2021-12-31 | Tunes per-zone probability thresholds τ; decides zone eligibility |
    | **Test** | 2022-01-01 → 2025-12-31 | Reported once — never inspected during development |
    | **2026 Live** | 2026-01-01 → present | Genuinely post-deployment; predicted in real time |

    Code enforces this via `nyiso_dart.models.splits.assert_disjoint()`, called
    at every model-fit and threshold-tuning entry point. If the masks ever
    overlap (any code path), the pipeline raises before any data is touched.
    """
)


# ── The safe-vs-naive measurement ───────────────────────────────────────────
section_header("THE LEAKAGE MEASUREMENT", "we ran it both ways, on purpose")

st.markdown(
    """
    The NYISO Day-Ahead Market for operating day **D** closes at **05:00 Eastern on day D−1**.
    Every feature used to predict an operating hour `t` must have knowledge time strictly
    before `gate_closure(t)`.

    A naive implementation uses **D−1 same-clock-hour DART** as the lag-24 feature.
    But D−1 hour-of-day h doesn't actually settle until ~(h+1):00 on D−1 — which for
    every hour h ≥ 4 happens *after* the 05:00 gate. That's a leak.

    Instead of patching it away silently, the pipeline runs **both** specifications
    side by side. The safe variant uses D-2 / D-3 lags which are unconditionally
    settled. The empirical P&L gap quantifies the leakage rather than assuming it away.
    """
)

# Try to load the safe-vs-naive comparison
proj_root = Path(__file__).resolve().parent.parent.parent
csv_path  = proj_root / "results" / "comparison_safe_vs_naive.csv"

if csv_path.exists():
    comp = pd.read_csv(csv_path)
    safe_row  = comp[comp["variant"] == "safe"].iloc[0]
    naive_row = comp[comp["variant"] == "naive"].iloc[0]
    inflation = naive_row["total_pnl"] / safe_row["total_pnl"]

    kpi_grid([
        kpi_tile("SAFE P&L", format_money(safe_row["total_pnl"], 0),
                 sub=f"{int(safe_row['n_trades']):,} trades · LEAK-FREE",
                 color="green", accent="green"),
        kpi_tile("NAIVE P&L", format_money(naive_row["total_pnl"], 0),
                 sub=f"{int(naive_row['n_trades']):,} trades · LEAKY",
                 color="amber", accent="amber"),
        kpi_tile("INFLATION FACTOR", f"{inflation:.2f}x",
                 sub="how much leak inflates P&L",
                 color="red", accent="red"),
    ])
    st.markdown(
        '<div style="font-size:0.78rem; color:#1a1a1a; margin-top:10px; padding:12px; '
        'background:#fafafa; border-left:2px solid #c62828; border-radius:2px;">'
        f'<b style="color:#c62828;">THE LESSON</b><br><br>'
        f'A naive lag specification would have made this strategy look {inflation:.1f}× '
        f'better than it actually is. Most published electricity-trading backtests do '
        f'not measure this. We do — every artifact on the rest of the dashboard uses '
        f'the SAFE variant.'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Run `python -m nyiso_dart.backtest.run` and `python -m nyiso_dart.backtest.report` "
            "to produce the comparison.")


# ── Audit details from manifest ─────────────────────────────────────────────
section_header("AUTOMATED AUDITS", "stamped into data/features/manifest.json at build time")

manifest_path = proj_root / "data" / "features" / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                    'letter-spacing:0.1em;">SAFE MATRIX AUDIT</div>',
                    unsafe_allow_html=True)
        st.json(manifest.get("audit_safe", {}))
    with a2:
        st.markdown('<div style="font-size:0.7rem; color:#586069; text-transform:uppercase; '
                    'letter-spacing:0.1em;">NAIVE MATRIX AUDIT</div>',
                    unsafe_allow_html=True)
        st.json(manifest.get("audit_naive", {}))
else:
    st.info("Manifest not built yet. Run `python -m nyiso_dart.features.build`.")


# ── Threshold tuning discipline ─────────────────────────────────────────────
section_header("THRESHOLD TUNING")

st.markdown(
    """
    The probability cutoff τ is the **only hyperparameter** tuned in this strategy.
    It is selected per (zone, side) on the **2020–2021 validation period only**.

    For each (zone, side), we sweep τ ∈ [0.50, 0.99] in 0.01 steps and pick the value
    that maximizes unit-size P&L on validation (net of $1/MWh transaction cost).
    Eligibility:

    > A (zone, side) is **eligible** to trade on the test period iff its best validation
    > τ produces strictly positive total P&L AND strictly positive avg P&L per trade.

    Zones failing this gate are excluded from the test backtest entirely. The test
    period was **never inspected during this tuning** — all test results come from one
    pipeline call after τ and eligibility are locked.
    """
)


# ── Other biases ────────────────────────────────────────────────────────────
section_header("OTHER BIASES")

st.markdown(
    """
    | Bias | Where it would creep in | Prevention |
    |---|---|---|
    | **Train/test contamination** | Random shuffling, k-fold CV | Calendar-date split, asserted disjoint at every entry |
    | **Cherry-picking** | Reporting only profitable zones | All 11 × 2 reported; eligibility decided mechanically |
    | **DST / timezone bugs** | Spring-forward, fall-back boundaries | UTC storage; tz-aware lag arithmetic with NaT for ambiguous hours |
    | **Threshold data snooping** | Picking τ to fit test results | τ frozen on validation; one-shot test eval |
    | **Scaler leak** | Fitting StandardScaler on all data | Scaler fit on TRAIN rows only |
    | **Forecast vintage leak** | Using latest available DA load forecast | Pick the latest Publish Time *strictly before* gate closure |
    | **Survivorship in zones** | Quietly dropping ineligible zones | Eligibility surfaced in the table; comparison published |
    """
)


# ── What's not in the backtest ──────────────────────────────────────────────
section_header("WHAT'S NOT IN THE BACKTEST", "honest disclosure")

st.markdown(
    """
    - **Transaction costs.** $1/MWh round-trip *is* deducted at threshold tuning and
      backtest (NYISO admin charge). RT uplift charges spike during stressed periods —
      these slightly compress the very largest payoffs, not modeled.
    - **Market impact at scale.** Unit-size (1 MWh) trading is impact-free. Submitting
      hundreds or thousands of MWh would shift the DA clearing price. A structural
      impact model based on DA bid stacks would extend this — Section 4 of the
      reference paper (Hubert et al. 2026).
    - **NYISO LMP revisions.** Occasionally re-published after initial settlement.
      Pipeline uses whichever values gridstatus serves at retrieval time.
    - **Margin opportunity cost.** NYISO requires margin (~$2–5/MWh) for virtual
      bidding capacity. Not modeled as cost-of-capital.
    """
)


# ── Why logistic regression ─────────────────────────────────────────────────
section_header("WHY LOGISTIC REGRESSION")

st.markdown(
    """
    The reference paper (Hubert, Lolas & Sircar 2026) explicitly tested gradient
    boosting and neural networks. They **lost out-of-sample** despite marginal in-sample
    gains. Three reasons:

    1. **Sample efficiency.** Spike events are 2–6% of training rows. Tree/NN models
       overfit the rare-event noise. Logistic with L2 regularization stays honest.
    2. **Interpretability.** Every coefficient is inspectable. A coefficient that jumps
       unexpectedly between training runs is a red flag — visible here, hidden in
       deeper models.
    3. **Stable probabilities.** Logistic outputs survive regime shifts. Flexible models
       produce confidence ratings that don't.

    Configuration:
    `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)` →
    wrapped in `Pipeline([StandardScaler, LogisticRegression])` and fit per (zone, side).
    """
)
