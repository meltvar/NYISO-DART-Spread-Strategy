"""
Cached data loaders for the dashboard.

All loaders are decorated with @st.cache_data so the underlying parquet files
are read once per session and shared across pages. Switching pages should be
near-instant.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve project root from this file's location: dashboard/lib/data.py -> root
PROJECT  = Path(__file__).resolve().parent.parent.parent
DATA     = PROJECT / "data"
MODELS   = PROJECT / "models"
RESULTS  = PROJECT / "results"
FEATS    = DATA / "features"
PROC     = DATA / "processed"

ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUDVL",
         "LONGIL", "MHKVL", "MILLWD", "NORTH", "NYC", "WEST"]
TZ    = "America/New_York"

# Primary variant: leak-free D-2/D-3 lag specification. The "naive" variant
# (literal D-1/D-2 lags) is kept only for the methodology page that quantifies
# the leakage cost; never use it on the headline.
PRIMARY = "safe"


# ---------------------------------------------------------------------------
# Core artifacts
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading panel...")
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROC / "panel.parquet")


@st.cache_data(show_spinner="Loading 2026 panel...")
def load_panel_2026() -> pd.DataFrame | None:
    p = PROC / "panel_live_2026.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_trades(variant: str = PRIMARY) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / variant / "trades.parquet")
    df["correct"] = df.apply(
        lambda r: r["dart"] >= 5.0 if r["side"] == "pos" else r["dart"] <= -30.0,
        axis=1,
    )
    return df


@st.cache_data
def load_cumulative_pnl(variant: str = PRIMARY) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / variant / "cumulative_pnl.parquet")


@st.cache_data
def load_zone_summary(variant: str = PRIMARY) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / variant / "zone_summary.parquet")


@st.cache_data
def load_thresholds(variant: str = PRIMARY) -> pd.DataFrame:
    return pd.read_parquet(MODELS / f"thresholds_{variant}.parquet")


@st.cache_data
def load_predictions(variant: str = PRIMARY) -> pd.DataFrame:
    return pd.read_parquet(FEATS / f"predictions_{variant}.parquet")


@st.cache_data
def load_features(variant: str = PRIMARY) -> pd.DataFrame:
    return pd.read_parquet(FEATS / f"X_{variant}.parquet")


@st.cache_data
def load_precision_recall(variant: str = PRIMARY) -> pd.DataFrame:
    p = RESULTS / variant / "precision_recall.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data
def load_yearly(variant: str = PRIMARY) -> pd.DataFrame:
    p = RESULTS / variant / "zone_year_summary.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# 2026 live trades — derived on the fly from the live panel + saved models
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing 2026 live trades...")
def compute_2026_trades() -> pd.DataFrame:
    """Apply the frozen models to 2026 data and return the trade log."""
    import joblib
    import numpy as np
    import sys
    sys.path.insert(0, str(PROJECT))
    from nyiso_dart.features.build import build_features

    panel_full = load_panel_2026()
    if panel_full is None:
        return pd.DataFrame()

    live_end = pd.Timestamp("2026-05-15 23:00:00", tz=TZ)

    artefacts = build_features(panel_full)
    X = artefacts[f"X_{PRIMARY}"]
    mask = (X.index.year == 2026) & (X.index <= live_end)
    X_2026 = X[mask]
    valid = X_2026.notna().all(axis=1)

    preds = pd.DataFrame(index=X_2026.index, dtype="float64")
    for zone in ZONES:
        for side in ("pos", "neg"):
            pipe = joblib.load(MODELS / PRIMARY / f"{zone}_{side}.joblib")
            col = pd.Series(np.nan, index=X_2026.index)
            if valid.any():
                col[valid] = pipe.predict_proba(X_2026[valid].values)[:, 1]
            preds[f"{zone}_{side}"] = col

    thr = load_thresholds()
    elig = thr[thr["eligible"]].copy()

    dart_wide = panel_full.pivot(
        index="interval_start_local", columns="zone", values="dart"
    )[ZONES]
    dart_2026 = dart_wide[(dart_wide.index.year == 2026) & (dart_wide.index <= live_end)]

    rows = []
    for _, r in elig.iterrows():
        zone, side, tau = r["zone"], r["side"], float(r["best_tau"])
        p = preds[f"{zone}_{side}"]
        d = dart_2026[zone]
        ok = p.notna() & d.notna() & (p >= tau)
        if not ok.any():
            continue
        payoff = d[ok] if side == "pos" else -d[ok]
        for t, pv, dv, py in zip(ok[ok].index, p[ok], d[ok], payoff):
            rows.append({
                "interval_start_local": t,
                "zone": zone, "side": side,
                "proba": float(pv), "dart": float(dv), "payoff": float(py),
                "year": t.year, "month": t.month, "hour": t.hour,
                "correct": (float(dv) >= 5.0 if side == "pos" else float(dv) <= -30.0),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("interval_start_local").reset_index(drop=True)
