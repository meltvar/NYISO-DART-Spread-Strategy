"""
Build the 52-dimensional feature matrix and the 22 binary label series
from data/processed/panel.parquet.

The feature vector for each hour t on day D:
    4 zone-level predictors  x  11 zones  =  44 features
    6 calendar features                   =   6 features
    2 past-spike-cluster scalars          =   2 features
                                         ─────────────
    Total                                =  52 features

Zone-level predictors (all sourced from D-2 or D-3, unconditionally settled):
    - day-ahead load forecast (this hour, published before gate closure)
    - lagged DART at 48h  (D-2 same clock-hour)
    - lagged DART at 72h  (D-3 same clock-hour)
    - lagged load forecast error at 48h  (D-2 same clock-hour)

Past-spike-cluster scalars (sourced from the full operating day D-2):
    - past_pos_spikes_d2  count of (zone, hour) pairs on D-2 with DART >= +5
    - past_neg_spikes_d2  count of (zone, hour) pairs on D-2 with DART <= -30

Gate-closure invariant
----------------------
The NYISO DAM for operating day D closes at 05:00 ET on day D-1.
D-2 same-hour DART settles at ~(hour+1):00 on D-2, which is at least
5 hours before gate closure (the latest D-2 hour settles ~00:00 D-1,
still 5h before gate). D-3 is even older. The past-spike scalars are
aggregated over all 24 hours of D-2 — the latest of those also settles
~00:00 D-1. All features are unconditionally leak-free for every hour
0-23 and across DST boundaries.

Both X_safe and X_naive are written as identical matrices (the safe/naive
distinction is vacuous with D-2/D-3 lags — both files kept so the rest
of the pipeline works without modification).

Labels:
    y_{z,pos}[t] = 1 if DART[t,z] >= +5  $/MWh   (DEC signal)
    y_{z,neg}[t] = 1 if DART[t,z] <= -30 $/MWh   (INC signal)

Outputs
-------
data/features/X_safe.parquet     index=interval_start_local,  52 columns
data/features/X_naive.parquet    identical to X_safe
data/features/y.parquet          index=interval_start_local,  22 columns
data/features/manifest.json      audit results, label thresholds, row counts
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
    DART_LAG_HOURS,
    FEATURES_DIR,
    GAMMA_NEG,
    GAMMA_POS,
    LOCAL_TZ,
    PROCESSED_DIR,
    SUMMER_MONTHS,
    WINTER_MONTHS,
    ZONES,
    gate_closure_for_series,
    is_peak_hour,
)

log = logging.getLogger(__name__)

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
X_SAFE_PATH = FEATURES_DIR / "X_safe.parquet"
X_NAIVE_PATH = FEATURES_DIR / "X_naive.parquet"
Y_PATH = FEATURES_DIR / "y.parquet"
MANIFEST_PATH = FEATURES_DIR / "manifest.json"


# ---------------------------------------------------------------------------
# Lag-target arithmetic
# ---------------------------------------------------------------------------
# For an operating hour t on day D, the DA market closes at 05:00 ET on day D-1.
# A feature derived from a panel value at hour s is usable only if the value
# at hour s is settled by gate_closure(t).
#
# DART[s] and load_forecast_error[s] are "settled" approximately one hour
# after s (RT publishes about then). So the constraint is:
#     s + 1h  <  gate_closure(t)
#
# For lag-N-days from hour t at hour-of-day h:
#   - "same-hour-of-day, N days back" -> s = t - N days, hour_of_day(s) = h
#   - settled by gate closure iff:
#         (t - N days, h) + 1h < (t.date - 1d, 5h)
#     i.e.
#         (-N + 1) days + (h + 1)h < (-1) day + 5h
#     i.e.
#         N >= 1  AND  (if N == 1 then h < 4 else always)
#
# Conclusion:
#   * "lag24": use D-1 same-hour if h<4, else D-2 same-hour.
#   * "lag48": use D-2 same-hour if h<4, else D-3 same-hour.
#
# This is the "point-in-time-safe" definition. The "naive" definition always
# uses D-1 for lag24 and D-2 for lag48, accepting the leak for hours 4-23.


def _shift_days_clock(t: pd.Series, days: int) -> pd.Series:
    """Shift a tz-aware Series back by `days` calendar days, preserving
    wall-clock hour-of-day.

    Implementation: strip the timezone, subtract the days in tz-naive space,
    then re-localize. DST boundaries are handled cleanly:
      - Spring-forward "non-existent" target hour -> NaT
      - Fall-back "ambiguous" target hour -> NaT
    These edge rows then carry NaN in the resulting lag column.

    Clock-time semantics matter for our look-ahead invariant: "same hour of
    day yesterday" must be evaluated in clock time, otherwise the fall-back
    day's 25 clock-hours collapse one of them onto gate closure exactly.
    """
    naive = t.dt.tz_localize(None) - pd.Timedelta(days=days)
    return naive.dt.tz_localize(LOCAL_TZ, nonexistent="NaT", ambiguous="NaT")


def _safe_lag_targets(t: pd.Series, base_lag_days: int) -> pd.Series:
    """Point-in-time-safe lag target. Same clock-hour-of-day, `base_lag_days`
    days back if hour_of_day<4, else `base_lag_days+1` days back. The hour<4
    case is the latest same-hour-of-day source still settled by gate_closure(t)."""
    early = _shift_days_clock(t, base_lag_days)
    late = _shift_days_clock(t, base_lag_days + 1)
    return early.where(t.dt.hour < 4, late)


def _naive_lag_targets(t: pd.Series, base_lag_days: int) -> pd.Series:
    """Literal lag: same clock-hour-of-day, `base_lag_days` calendar days
    earlier. May include a small look-ahead for hours 4-23 because the
    source DART value is not yet settled at gate closure for late hours."""
    return _shift_days_clock(t, base_lag_days)


# ---------------------------------------------------------------------------
# Wide reshapes (one column per zone)
# ---------------------------------------------------------------------------
def _wide_by_zone(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot the long panel into wide form: index=interval_start_local, cols=zone."""
    w = panel.pivot(index="interval_start_local", columns="zone", values=value_col)
    return w[ZONES]  # enforce zone order


# ---------------------------------------------------------------------------
# Lag application via self-merge (DST-safe, no row-shift assumptions)
# ---------------------------------------------------------------------------
def _apply_lag(
    wide_values: pd.DataFrame,
    lag_target_per_t: pd.Series,
) -> pd.DataFrame:
    """For each timestamp t in `wide_values.index`, look up the row at
    `lag_target_per_t[t]` and return it as the row labeled t.

    Implementation is a left-merge on timestamp, which handles DST gaps cleanly
    (missing target timestamps produce NaN rows).
    """
    assert wide_values.index.equals(lag_target_per_t.index)
    src = wide_values.reset_index().rename(columns={"interval_start_local": "_target"})
    tgt = pd.DataFrame({"interval_start_local": lag_target_per_t.index})
    # Use .array (preserves tz) not .values (strips tz to numpy datetime64[ns])
    tgt["_target"] = lag_target_per_t.array
    merged = tgt.merge(src, on="_target", how="left")
    out = merged.set_index("interval_start_local")[wide_values.columns]
    return out


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Six calendar features.

    Columns:
      hour_of_day      0-23 integer
      month_of_year    1-12 integer
      is_winter        Dec/Jan/Feb
      is_summer        Jun/Jul/Aug
      is_weekend       Sat/Sun
      is_holiday       US federal holidays
    """
    try:
        import holidays
    except ImportError as e:
        raise RuntimeError("pip install holidays") from e

    ny = holidays.country_holidays("US", subdiv="NY")
    df = pd.DataFrame(index=idx)
    df["hour_of_day"] = idx.hour
    df["month_of_year"] = idx.month
    df["is_winter"] = idx.month.isin(WINTER_MONTHS).astype(int)
    df["is_summer"] = idx.month.isin(SUMMER_MONTHS).astype(int)
    df["is_weekend"] = (idx.weekday >= 5).astype(int)
    df["is_holiday"] = pd.Series(
        [d.date() in ny for d in idx], index=idx
    ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------
def _build_labels(dart_wide: pd.DataFrame) -> pd.DataFrame:
    """Two binary labels per zone per hour.

    y_{z,pos}[t] = 1 iff DART[t, z] >= +GAMMA_POS  (DEC trade signal)
    y_{z,neg}[t] = 1 iff DART[t, z] <= -GAMMA_NEG  (INC trade signal)
    """
    y_pos = (dart_wide >= GAMMA_POS).astype("Int8")
    y_neg = (dart_wide <= -GAMMA_NEG).astype("Int8")
    y_pos.columns = [f"{z}_pos" for z in y_pos.columns]
    y_neg.columns = [f"{z}_neg" for z in y_neg.columns]
    return pd.concat([y_pos, y_neg], axis=1)


# ---------------------------------------------------------------------------
# Past-spike-cluster features
# ---------------------------------------------------------------------------
def _past_spikes_d2(dart_w: pd.DataFrame) -> pd.DataFrame:
    """For each operating hour t on day D, count spike events on D-2.

    past_pos_spikes_d2: number of (zone, hour) pairs on D-2 with DART >= +5
    past_neg_spikes_d2: number of (zone, hour) pairs on D-2 with DART <= -30

    Uses datetime.date arithmetic to avoid DST wall-clock ambiguities — two
    calendar days back is unambiguous at the date level. D-2's last hour
    settles ~00:00 D-1, 5 hours before gate closure. Zero look-ahead.
    """
    from datetime import timedelta

    # Per-hour spike indicator summed across all 11 zones -> scalar per hour
    pos_by_hour = (dart_w >= GAMMA_POS).sum(axis=1)
    neg_by_hour = (dart_w <= -GAMMA_NEG).sum(axis=1)

    # Aggregate to per-operating-date using date objects (DST-safe)
    dates = pd.Series([ts.date() for ts in dart_w.index], index=dart_w.index)
    daily_pos = pos_by_hour.groupby(dates).sum()   # index: datetime.date
    daily_neg = neg_by_hour.groupby(dates).sum()

    # Look up D-2 date for each row (calendar subtraction, no DST ambiguity)
    d2_dates = pd.Series(
        [d - timedelta(days=2) for d in dates],
        index=dart_w.index,
    )
    past_pos = daily_pos.reindex(d2_dates.values).values
    past_neg = daily_neg.reindex(d2_dates.values).values

    return pd.DataFrame(
        {"past_pos_spikes_d2": past_pos.astype(float),
         "past_neg_spikes_d2": past_neg.astype(float)},
        index=dart_w.index,
    )


# ---------------------------------------------------------------------------
# Feature-matrix assembly
# ---------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build X_safe, X_naive, y."""
    log.info("Pivoting panel to wide-by-zone form...")
    dart_w = _wide_by_zone(panel, "dart")
    da_lf_w = _wide_by_zone(panel, "da_load_forecast")
    lfe_w = _wide_by_zone(panel, "load_forecast_error")

    t_series = pd.Series(dart_w.index, index=dart_w.index)

    # D-2 and D-3 same clock-hour: unconditionally settled before gate closure
    # for every hour of day (0-23) and across DST boundaries.
    log.info("Computing unconditionally-safe lag targets (D-2, D-3)...")
    t_lag48 = _naive_lag_targets(t_series, 2)   # D-2 same clock-hour
    t_lag72 = _naive_lag_targets(t_series, 3)   # D-3 same clock-hour

    def _build_X() -> pd.DataFrame:
        log.info("  applying 48h lag to DART...")
        dart_l48 = _apply_lag(dart_w, t_lag48).add_suffix("_dart_lag48")
        log.info("  applying 72h lag to DART...")
        dart_l72 = _apply_lag(dart_w, t_lag72).add_suffix("_dart_lag72")
        log.info("  applying 48h lag to load forecast error...")
        lfe_l48 = _apply_lag(lfe_w, t_lag48).add_suffix("_lfe_lag48")
        log.info("  collecting current DA load forecast (no lag)...")
        da_lf = da_lf_w.add_suffix("_da_load_forecast")
        log.info("  computing calendar features...")
        cal = _calendar_features(dart_w.index)
        log.info("  computing past-spike-cluster features (D-2, leak-free)...")
        spk = _past_spikes_d2(dart_w)
        X = pd.concat([da_lf, dart_l48, dart_l72, lfe_l48, cal, spk], axis=1)
        block_cols = []
        for feat in ("da_load_forecast", "dart_lag48", "dart_lag72", "lfe_lag48"):
            for z in ZONES:
                block_cols.append(f"{z}_{feat}")
        cal_cols = list(cal.columns)
        spk_cols = list(spk.columns)   # past_pos_spikes_d2, past_neg_spikes_d2
        return X[block_cols + cal_cols + spk_cols]

    log.info("Assembling feature matrix (safe = naive, both use D-2/D-3 lags)...")
    X_safe = _build_X()
    X_naive = X_safe  # identical — safe/naive distinction is vacuous with D-2/D-3 lags

    log.info("Building labels...")
    y = _build_labels(dart_w)

    return {"X_safe": X_safe, "X_naive": X_naive, "y": y}


# ---------------------------------------------------------------------------
# Bias-prevention audit
# ---------------------------------------------------------------------------
def _audit_safe(X_safe: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Verify the look-ahead invariant for the feature matrix.

    Primary lag source is D-2 same clock-hour. D-2 hour h settles at
    approximately (h+1):00 on D-2, which is always before the 05:00 D-1
    gate closure. This audit should always return zero leaks.
    """
    idx = pd.Series(X_safe.index, index=X_safe.index)
    source_t = _shift_days_clock(idx, 2)        # D-2 same clock-hour
    realized_at = source_t + pd.Timedelta(hours=1)
    gc = gate_closure_for_series(idx)
    is_leak = realized_at.notna() & (realized_at >= gc)
    return {
        "safe_lag_leaks": int(is_leak.sum()),   # expected: always 0
        "safe_lag_nan_rows": int(source_t.isna().sum()),
    }


def _audit_naive(panel: pd.DataFrame) -> dict:
    """X_naive is now identical to X_safe (D-2/D-3 lags). Audit mirrors safe."""
    idx = panel["interval_start_local"].drop_duplicates().sort_values().reset_index(drop=True)
    source_t = _shift_days_clock(idx, 2)        # D-2, same as safe
    realized_at = source_t + pd.Timedelta(hours=1)
    gc = gate_closure_for_series(idx)
    leak = realized_at.notna() & (realized_at >= gc)
    return {
        "naive_lag_leak_hours": int(leak.sum()),     # expected: always 0
        "naive_leak_fraction": float(leak.sum() / len(idx)),
        "naive_lag_nan_rows": int(source_t.isna().sum()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build features and labels")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if the SAFE audit finds any leak (it never should)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if not args.panel.exists():
        log.error("Panel not found: %s -- run `python -m nyiso_dart.data.build` first.",
                  args.panel)
        return 1

    log.info("Loading panel: %s", args.panel)
    panel = pd.read_parquet(args.panel)

    artefacts = build_features(panel)
    X_safe, X_naive, y = artefacts["X_safe"], artefacts["X_naive"], artefacts["y"]

    # Audits
    safe_audit = _audit_safe(X_safe, panel)
    naive_audit = _audit_naive(panel)
    log.info("Audit (safe):  %s", safe_audit)
    log.info("Audit (naive): %s", naive_audit)

    if safe_audit["safe_lag_leaks"] > 0:
        msg = (
            f"FATAL: safe matrix has {safe_audit['safe_lag_leaks']} look-ahead leaks. "
            "Bug in lag arithmetic; refusing to save."
        )
        if args.strict:
            raise RuntimeError(msg)
        log.error(msg)
        return 2

    # Save
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    X_safe.to_parquet(X_SAFE_PATH)
    X_naive.to_parquet(X_NAIVE_PATH)
    y.to_parquet(Y_PATH)

    manifest = {
        "panel_path": str(args.panel),
        "rows": int(len(X_safe)),
        "feature_cols": int(X_safe.shape[1]),
        "label_cols": int(y.shape[1]),
        "gamma_pos": GAMMA_POS,
        "gamma_neg": GAMMA_NEG,
        "audit_safe": safe_audit,
        "audit_naive": naive_audit,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    log.info("Wrote X_safe : %s  shape=%s", X_SAFE_PATH, X_safe.shape)
    log.info("Wrote X_naive: %s  shape=%s", X_NAIVE_PATH, X_naive.shape)
    log.info("Wrote y      : %s  shape=%s", Y_PATH, y.shape)
    log.info("Manifest     : %s", MANIFEST_PATH)
    print("\n--- manifest ---")
    print(MANIFEST_PATH.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
