"""
Build the canonical hourly DART panel from raw downloads.

Inputs (data/raw/)
------------------
  da_lmp:           long, hourly, zonal LMPs with Energy/Loss/Congestion decomposition
  rt_lmp:           long, hourly, zonal LMPs with Energy/Loss/Congestion decomposition
  da_load_forecast: wide, hourly, multi-vintage. Each operating hour appears once per
                    Publish Time. Filtered by gate closure during reshape.
  actual_load:      wide, 5-minute, zonal. Resampled to hourly means during reshape.

Output (data/processed/panel.parquet)
-------------------------------------
One row per (interval_start_utc, zone) with columns:
  interval_start_utc       UTC pandas Timestamp
  interval_start_local     tz-aware America/New_York
  zone                     one of the 11 canonical zone abbreviations
  da_lmp                   $/MWh
  rt_lmp                   $/MWh
  da_energy                $/MWh (system-wide energy component)
  da_loss                  $/MWh (zonal loss component)
  da_congestion            $/MWh (zonal congestion component)
  rt_energy, rt_loss, rt_congestion  same for RT
  dart                     = da_lmp - rt_lmp
  actual_load              MW, hourly mean of 5-min values
  da_load_forecast         MW, forecast for this hour whose Publish Time was the
                           latest before gate_closure_for(interval_start_local)
  forecast_publish_time    when that forecast became knowable
  load_forecast_error      actual_load - da_load_forecast (only meaningful after
                           the operating hour completes; downstream we lag this)

Bias-prevention design
----------------------
- LMP and actual_load are realized values. Their "knowledge time" is their
  publish time, which we treat as `interval_end + epsilon`. Downstream code
  lags them appropriately when used as features.
- DA load forecast is forward-looking. There are multiple vintages per
  (interval_start, zone). We select exactly one: the latest Publish Time
  strictly less than gate_closure_for(interval_start_local). This is the
  forecast a trader sitting at the DA gate could have seen.
- Raw files in data/raw/ are not touched. This module reads them and writes
  data/processed/panel.parquet.

Usage
-----
    python -m nyiso_dart.data.build
    python -m nyiso_dart.data.build --years 2024
    python -m nyiso_dart.data.build --years 2015 2016 2017
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from nyiso_dart.config import (
    LOCAL_TZ,
    PROCESSED_DIR,
    RAW_DIR,
    ZONE_ALIASES,
    ZONES,
    gate_closure_for_series,
)
from nyiso_dart.data.download import raw_path
from nyiso_dart.data.validate import DATASET_SCHEMAS

log = logging.getLogger(__name__)

PANEL_PATH = PROCESSED_DIR / "panel.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _alias(name: str) -> str:
    return ZONE_ALIASES.get(name.strip().upper(), name.strip().upper().replace(" ", ""))


def _years_on_disk(dataset: str) -> list[int]:
    ds_dir = RAW_DIR / dataset
    if not ds_dir.exists():
        return []
    years = []
    for f in ds_dir.glob("year=*.parquet"):
        try:
            years.append(int(f.stem.split("=")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(years)


def _read_years(dataset: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        p = raw_path(dataset, y)
        if not p.exists():
            log.warning("Missing %s year=%d (%s); skipping", dataset, y, p)
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"No raw files found for {dataset} in {years}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# LMP reshape
# ---------------------------------------------------------------------------
def _load_lmp_long(dataset: str, years: list[int]) -> pd.DataFrame:
    """Load a long-format LMP dataset and normalize columns.

    Output columns:
      interval_start_local (tz-aware Eastern)
      zone (canonical abbreviation, 11 zones only)
      lmp, energy, loss, congestion
    """
    schema = DATASET_SCHEMAS[dataset]
    df = _read_years(dataset, years)

    df = df.rename(columns={schema["time_col"]: "interval_start_local",
                            schema["location_col"]: "zone"})
    df["zone"] = df["zone"].map(_alias)
    df = df[df["zone"].isin(ZONES)].copy()

    # Keep only the columns we need; standardize names
    keep = {
        "interval_start_local": "interval_start_local",
        "zone": "zone",
        "LMP": "lmp",
        "Energy": "energy",
        "Congestion": "congestion",
        "Loss": "loss",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

    # Ensure tz-aware Eastern
    if df["interval_start_local"].dt.tz is None:
        df["interval_start_local"] = df["interval_start_local"].dt.tz_localize(LOCAL_TZ)
    else:
        df["interval_start_local"] = df["interval_start_local"].dt.tz_convert(LOCAL_TZ)

    df = df.drop_duplicates(subset=["interval_start_local", "zone"], keep="last")
    df = df.sort_values(["interval_start_local", "zone"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Actual load reshape: 5-min wide -> hourly long
# ---------------------------------------------------------------------------
def _load_actual_long(years: list[int]) -> pd.DataFrame:
    """Reshape actual_load: wide 5-min -> long hourly mean.

    Output columns:
      interval_start_local (tz-aware Eastern, on hour grid)
      zone (canonical abbreviation)
      actual_load (MW, hourly mean of 5-min observations)
    """
    df = _read_years("actual_load", years)
    schema = DATASET_SCHEMAS["actual_load"]
    tcol = schema["time_col"]

    # Identify zone columns by alias
    col_to_zone = {c: _alias(c) for c in df.columns if _alias(c) in ZONES}
    if len(col_to_zone) != len(ZONES):
        missing = set(ZONES) - set(col_to_zone.values())
        raise ValueError(f"actual_load missing zone columns after alias: {missing}")

    df = df[[tcol] + list(col_to_zone.keys())].rename(columns={tcol: "ts"})
    df = df.rename(columns=col_to_zone)

    # DST-safe hourly aggregation: do the floor in UTC where every hour is
    # unambiguous, then convert the bucket label back to Eastern. The fall-back
    # day in NYISO has two distinct 01:00 ET hours (EDT then EST) which pandas
    # cannot floor directly on tz-aware data without an `ambiguous` rule.
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(LOCAL_TZ)
    ts_utc = df["ts"].dt.tz_convert("UTC")
    hour_bucket_utc = ts_utc.dt.floor("h")

    df["interval_start_local"] = hour_bucket_utc.dt.tz_convert(LOCAL_TZ)
    hourly = (
        df.drop(columns="ts")
        .groupby("interval_start_local", as_index=False, sort=True, observed=True)
        .mean(numeric_only=True)
    )

    long = hourly.melt(
        id_vars="interval_start_local",
        value_vars=list(ZONES),
        var_name="zone",
        value_name="actual_load",
    )
    long = long.sort_values(["interval_start_local", "zone"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# DA load forecast reshape: wide multi-vintage -> long, gate-closure-filtered
# ---------------------------------------------------------------------------
def _load_forecast_long(years: list[int]) -> pd.DataFrame:
    """Reshape DA zonal load forecast.

    The raw frame has multiple Publish Time vintages per (Interval Start, Zone).
    We keep exactly one vintage per (interval_start, zone): the latest Publish
    Time strictly less than gate_closure_for(interval_start). That is the
    forecast a trader sitting at the DA gate could legally have used to bid
    for operating hour `interval_start`.

    Output columns:
      interval_start_local (tz-aware Eastern)
      zone (canonical abbreviation)
      da_load_forecast (MW)
      forecast_publish_time (tz-aware Eastern, "knowledge time")
    """
    df = _read_years("da_load_forecast", years)
    schema = DATASET_SCHEMAS["da_load_forecast"]
    tcol = schema["time_col"]
    pcol = schema["publish_time_col"]

    col_to_zone = {c: _alias(c) for c in df.columns if _alias(c) in ZONES}
    if len(col_to_zone) != len(ZONES):
        missing = set(ZONES) - set(col_to_zone.values())
        raise ValueError(f"da_load_forecast missing zone columns after alias: {missing}")

    keep_cols = [tcol, pcol] + list(col_to_zone.keys())
    df = df[keep_cols].rename(columns={tcol: "interval_start_local",
                                       pcol: "forecast_publish_time"})
    df = df.rename(columns=col_to_zone)

    # Localize / convert to Eastern
    for c in ("interval_start_local", "forecast_publish_time"):
        if df[c].dt.tz is None:
            df[c] = df[c].dt.tz_localize(LOCAL_TZ)
        else:
            df[c] = df[c].dt.tz_convert(LOCAL_TZ)

    long = df.melt(
        id_vars=["interval_start_local", "forecast_publish_time"],
        value_vars=list(ZONES),
        var_name="zone",
        value_name="da_load_forecast",
    )

    # --- Bias-critical step ----------------------------------------------
    # For each operating hour, retain only vintages knowable by gate close.
    gc = gate_closure_for_series(long["interval_start_local"])
    long = long[long["forecast_publish_time"] < gc].copy()

    # Among remaining vintages, pick the latest one per (interval_start, zone).
    idx = long.groupby(
        ["interval_start_local", "zone"], sort=False
    )["forecast_publish_time"].idxmax()
    long = long.loc[idx].reset_index(drop=True)
    # ---------------------------------------------------------------------

    long = long.sort_values(["interval_start_local", "zone"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# Panel build
# ---------------------------------------------------------------------------
def build_panel(years: list[int]) -> pd.DataFrame:
    log.info("Loading DA LMP for years %s", years)
    da = _load_lmp_long("da_lmp", years).rename(
        columns={"lmp": "da_lmp", "energy": "da_energy",
                 "loss": "da_loss", "congestion": "da_congestion"}
    )

    log.info("Loading RT LMP for years %s", years)
    rt = _load_lmp_long("rt_lmp", years).rename(
        columns={"lmp": "rt_lmp", "energy": "rt_energy",
                 "loss": "rt_loss", "congestion": "rt_congestion"}
    )

    log.info("Loading actual load for years %s (5-min -> hourly)", years)
    actual = _load_actual_long(years)

    log.info("Loading DA load forecast for years %s (vintage-filtered)", years)
    forecast = _load_forecast_long(years)

    log.info("Merging panel...")
    panel = da.merge(rt, on=["interval_start_local", "zone"], how="inner")
    panel = panel.merge(actual, on=["interval_start_local", "zone"], how="left")
    panel = panel.merge(forecast, on=["interval_start_local", "zone"], how="left")

    # Derived columns
    panel["dart"] = panel["da_lmp"] - panel["rt_lmp"]
    panel["load_forecast_error"] = panel["actual_load"] - panel["da_load_forecast"]

    # UTC mirror of interval_start for stable sorting / external use
    panel["interval_start_utc"] = panel["interval_start_local"].dt.tz_convert("UTC")

    # Column ordering
    cols = [
        "interval_start_utc", "interval_start_local", "zone",
        "da_lmp", "rt_lmp", "dart",
        "da_energy", "da_loss", "da_congestion",
        "rt_energy", "rt_loss", "rt_congestion",
        "actual_load", "da_load_forecast", "forecast_publish_time",
        "load_forecast_error",
    ]
    panel = panel[[c for c in cols if c in panel.columns]]
    panel = panel.sort_values(["interval_start_local", "zone"]).reset_index(drop=True)

    log.info("Panel built: %s rows, %s zones, %s hours",
             f"{len(panel):,}",
             panel["zone"].nunique(),
             panel["interval_start_local"].nunique())
    return panel


def _print_summary(panel: pd.DataFrame) -> None:
    print(f"\nPanel shape: {panel.shape}")
    print(f"Zones: {sorted(panel['zone'].unique())}")
    print(f"Time range: [{panel['interval_start_local'].min()} .. {panel['interval_start_local'].max()}]")
    print(f"Hours per zone:")
    print(panel.groupby("zone").size().to_string())
    print(f"\nNulls per column:")
    print(panel.isna().sum().to_string())
    print(f"\nDART stats:")
    print(panel["dart"].describe().to_string())


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Explicit years to build (default: every year present on disk)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PANEL_PATH,
        help=f"Output parquet path (default: {PANEL_PATH})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    if args.years is None:
        # Use intersection of years present across all required datasets
        years_sets = [set(_years_on_disk(ds)) for ds in
                      ("da_lmp", "rt_lmp", "da_load_forecast", "actual_load")]
        years = sorted(set.intersection(*years_sets)) if all(years_sets) else []
        if not years:
            log.error("No years are present across all 4 raw datasets. "
                      "Run download.py first.")
            return 1
    else:
        years = sorted(args.years)

    log.info("Building panel for years: %s", years)
    panel = build_panel(years)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    log.info("Wrote panel: %s", args.output)
    _print_summary(panel)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
