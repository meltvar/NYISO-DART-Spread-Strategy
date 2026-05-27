"""
Single source of truth for project constants and pipeline parameters.

Every value here defines the contract that downstream modules rely on:
- Calendar splits between training, validation, and test
- Spike thresholds defining the binary labels
- Gate-closure timing for the look-ahead invariant
- Zone identifiers and aliases used when ingesting raw NYISO data

Changing any value here is equivalent to redefining the strategy and should
require explicit justification.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _p in (DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------
MARKET = "NYISO"

# NYISO operates on Eastern Time with DST. All gate-closure arithmetic must
# go through this zoneinfo so spring-forward and fall-back days are handled
# correctly. Storage timestamps remain in UTC.
LOCAL_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Day-Ahead Market gate closure: NYISO's DAM for operating day D closes at
# 05:00 Eastern on day D-1. Every predictor must be settled strictly before
# this clock time.
GATE_CLOSURE_HOUR_LOCAL = 5  # 05:00 Eastern

# The 11 NYISO load zones.
ZONES: list[str] = [
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUDVL",
    "LONGIL",
    "MHKVL",
    "MILLWD",
    "NORTH",
    "NYC",
    "WEST",
]

# Six "large-demand" zones used for narrative-focused summary tables.
FOCUS_ZONES: list[str] = ["CAPITL", "CENTRL", "LONGIL", "NORTH", "NYC", "WEST"]

# GridStatus / NYISO publications occasionally use longer zone names.
# Map them to the canonical abbreviations on ingest.
ZONE_ALIASES: dict[str, str] = {
    "N.Y.C.": "NYC",
    "NYC": "NYC",
    "LONG IL": "LONGIL",
    "LONG ISLAND": "LONGIL",
    "LONGIL": "LONGIL",
    "CAPITAL": "CAPITL",
    "CENTRAL": "CENTRL",
    "GENESEE": "GENESE",
    "MOHAWK VL": "MHKVL",
    "MILLWOOD": "MILLWD",
    "HUD VL": "HUDVL",
    "HUDSON VL": "HUDVL",
    "DUNWOODIE": "DUNWOD",
}


# ---------------------------------------------------------------------------
# Spike thresholds — the binary labels for the classifiers
# ---------------------------------------------------------------------------
GAMMA_POS: float = 5.0   # $/MWh, positive DART spike threshold (DEC signal)
GAMMA_NEG: float = 30.0  # $/MWh, negative DART spike threshold (INC signal)


# ---------------------------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------------------------
# Locked by calendar date. Do not change. Test period is the held-out set;
# nothing in training or threshold selection may inspect it.
TRAIN_START = "2015-01-01"
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2025-12-31"


# ---------------------------------------------------------------------------
# Validation-set threshold sweep grid
# ---------------------------------------------------------------------------
# Probability cutoffs are tuned on validation to maximise unit-size P&L,
# separately for each (zone, side).
TAU_GRID: np.ndarray = np.round(np.arange(0.50, 1.00, 0.01), 2)


# ---------------------------------------------------------------------------
# Calendar feature definitions
# ---------------------------------------------------------------------------
WINTER_MONTHS = {12, 1, 2}
SUMMER_MONTHS = {6, 7, 8}
# Remaining months → "Shoulder"

# NYISO peak hours: Hour Beginning 07:00 through 22:00 on weekdays
# (industry-standard "on-peak" window).
PEAK_HOUR_RANGE = range(7, 23)


def season_of(month: int) -> str:
    if month in WINTER_MONTHS:
        return "Winter"
    if month in SUMMER_MONTHS:
        return "Summer"
    return "Shoulder"


def is_peak_hour(hour: int, weekday: int) -> bool:
    """NYISO on-peak definition: weekdays HB07-HB22."""
    return weekday < 5 and hour in PEAK_HOUR_RANGE


# ---------------------------------------------------------------------------
# Gate-closure arithmetic
# ---------------------------------------------------------------------------
# The NYISO Day-Ahead Market for operating day D closes at 05:00 Eastern on
# day D-1. Every feature used to predict hour t must have a knowledge time
# strictly less than gate_closure_for(t).
#
# Equivalent prediction horizon:
#   operating hour 00:00 of D  ->  19 hours ahead of gate closure
#   operating hour 23:00 of D  ->  42 hours ahead of gate closure
import pandas as pd  # noqa: E402  (kept inside this section for clarity)


def gate_closure_for(interval_start_local: pd.Timestamp) -> pd.Timestamp:
    """
    NYISO DA gate closure for an operating hour.

    For an operating hour `interval_start_local` (tz-aware America/New_York),
    return the timestamp of the DA market gate that priced that hour:
    05:00 Eastern on the day BEFORE the operating day.
    """
    operating_midnight = interval_start_local.normalize()
    return operating_midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR_LOCAL)


def gate_closure_for_series(interval_start_local: pd.Series) -> pd.Series:
    """Vectorized version of gate_closure_for for a tz-aware Series."""
    operating_midnight = interval_start_local.dt.normalize()
    return operating_midnight - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR_LOCAL)


# ---------------------------------------------------------------------------
# Feature lag horizons
# ---------------------------------------------------------------------------
DART_LAG_HOURS: tuple[int, ...] = (48, 72)
LOAD_FORECAST_ERROR_LAG_HOURS: tuple[int, ...] = (48,)


# ---------------------------------------------------------------------------
# Datasets to download from gridstatus
# ---------------------------------------------------------------------------
RAW_DATASETS: tuple[str, ...] = (
    "da_lmp",            # Day-Ahead LMP, hourly, zonal
    "rt_lmp",            # Real-Time LMP, hourly, zonal
    "da_load_forecast",  # Day-Ahead load forecast, zonal + system
    "actual_load",       # Realised load, zonal + system
)
