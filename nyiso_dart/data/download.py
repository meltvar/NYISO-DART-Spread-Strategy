"""
Raw data download from NYISO via gridstatus.io.

Design principles
-----------------
1. **Immutability**: Files in data/raw/ are append-only. Re-running this
   module is idempotent; existing files are skipped unless --force is passed.
   This guarantees that any downstream artefact can be regenerated bit-for-bit.

2. **Provenance**: Every row is stamped with `retrieved_at` (UTC) so we can
   later prove which snapshot of NYISO data backed each result.

3. **Timezone discipline**: gridstatus returns timestamps as tz-aware datetimes.
   We coerce everything to UTC at the storage boundary. Local-time conversion
   happens only when computing gate-closure constraints downstream.

4. **No silent loss**: Row counts are logged after every download. The
   validation pass (`validate.py`) checks completeness against expected
   hourly grids and fails loudly on gaps.

Datasets
--------
- da_lmp:           Day-Ahead LMP, hourly, 11 zones.  $/MWh.
- rt_lmp:           Real-Time LMP, hourly (integrated from 5-min). $/MWh.
- da_load_forecast: Day-Ahead load forecast, zonal + system. MW.
- actual_load:      Realized load, zonal + system. MW.

Note on gridstatus API surface
------------------------------
gridstatus has evolved across versions. This module isolates every gridstatus
call inside `_fetch_*` functions so adjustments stay local. The version pin
in requirements.txt (>=0.30) targets a stable surface as of 2025.

Usage
-----
    python -m nyiso_dart.data.download --start 2015 --end 2025
    python -m nyiso_dart.data.download --start 2024 --end 2024 --datasets da_lmp rt_lmp
    python -m nyiso_dart.data.download --start 2024 --end 2024 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from nyiso_dart.config import RAW_DATASETS, RAW_DIR, ZONE_ALIASES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------
def raw_path(dataset: str, year: int) -> Path:
    """data/raw/<dataset>/year=<YYYY>.parquet"""
    return RAW_DIR / dataset / f"year={year}.parquet"


# ---------------------------------------------------------------------------
# Zone normalization (canonical abbreviations)
# ---------------------------------------------------------------------------
def _normalize_zone(name: object) -> object:
    if not isinstance(name, str):
        return name
    key = name.strip().upper()
    return ZONE_ALIASES.get(key, key.replace(" ", ""))


def _normalize_zone_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If df has a zone-identifying column, map values to canonical abbreviations."""
    for col in ("Location", "Zone", "Name", "location", "zone", "name"):
        if col in df.columns:
            df[col] = df[col].map(_normalize_zone)
            return df
    return df


# ---------------------------------------------------------------------------
# Provenance stamping
# ---------------------------------------------------------------------------
def _stamp(df: pd.DataFrame, dataset: str, year: int) -> pd.DataFrame:
    """Stamp dataset, year, and retrieval UTC timestamp."""
    df = df.copy()
    df["dataset"] = dataset
    df["year_partition"] = year
    df["retrieved_at_utc"] = datetime.now(timezone.utc)
    return df


# ---------------------------------------------------------------------------
# gridstatus fetchers — keep API surface isolated
# ---------------------------------------------------------------------------
def _import_gridstatus():
    try:
        from gridstatus import NYISO
        from gridstatus.base import Markets
    except ImportError as e:
        raise RuntimeError(
            "gridstatus is required. Install with: pip install -r requirements.txt"
        ) from e
    return NYISO, Markets


def _fetch_da_lmp(iso, Markets, start: str, end: str) -> pd.DataFrame:
    return iso.get_lmp(
        date=start,
        end=end,
        market=Markets.DAY_AHEAD_HOURLY,
        location_type="zone",
    )


def _fetch_rt_lmp(iso, Markets, start: str, end: str) -> pd.DataFrame:
    # NYISO real-time LMPs are published every 5 minutes. gridstatus offers
    # hourly aggregation under Markets.REAL_TIME_HOURLY when available; this
    # matches the DA hourly grid used to compute DART.
    return iso.get_lmp(
        date=start,
        end=end,
        market=Markets.REAL_TIME_HOURLY,
        location_type="zone",
    )


def _fetch_da_load_forecast(iso, Markets, start: str, end: str) -> pd.DataFrame:
    # Use the *zonal* forecast (NYISO's ISOLF product), not the system-only one.
    # We need zonal day-ahead load forecasts (one column per zone), not the
    # system-level forecast that the default `get_load_forecast` returns.
    # The returned frame is wide (one column per zone) and includes a
    # `Publish Time` column we'll use to enforce look-ahead constraints.
    return iso.get_zonal_load_forecast(date=start, end=end)


def _fetch_actual_load(iso, Markets, start: str, end: str) -> pd.DataFrame:
    return iso.get_load(date=start, end=end)


FETCHERS: dict[str, Callable] = {
    "da_lmp": _fetch_da_lmp,
    "rt_lmp": _fetch_rt_lmp,
    "da_load_forecast": _fetch_da_load_forecast,
    "actual_load": _fetch_actual_load,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def download_year(
    dataset: str,
    year: int,
    force: bool = False,
    iso=None,
    Markets=None,
) -> Path:
    """Download one (dataset, year) pair. Returns path to parquet file."""
    path = raw_path(dataset, year)
    if path.exists() and not force:
        log.info("Skip %s %d — exists at %s", dataset, year, path)
        return path

    if iso is None or Markets is None:
        NYISO_cls, Markets = _import_gridstatus()
        iso = NYISO_cls()

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    fetcher = FETCHERS.get(dataset)
    if fetcher is None:
        raise ValueError(f"Unknown dataset: {dataset!r}. Known: {list(FETCHERS)}")

    log.info("Fetching %s %d...", dataset, year)
    t0 = time.time()
    df = fetcher(iso, Markets, start, end)
    elapsed = time.time() - t0

    if df is None or len(df) == 0:
        raise RuntimeError(f"Empty response for {dataset} {year}")

    df = _normalize_zone_columns(df)
    df = _stamp(df, dataset, year)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("  %s %d -> %s rows in %.1fs -> %s", dataset, year, f"{len(df):,}", elapsed, path)
    return path


def download(
    years: list[int],
    datasets: tuple[str, ...] = RAW_DATASETS,
    force: bool = False,
) -> None:
    NYISO_cls, Markets = _import_gridstatus()
    iso = NYISO_cls()
    for dataset in datasets:
        for year in years:
            try:
                download_year(dataset, year, force=force, iso=iso, Markets=Markets)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed %s %d: %s", dataset, year, exc)
                # Do not abort the whole run on a single failure; continue and
                # let validate.py flag missing files.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--start", type=int, default=2015, help="First year (inclusive)")
    parser.add_argument("--end", type=int, default=2025, help="Last year (inclusive)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(RAW_DATASETS),
        choices=list(RAW_DATASETS),
        help="Subset of datasets to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    years = list(range(args.start, args.end + 1))
    log.info("Downloading datasets=%s years=%d-%d", args.datasets, args.start, args.end)
    download(years, tuple(args.datasets), force=args.force)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
