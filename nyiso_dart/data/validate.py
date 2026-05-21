"""
Validation of raw downloads in data/raw/.

Each dataset has a different shape (long vs wide, hourly vs 5-min, with vs
without a Publish Time column). The validator dispatches on dataset name
to apply the right set of checks. It never modifies raw data; it only reports.

Checks
------
For every (dataset, year) file:
  - File exists and is readable
  - Provenance stamps present (`dataset`, `year_partition`, `retrieved_at_utc`)
  - Time axis present, year-bounded sanity
  - All 11 canonical zones present (either as long-format `Location` values
    or as wide-format columns after alias resolution)
  - No nulls in the value columns

The validator emits WARN (non-fatal) and ERR (fatal in --strict mode) lines.

Usage
-----
    python -m nyiso_dart.data.validate
    python -m nyiso_dart.data.validate --year 2024
    python -m nyiso_dart.data.validate --strict
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nyiso_dart.config import RAW_DATASETS, RAW_DIR, ZONE_ALIASES, ZONES
from nyiso_dart.data.download import raw_path

log = logging.getLogger(__name__)


# Dataset shape registry. Used by validate and (later) by build.py.
# This is the explicit contract for what each raw file looks like.
DATASET_SCHEMAS: dict[str, dict] = {
    "da_lmp": {
        "format": "long",
        "frequency": "hourly",
        "time_col": "Interval Start",
        "location_col": "Location",
        "value_cols": ("LMP", "Energy", "Congestion", "Loss"),
        "publish_time_col": None,
    },
    "rt_lmp": {
        "format": "long",
        "frequency": "hourly",
        "time_col": "Interval Start",
        "location_col": "Location",
        "value_cols": ("LMP", "Energy", "Congestion", "Loss"),
        "publish_time_col": None,
    },
    "da_load_forecast": {
        # Wide: one column per zone, with a system "NYISO" column and a
        # Publish Time column for vintage tracking.
        "format": "wide",
        "frequency": "hourly",
        "time_col": "Interval Start",
        "location_col": None,
        "system_col": "NYISO",
        "publish_time_col": "Publish Time",
    },
    "actual_load": {
        # Wide: zone columns + system "Load" column. 5-minute resolution.
        "format": "wide",
        "frequency": "5min",
        "time_col": "Time",
        "location_col": None,
        "system_col": "Load",
        "publish_time_col": None,
    },
}


@dataclass
class FileReport:
    dataset: str
    year: int
    path: Path
    exists: bool = False
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    zones_present: list[str] = field(default_factory=list)
    zones_missing: list[str] = field(default_factory=list)
    extra_locations: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    publish_time_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and not self.errors


def _alias(name: str) -> str:
    """Map a zone-like label (column header or value) to the canonical abbreviation."""
    key = name.strip().upper()
    return ZONE_ALIASES.get(key, key.replace(" ", ""))


def _check_provenance(df: pd.DataFrame, rep: FileReport) -> None:
    for stamp in ("dataset", "year_partition", "retrieved_at_utc"):
        if stamp not in df.columns:
            rep.errors.append(f"missing provenance column: {stamp}")


def _check_time_axis(df: pd.DataFrame, rep: FileReport, schema: dict) -> pd.Series | None:
    tcol = schema["time_col"]
    if tcol not in df.columns:
        rep.errors.append(f"missing time column: {tcol}")
        return None
    ts = pd.to_datetime(df[tcol])
    rep.min_time, rep.max_time = ts.min(), ts.max()
    # For datasets with a forward-looking Publish Time, the time axis can
    # legitimately extend past year-end (forecasts published late in the
    # year project 7 days ahead). Don't warn on that.
    if schema["publish_time_col"] is None:
        if rep.min_time.year != rep.year or rep.max_time.year != rep.year:
            rep.warnings.append(
                f"timestamps span beyond year={rep.year}: {rep.min_time} .. {rep.max_time}"
            )
    return ts


def _check_long_zones(df: pd.DataFrame, rep: FileReport, schema: dict) -> None:
    lcol = schema["location_col"]
    if lcol not in df.columns:
        rep.errors.append(f"missing location column: {lcol}")
        return
    present = {_alias(v) for v in df[lcol].dropna().unique() if isinstance(v, str)}
    rep.zones_present = sorted(present & set(ZONES))
    rep.zones_missing = sorted(set(ZONES) - present)
    rep.extra_locations = sorted(present - set(ZONES))
    if rep.zones_missing:
        rep.errors.append(f"missing zones: {rep.zones_missing}")


def _check_wide_zones(df: pd.DataFrame, rep: FileReport) -> None:
    aliased = {_alias(c): c for c in df.columns}
    rep.zones_present = sorted(set(aliased) & set(ZONES))
    rep.zones_missing = sorted(set(ZONES) - set(aliased))
    # Don't report non-zone columns as "extra locations" — too noisy.
    if rep.zones_missing:
        rep.errors.append(f"missing zone columns: {rep.zones_missing}")


def _check_publish_time(df: pd.DataFrame, rep: FileReport, schema: dict) -> None:
    pcol = schema["publish_time_col"]
    if pcol is None:
        return
    if pcol not in df.columns:
        rep.errors.append(f"missing publish time column: {pcol}")
        return
    pt = pd.to_datetime(df[pcol])
    rep.publish_time_range = (pt.min(), pt.max())
    # For a vintage to be usable for an operating hour t, its publish time
    # must be < gate_closure(day_of(t)). Per-row validation belongs in
    # build.py; here we just confirm Publish Time isn't in the future.
    if pt.max() > pd.Timestamp.now(tz=pt.dt.tz):
        rep.warnings.append(f"publish times extend into the future: max={pt.max()}")


def _check_value_nulls(df: pd.DataFrame, rep: FileReport, schema: dict) -> None:
    if schema["format"] == "long":
        for c in schema.get("value_cols", ()):
            if c in df.columns:
                n = int(df[c].isna().sum())
                if n > 0:
                    rep.null_counts[c] = n
    else:
        # Wide: check zone columns and system column
        cols_to_check = list(rep.zones_present) if False else []
        # Pull the actual column names (gridstatus uses "Hud Vl" etc.)
        col_map = {_alias(c): c for c in df.columns}
        for z in ZONES:
            if z in col_map:
                cols_to_check.append(col_map[z])
        if "system_col" in schema and schema["system_col"] in df.columns:
            cols_to_check.append(schema["system_col"])
        for c in cols_to_check:
            n = int(df[c].isna().sum())
            if n > 0:
                rep.null_counts[c] = n


def validate_file(dataset: str, year: int) -> FileReport:
    path = raw_path(dataset, year)
    rep = FileReport(dataset=dataset, year=year, path=path)

    if not path.exists():
        rep.errors.append(f"file does not exist: {path}")
        return rep

    rep.exists = True
    df = pd.read_parquet(path)
    rep.rows = len(df)
    rep.columns = list(df.columns)

    schema = DATASET_SCHEMAS.get(dataset)
    if schema is None:
        rep.errors.append(f"no schema registered for dataset={dataset}")
        return rep

    _check_provenance(df, rep)
    _check_time_axis(df, rep, schema)
    if schema["format"] == "long":
        _check_long_zones(df, rep, schema)
    else:
        _check_wide_zones(df, rep)
    _check_publish_time(df, rep, schema)
    _check_value_nulls(df, rep, schema)

    return rep


def _print_report(rep: FileReport) -> None:
    status = "OK " if rep.ok and not rep.warnings else ("FAIL" if rep.errors else "WARN")
    print(f"  [{status}] {rep.dataset:>20s}  year={rep.year}  rows={rep.rows:>8,}")
    if rep.min_time is not None:
        print(f"         time range: [{rep.min_time} .. {rep.max_time}]")
    if rep.publish_time_range is not None:
        pt_min, pt_max = rep.publish_time_range
        print(f"         publish time range: [{pt_min} .. {pt_max}]")
    print(f"         zones_present={len(rep.zones_present)}/11")
    if rep.extra_locations:
        print(f"         extra (filtered downstream): {rep.extra_locations}")
    if rep.null_counts:
        for c, n in rep.null_counts.items():
            print(f"         nulls in {c}: {n}")
    for w in rep.warnings:
        print(f"         WARN: {w}")
    for e in rep.errors:
        print(f"         ERR : {e}")


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--datasets", nargs="+", default=list(RAW_DATASETS))
    parser.add_argument("--year", type=int, help="Validate one specific year")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any warning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    years_by_dataset: dict[str, list[int]] = {}
    for ds in args.datasets:
        ds_dir = RAW_DIR / ds
        if not ds_dir.exists():
            continue
        ys = []
        for f in ds_dir.glob("year=*.parquet"):
            try:
                y = int(f.stem.split("=")[1])
                if args.year is None or y == args.year:
                    ys.append(y)
            except (IndexError, ValueError):
                continue
        years_by_dataset[ds] = sorted(ys)

    if not any(years_by_dataset.values()):
        print("No raw files found. Run `python -m nyiso_dart.data.download` first.")
        return 1

    n_err = n_warn = 0
    for ds, years in years_by_dataset.items():
        if not years:
            continue
        print(f"\n[{ds}]  years on disk: {years}")
        for y in years:
            rep = validate_file(ds, y)
            _print_report(rep)
            n_err += len(rep.errors)
            n_warn += len(rep.warnings)

    print(f"\nSummary: {n_err} errors, {n_warn} warnings")
    if n_err > 0:
        return 2
    if args.strict and n_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
