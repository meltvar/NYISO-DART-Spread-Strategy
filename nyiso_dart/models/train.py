"""
Fit the 22 zone-side logistic regressions, separately for each feature variant.

Total: 44 fitted models. For each (variant, zone, side) we save:
  - models/{variant}/{zone}_{side}.joblib  -- sklearn Pipeline (scaler+logreg)
  - row-aligned predicted probability vector, contributed to
    data/features/predictions_{variant}.parquet

Bias-prevention design
----------------------
- TRAIN rows only ever come from [TRAIN_START, TRAIN_END]. The test/val masks
  are computed but never used to fit anything.
- The scaler is fit on TRAIN rows only and applied to ALL rows. This prevents
  feature-distribution information from validation/test from leaking into the
  TRAIN-time standardization.
- Rows where any feature is NaN (DST boundary, data-gap month) are dropped at
  fit time. Their predictions still get emitted (using the fitted model on
  whatever non-NaN features they have), but if a row has any NaN we emit NaN
  probability for it and let downstream code handle the gap.

Hyperparameters
---------------
Plain logistic regression with cross-entropy loss. sklearn defaults:
  - penalty = "l2"
  - C       = 1.0
  - solver  = "lbfgs"
  - class_weight = None  (handled downstream via threshold tuning)
  - max_iter = 1000  (raised from default 100 to ensure convergence)

The threshold tuning step (separate module) compensates for class-imbalance
bias in the resulting probabilities by picking per-zone-side cutoffs on
validation data.

Usage
-----
    python -m nyiso_dart.models.train
    python -m nyiso_dart.models.train --variant safe
    python -m nyiso_dart.models.train --variant naive
    python -m nyiso_dart.models.train --variants safe naive  # default
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nyiso_dart.config import FEATURES_DIR, MODELS_DIR, ZONES
from nyiso_dart.models.splits import assert_disjoint, test_mask, train_mask, val_mask

log = logging.getLogger(__name__)

SIDES = ("pos", "neg")
VARIANTS = ("safe", "naive")


def _x_path(variant: str) -> Path:
    return FEATURES_DIR / f"X_{variant}.parquet"


def _model_path(variant: str, zone: str, side: str) -> Path:
    return MODELS_DIR / variant / f"{zone}_{side}.joblib"


def _predictions_path(variant: str) -> Path:
    return FEATURES_DIR / f"predictions_{variant}.parquet"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight=None,
                ),
            ),
        ]
    )


def _train_one(
    X: pd.DataFrame,
    y: pd.Series,
    tr_mask: pd.Series,
) -> tuple[Pipeline, pd.Series, dict]:
    """Fit one model on rows where tr_mask is True AND no NaN in (X row, y).
    Return (pipeline, predicted_proba_for_all_rows, stats)."""
    # Eligible TRAIN rows
    full_mask = tr_mask.values & ~X.isna().any(axis=1).values & ~y.isna().values
    X_tr = X.loc[full_mask]
    y_tr = y.loc[full_mask].astype(int)

    n_pos = int(y_tr.sum())
    n_tr = int(len(y_tr))
    stats = {
        "n_train_rows": n_tr,
        "n_train_pos": n_pos,
        "pos_rate_train": float(n_pos / n_tr) if n_tr else 0.0,
    }

    if n_tr == 0 or n_pos == 0 or n_pos == n_tr:
        # Degenerate: no rows, all-positive, or all-negative. Emit constant predictions.
        log.warning("  -> degenerate label split (n=%d, pos=%d); using base-rate prediction",
                    n_tr, n_pos)
        base_rate = stats["pos_rate_train"]
        proba = pd.Series(base_rate, index=X.index)
        return None, proba, stats

    pipe = _build_pipeline()
    pipe.fit(X_tr.values, y_tr.values)

    # Predict for all rows; NaN-row predictions are NaN.
    proba = pd.Series(np.nan, index=X.index, dtype="float64")
    full_X_mask = ~X.isna().any(axis=1).values
    proba.loc[full_X_mask] = pipe.predict_proba(X.loc[full_X_mask].values)[:, 1]
    return pipe, proba, stats


def train_variant(variant: str) -> dict:
    log.info("== Training variant: %s ==", variant)
    X = pd.read_parquet(_x_path(variant))
    y = pd.read_parquet(FEATURES_DIR / "y.parquet")

    if not X.index.equals(y.index):
        raise RuntimeError(f"X and y indices differ for variant={variant!r}")

    idx = X.index
    assert_disjoint(idx)
    tr = train_mask(idx)
    va = val_mask(idx)
    te = test_mask(idx)
    log.info("  rows: train=%d val=%d test=%d total=%d",
             int(tr.sum()), int(va.sum()), int(te.sum()), len(idx))

    variant_dir = MODELS_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    pred_cols = {}
    fit_stats: dict[str, dict] = {}

    for zone in ZONES:
        for side in SIDES:
            label_col = f"{zone}_{side}"
            log.info("  fitting %s (variant=%s)...", label_col, variant)
            pipe, proba, stats = _train_one(X, y[label_col], tr)
            if pipe is not None:
                joblib.dump(pipe, _model_path(variant, zone, side))
            pred_cols[label_col] = proba.values
            fit_stats[label_col] = stats

    preds = pd.DataFrame(pred_cols, index=idx)
    preds.to_parquet(_predictions_path(variant))
    log.info("  wrote predictions: %s shape=%s", _predictions_path(variant), preds.shape)

    return {
        "variant": variant,
        "n_models": int(sum(1 for s in fit_stats.values() if s["n_train_pos"] > 0)),
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "fit_stats": fit_stats,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Fit 22 zone-side logistic regressions per variant")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        choices=list(VARIANTS),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    manifest: dict[str, dict] = {}
    for v in args.variants:
        manifest[v] = train_variant(v)

    out = MODELS_DIR / "train_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Manifest written to %s", out)

    # Brief summary
    for v, m in manifest.items():
        print(f"\n[{v}] models fitted: {m['n_models']}/22")
        print(f"     train/val/test rows: {m['rows']}")
        # Show positive rate per (zone, side)
        rows = []
        for label, s in m["fit_stats"].items():
            rows.append((label, s["n_train_rows"], s["n_train_pos"], s["pos_rate_train"]))
        df = pd.DataFrame(rows, columns=["label", "n_train", "n_pos", "pos_rate"])
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
