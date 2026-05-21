# Exploratory scripts

One-off analysis and trace scripts used during development. Not part of the
production pipeline. The reusable pipeline lives in the `nyiso_dart/` package.

| Script | Purpose |
|---|---|
| `trace_one_trade.py` | Walk a single DEC trade through every stage of the model (features → scaling → logistic regression → decision → settlement) for an extreme winter event |
| `trace_quiet_trade.py` | Same end-to-end trace for a quiet shoulder-season trade |
| `find_quiet_trade.py` | Helper to identify representative shoulder-season trades from the trades log |
| `check_jan2026.py` | Bias-verification checks on January 2026 results (LMP plausibility, decomposition, naive momentum baseline) |
| `tc_analysis.py` | Transaction-cost sensitivity analysis across the 2022–2025 test trades |

Run any script from the project root:

```bash
python scripts/trace_one_trade.py
```

Each script assumes the data pipeline has already been run (`data/processed/`,
`data/features/`, `models/`, `results/` populated).
