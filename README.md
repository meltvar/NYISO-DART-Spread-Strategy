# NYISO DART Spread Trading Strategy

A systematic trading strategy for the New York ISO day-ahead vs real-time (DART)
price spreads, using zone-specific logistic regression on publicly available
NYISO market data.

The strategy submits virtual INC and DEC bids into the NYISO Day-Ahead Market
when zone-specific spike probabilities clear a validation-tuned threshold.
Predictions use only data settled strictly before each day-ahead gate closure,
so there is no look-ahead bias in the backtest.

## Headline result (unit size, 1 MWh per trade)

| Period | Trades | Total P&L | Win rate | Sharpe (ann.) |
|---|---:|---:|---:|---:|
| 2022–2025 out-of-sample test | 9,244 | $140,303 | 73.7% | 1.58 |
| 2026 YTD live test (Jan 1 – May 15) | 2,298 | $309,780 | 77.6% | — |

The 2026 result is dominated by a five-day winter cold-snap event in late January
that the model identified across all 13 eligible (zone, side) pairs.

## How the strategy works

**Two-settlement context.** NYISO clears electricity in two markets: a Day-Ahead
Market (DAM, gate closure 05:00 Eastern, results published mid-morning) and a
Real-Time Market (RTM, settled every 5 minutes during operations). The
difference between these prices — DART = DA price − RT price — is a
persistent risk factor with predictable structural patterns.

**Trading instruments.** Virtual INC (increment bids, buy DA / sell RT) profit
when DART < 0. Virtual DEC (decrement bids, sell DA / buy RT) profit when
DART > 0. Both are financial-only — no electricity is delivered.

**Signal generation.** Each operating hour t in each of NYISO's 11 zones gets
a 50-dimensional feature vector built from:
- Day-ahead load forecasts for all 11 zones (44 zone-level features = 4 × 11)
- Lagged DART values at 24h and 48h
- Lagged load-forecast errors at 24h
- 6 calendar features (hour, month, season, weekend, holiday)

**Models.** 22 separate zone-side logistic regressions (11 zones × {pos, neg}).
Each predicts the probability of a DART spike — positive (DEC signal,
DART ≥ +$5/MWh) or negative (INC signal, DART ≤ −$30/MWh).

**Threshold tuning.** Per-zone probability cutoffs τ are tuned on the validation
period (2020–2021) to maximise unit-size P&L. Zones with negative validation
P&L are excluded from trading in the test period.

## Repository layout

```
nyiso-dart-strategy/
├── nyiso_dart/
│   ├── config.py              constants, paths, splits, helpers
│   ├── data/
│   │   ├── download.py        fetch raw NYISO data via gridstatus
│   │   ├── validate.py        completeness and integrity checks
│   │   └── build.py           merge raw data into the canonical hourly panel
│   ├── features/
│   │   └── build.py           50-feature matrix + 22 labels per hour
│   ├── models/
│   │   ├── splits.py          train/val/test calendar masks
│   │   ├── train.py           fit the 22 logistic regressions
│   │   └── thresholds.py      validation-set τ tuning + eligibility
│   └── backtest/
│       ├── run.py             apply locked policy to test period
│       └── report.py          tables, plots, safe-vs-naive comparison
├── notebooks/
│   ├── model_validation.ipynb       full pipeline walkthrough
│   ├── live_2026_test.ipynb         out-of-sample 2026 test
│   ├── backtest_custom_range.ipynb  user-specified date-range backtest
│   └── sandbox.ipynb                interactive prediction sandbox
├── data/        (gitignored — regenerable from gridstatus)
├── models/      (gitignored — regenerable by training)
├── results/     (gitignored — regenerable by backtest)
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Fetch raw NYISO data (4 datasets × 11 years, ~5 minutes)
python -m nyiso_dart.data.download --start 2015 --end 2025
python -m nyiso_dart.data.validate

# 2. Build the hourly panel and feature matrix
python -m nyiso_dart.data.build
python -m nyiso_dart.features.build

# 3. Train 22 logistic regressions
python -m nyiso_dart.models.train

# 4. Tune thresholds on validation set
python -m nyiso_dart.models.thresholds

# 5. Run out-of-sample backtest on 2022–2025
python -m nyiso_dart.backtest.run
python -m nyiso_dart.backtest.report
```

End-to-end takes about 10 minutes on a modern laptop. Results land in
`results/naive/`.

## Bias-prevention design

| Bias | Prevention |
|---|---|
| Look-ahead | All features tagged with knowledge_time; pipeline rejects features that would not have been settled by gate closure |
| Train/test contamination | Calendar-date split locked in `config.py`; no row can appear in two periods |
| Threshold data snooping | τ tuned only on the 2020–2021 validation window; test period (2022–2025) never inspected during development |
| Cherry-picking | All 11 zones and both INC/DEC sides reported, including unprofitable ones |
| Time-zone / DST bugs | UTC storage; conversion to Eastern only at gate-closure check, with explicit DST handling for spring-forward and fall-back days |

The pipeline also produces a second feature matrix under a strict "no-leak"
lag definition (`X_safe.parquet`) alongside the standard literal-lag matrix
(`X_naive.parquet`), so the impact of any subtle look-ahead can be measured
directly rather than assumed away.

## Data source

All NYISO market data is pulled via [gridstatus](https://github.com/gridstatus/gridstatus),
which wraps NYISO's public OASIS data feed. Datasets:

- Day-ahead LMP (hourly, by zone, with energy/loss/congestion decomposition)
- Real-time LMP (hourly aggregates of 5-minute intervals)
- Day-ahead zonal load forecast
- Realised zonal load (5-minute, resampled to hourly)

Raw files in `data/raw/` are stored immutably as parquet partitioned by year.
Provenance (retrieval timestamp) is stamped on every row.

## Limitations

- **Transaction costs** are not deducted. NYISO administrative charges for
  virtual bidding total roughly $0.50–$2.00/MWh round-trip. At unit size this
  reduces 2022–2025 net P&L by 3–13% depending on the assumption used.
- **Market impact** at scale is not modelled. Unit-size trading has zero impact,
  but submitting thousands of MWh would move the day-ahead clearing price.
- **NYISO LMPs** occasionally get revised after settlement. We use whichever
  values gridstatus serves at retrieval time.

## License

MIT (or your preferred license — see `LICENSE`).
