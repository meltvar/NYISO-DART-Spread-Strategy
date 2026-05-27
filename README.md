# NYISO DART Spread Trading Strategy

A systematic trading strategy for the New York ISO day-ahead vs real-time (DART)
price spreads, using zone-specific logistic regression on publicly available
NYISO market data.

The strategy submits virtual INC and DEC bids into the NYISO Day-Ahead Market
when zone-specific spike probabilities clear a validation-tuned threshold.
Predictions use only data settled strictly before each day-ahead gate closure,
so there is no look-ahead bias in the backtest. All reported figures are net of
a $1.00/MWh round-trip transaction cost (NYISO virtual bidding admin charge).

## Headline result (unit size, 1 MWh per trade, net of $1/MWh TC)

| Period | Trades | Total P&L | Win rate | Sharpe (ann.) |
|---|---:|---:|---:|---:|
| 2022–2025 out-of-sample test | 8,693 | $64,232 | 65.1% | 0.921 |
| 2026 YTD live test (Jan 1 – May 15) | 1,782 | $109,278 | 64.7% | — |

The 2026 result is dominated by a five-day winter cold-snap event in late January
that the model identified across all 13 eligible (zone, side) pairs. Feb–May
is essentially flat, consistent with shoulder-season behaviour.

**Safe vs naive.** The pipeline runs two lag specifications side by side. The
"naive" variant uses D−1 same-clock-hour DART (leaks for hours ≥ 4 because
D−1 hour h settles after the 05:00 ET gate for h ≥ 4). The "safe" variant uses
D−2/D−3 lags, which are unconditionally settled before gate closure. The naive
specification inflates 2022–2025 P&L by roughly 3×. All headline figures above
use the leak-free safe variant.

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
a 52-dimensional feature vector built from:
- Day-ahead load forecasts for all 11 zones (44 zone-level features = 4 × 11)
- Lagged DART at 48h and 72h (D−2 and D−3 same clock-hour)
- Lagged load-forecast errors at 48h
- 6 calendar features (hour, month, season, weekend, holiday)
- 2 past-spike-cluster scalars (count of spike hours on D−2)

**Models.** 22 separate zone-side logistic regressions (11 zones × {pos, neg}).
Each predicts the probability of a DART spike — positive (DEC signal,
DART ≥ +$5/MWh) or negative (INC signal, DART ≤ −$30/MWh).

**Threshold tuning.** Per-zone probability cutoffs τ are tuned on the validation
period (2020–2021) to maximise unit-size P&L net of TC. Zones with negative
validation P&L are excluded from trading in the test period.

**Conflict resolution.** Because the 22 classifiers are independent binary
models, both the pos and neg classifiers can fire for the same (timestamp, zone).
When this occurs, only the side with the larger margin above its threshold
(proba − τ) is kept; the other is discarded. This affects roughly 6 pairs per
4-year test period and has negligible P&L impact.

## Repository layout

```
nyiso-dart-strategy/
├── nyiso_dart/                    Core Python package (pipeline)
│   ├── config.py                  constants, paths, splits, helpers
│   ├── data/
│   │   ├── download.py            fetch raw NYISO data via gridstatus
│   │   ├── validate.py            completeness and integrity checks
│   │   └── build.py               merge raw data into the canonical hourly panel
│   ├── features/
│   │   └── build.py               52-feature matrix + 22 labels per hour
│   ├── models/
│   │   ├── splits.py              train/val/test calendar masks
│   │   ├── train.py               fit the 22 logistic regressions
│   │   └── thresholds.py          validation-set τ tuning + eligibility
│   └── backtest/
│       ├── run.py                 apply locked policy to test period
│       └── report.py              tables, plots, safe-vs-naive comparison
├── dashboard/                     Streamlit dashboard (interactive)
│   ├── App.py                     Overview / headline KPIs
│   ├── pages/                     Performance, 2026 Live, Sandbox, Methodology, How It Works
│   └── lib/                       data loaders, plot helpers, metric calculations
├── data/
│   ├── raw/                       (gitignored — regenerable from gridstatus)
│   ├── processed/panel.parquet    canonical hourly panel, all zones 2015–2025
│   └── features/
│       └── predictions_safe.parquet  model probabilities for the safe variant
├── models/                        22 fitted joblib pipelines + threshold tables
├── results/safe/                  trades, cumulative P&L, metrics for safe variant
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
`results/safe/`.

## Interactive dashboard

A Streamlit dashboard sits on top of the pipeline for interactive exploration:

```bash
streamlit run dashboard/App.py
```

Opens in the browser with five pages:

- **Overview** — headline KPIs (P&L, Sharpe, win rate), cumulative equity curve, DEC vs INC split, full zone attribution
- **📊 Performance** — yearly P&L, monthly heatmap, precision/recall scatter, trade payoff distribution
- **🔮 2026 Live Test** — true out-of-sample on post-deployment data, year-over-year comparison, January cold-snap deep dive
- **🎯 Sandbox** — pick any date range from 2015 to present, full statistics computed on demand
- **🔬 Methodology** — bias prevention discipline, safe-vs-naive leakage measurement, audit results
- **📚 How It Works** — DART/INC/DEC explainer, the 11 zones, end-to-end trade timeline

Pre-computed artifacts (`panel.parquet`, `predictions_safe.parquet`, `models/`, `results/`) are
committed to the repo, so the dashboard runs without re-running the pipeline.
Page navigation is near-instant thanks to `@st.cache_data`.

## Bias-prevention design

| Bias | Prevention |
|---|---|
| Look-ahead | All features use D−2/D−3 lags, unconditionally settled before the 05:00 ET gate |
| Train/test contamination | Calendar-date split locked in `config.py`; `assert_disjoint()` called at every entry point |
| Threshold data snooping | τ tuned only on the 2020–2021 validation window; test period never inspected during development |
| Cherry-picking | All 11 zones and both INC/DEC sides reported, including ineligible ones |
| Time-zone / DST bugs | UTC storage; tz-aware lag arithmetic with NaT for ambiguous hours |
| Scaler leak | StandardScaler fitted on train rows only |
| Leakage measurement | Naive (D−1 lag) and safe (D−2 lag) specifications run side by side; empirical P&L gap quantifies leakage (~3×) |

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

- **Market impact** at scale is not modelled. Unit-size (1 MWh) trading has
  zero price impact, but submitting hundreds or thousands of MWh would shift
  the day-ahead clearing price.
- **NYISO LMPs** occasionally get revised after initial settlement. The pipeline
  uses whichever values gridstatus serves at retrieval time.
- **Margin opportunity cost.** NYISO requires collateral (~$2–5/MWh) for virtual
  bidding capacity. This cost-of-capital is not included in the P&L figures.
- **RT uplift charges** spike during stressed periods and are not modelled;
  these slightly compress the very largest payoffs.

## License

MIT (or your preferred license — see `LICENSE`).
