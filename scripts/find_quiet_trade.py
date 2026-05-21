"""Find a representative quiet-day trade from 2025 (no weather event)."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path

DATA    = Path('data')
RESULTS = Path('results')

# Load 2022-2025 trades and panel
trades = pd.read_parquet(RESULTS / 'naive' / 'trades.parquet')
panel  = pd.read_parquet(DATA / 'processed' / 'panel.parquet')

# Focus on 2025, shoulder months (April, October) -- no summer peak, no winter
shoulder = trades[
    (trades['interval_start_local'].dt.year == 2025) &
    (trades['interval_start_local'].dt.month.isin([3, 4, 10, 11]))
].copy()

print(f"Shoulder-season 2025 trades: {len(shoulder)}")
print()

# Show distribution of DART in these trades
print("DART distribution in shoulder 2025 trades:")
print(shoulder['dart'].describe().round(2).to_string())
print()

# Find a trade with modest, clean DART ($10-40 range) -- typical DEC signal
modest = shoulder[
    (shoulder['side'] == 'pos') &
    (shoulder['dart'] >= 12) &
    (shoulder['dart'] <= 40) &
    (shoulder['payoff'] > 0)   # winning trade
].sort_values('dart')

print(f"Modest DEC trades (DART $12-40): {len(modest)}")
print()
print("Sample of modest trades:")
print(modest[['interval_start_local','zone','proba','dart','payoff']].head(20).to_string(index=False))
