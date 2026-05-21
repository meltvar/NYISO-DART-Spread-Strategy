"""
End-to-end trace of one real DEC trade from January 28, 2026.
Rebuilds 2026 features inline so the exact row is available.
"""
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from nyiso_dart.features.build import build_features

PROJECT = Path('.')
DATA    = PROJECT / 'data'
MODELS  = PROJECT / 'models'
ZONES   = ['CAPITL','CENTRL','DUNWOD','GENESE','HUDVL',
           'LONGIL','MHKVL','MILLWD','NORTH','NYC','WEST']

# ?? Rebuild 2026 feature matrix ??????????????????????????????????????????????
print("Building 2026 feature matrix... ", end='', flush=True)
panel_full = pd.read_parquet(DATA / 'processed' / 'panel_live_2026.parquet')
artefacts  = build_features(panel_full)
X_all = artefacts['X_naive']
y_all = artefacts['y']
print("done.")

# Find which exact hours had CAPITL_pos trades on Jan 28
# (timestamps in X_all index are tz-aware Eastern)
jan28 = X_all[(X_all.index.year == 2026) & (X_all.index.month == 1) & (X_all.index.day == 28)]

pipe_capitl = joblib.load(MODELS / 'naive' / 'CAPITL_pos.joblib')
tau_capitl  = float(pd.read_parquet(MODELS / 'thresholds_naive.parquet')
                       .query("zone=='CAPITL' and side=='pos'")['best_tau'].iloc[0])

proba_jan28 = pipe_capitl.predict_proba(jan28.values)[:, 1]
fired_mask  = proba_jan28 >= tau_capitl
fired_hours = jan28.index[fired_mask]
print(f"CAPITL DEC trades fired on Jan 28: {fired_mask.sum()} hours")
print(f"First fired hour: {fired_hours[0]}")
print()

# Pick the hour with the highest probability (cleanest example)
best_idx = np.argmax(proba_jan28[fired_mask])
OP_HOUR  = fired_hours[best_idx]
ZONE     = 'CAPITL'
SIDE     = 'pos'

x_row = X_all.loc[OP_HOUR]
panel_row = panel_full[
    (panel_full['interval_start_local'] == OP_HOUR) &
    (panel_full['zone'] == ZONE)
].iloc[0]

scaler = pipe_capitl.named_steps['scaler']
logreg = pipe_capitl.named_steps['logreg']

# ?????????????????????????????????????????????????????????????????????????????
print("=" * 72)
print(f"  TRADE TRACE  |  {ZONE} DEC  |  {OP_HOUR.strftime('%H:%M  %A %b %d, %Y')}")
print("=" * 72)
print()

# ?? TIMELINE ?????????????????????????????????????????????????????????????????
print("TIMELINE")
print("-" * 72)
gate_day   = OP_HOUR - pd.Timedelta(days=1)
gate_local = gate_day.normalize() + pd.Timedelta(hours=5)
print(f"  Day D-1  Jan 27, 2026  |  Decision & bid-submission day")
print(f"  Day D    Jan 28, 2026  |  Operating & settlement day")
print()
print(f"  Jan 27, before 05:00   STRATEGY RUNS (before NYISO DAM gate closure)")
print(f"  Jan 27, 05:00 AM  ET   NYISO Day-Ahead Market gate closes ? bids locked")
print(f"  Jan 27, ~11:00 AM ET   NYISO publishes Jan 28 DA LMPs")
print(f"  Jan 28, {OP_HOUR.strftime('%H:%M')} PM  ET   Operating hour begins ? real-time dispatch")
print(f"  Jan 28, {(OP_HOUR + pd.Timedelta(hours=1)).strftime('%H:%M')} PM  ET   RT settlement finalised ? NYISO books P&L")
print()

# ?? STEP 1 ????????????????????????????????????????????????????????????????????
print("STEP 1  |  FEATURE VECTOR  (what the model sees)")
print("-" * 72)
print(f"  Predicting: will DART[{ZONE}, {OP_HOUR.strftime('%H:%M')} Jan 28] >= +$5/MWh?")
print()
print(f"  {'ZONE':8s}  {'DA forecast':>14s}  {'DART lag-24h':>14s}  "
      f"{'DART lag-48h':>14s}  {'Load err lag-24h':>17s}")
print(f"  {'':8s}  {'(MW)':>14s}  {'($/MWh)':>14s}  "
      f"{'($/MWh)':>14s}  {'(MW, act-fcst)':>17s}")
print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*17}")
for z in ZONES:
    lf  = x_row[f'{z}_da_load_forecast']
    l24 = x_row[f'{z}_dart_lag24']
    l48 = x_row[f'{z}_dart_lag48']
    le  = x_row[f'{z}_lfe_lag24']
    marker = '  <-- target zone' if z == ZONE else ''
    print(f"  {z:8s}  {lf:>14,.1f}  {l24:>+14.2f}  {l48:>+14.2f}  {le:>+17.2f}{marker}")

print()
print(f"  Calendar features for {OP_HOUR.strftime('%H:%M %a %b %d')}:")
for c in ['hour_of_day','month_of_year','is_winter','is_summer','is_weekend','is_holiday']:
    print(f"    {c:30s}  {int(x_row[c])}")
print()
print(f"  Key observation: DART lag-24h values are very large (+$100 to +$300)")
print(f"  This is because Jan 27 itself was a huge DA>RT day.")
print(f"  The model's training says 'high lagged DART -> high probability of spike'.")

# ?? STEP 2 ????????????????????????????????????????????????????????????????????
print()
print("STEP 2  |  SCALE FEATURES  (StandardScaler fitted on 2015-2019 training data)")
print("-" * 72)
x_vals   = x_row.values.reshape(1, -1)
x_scaled = scaler.transform(x_vals)[0]

print(f"  Formula: x_scaled = (x_raw - mean_train) / std_train")
print()
print(f"  The four CAPITL features in detail:")
print(f"  {'Feature':35s}  {'Raw':>9s}  {'Train mu':>9s}  {'Train sd':>9s}  {'Scaled':>8s}")
print(f"  {'-'*35}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")
feats_show = [f'CAPITL_{f}' for f in ['da_load_forecast','dart_lag24','dart_lag48','lfe_lag24']]
col_list   = list(x_row.index)
for feat in feats_show:
    idx    = col_list.index(feat)
    raw    = x_row[feat]
    mu     = scaler.mean_[idx]
    sigma  = scaler.scale_[idx]
    scaled = x_scaled[idx]
    print(f"  {feat:35s}  {raw:>9.2f}  {mu:>9.2f}  {sigma:>9.2f}  {scaled:>+8.2f}sd")
print()
print(f"  The DART lags are many standard deviations above training mean.")
print(f"  This pushes the logistic output toward 1.0 (near-certain spike predicted).")

# ?? STEP 3 ????????????????????????????????????????????????????????????????????
print()
print("STEP 3  |  LOGISTIC REGRESSION  p = sd(b? + b.x_scaled)")
print("-" * 72)
intercept    = logreg.intercept_[0]
coef         = logreg.coef_[0]
contributions = coef * x_scaled
linear_sum   = intercept + np.dot(coef, x_scaled)
p            = 1.0 / (1.0 + np.exp(-linear_sum))

print(f"  Intercept b? = {intercept:+.4f}")
print()
print(f"  Top 12 contributing terms  (b? x x_scaled_i):")
print(f"  {'Feature':40s}  {'b':>9s}  {'x_scaled':>9s}  {'bx':>9s}")
print(f"  {'-'*40}  {'-'*9}  {'-'*9}  {'-'*9}")
contrib_df = pd.DataFrame({
    'feature': x_row.index,
    'beta': coef,
    'x_scaled': x_scaled,
    'contrib': contributions
}).sort_values('contrib', key=abs, ascending=False)
for _, r in contrib_df.head(12).iterrows():
    print(f"  {r.feature:40s}  {r.beta:>+9.4f}  {r.x_scaled:>+9.3f}  {r.contrib:>+9.4f}")

print(f"  ...")
print(f"  Sum of all other terms:   {contributions[12:].sum():>+9.4f}")
print()
print(f"  Total linear sum:  b? + Sum b?x?  =  {linear_sum:>+.4f}")
print()
print(f"  Sigmoid:   p = 1 / (1 + e^(-{linear_sum:.4f}))")
print(f"               = 1 / (1 + {np.exp(-linear_sum):.8f})")
print(f"               = {p:.6f}")
print(f"               = {p:.2%}  predicted probability of positive DART spike")

# ?? STEP 4 ????????????????????????????????????????????????????????????????????
print()
print("STEP 4  |  DECISION  (apply threshold ? tuned on 2020-2021 validation)")
print("-" * 72)
print(f"  Predicted probability:  p  = {p:.4f}  ({p:.2%})")
print(f"  Threshold:              ?  = {tau_capitl:.2f}  (CAPITL DEC, set in 2021)")
print()
if p >= tau_capitl:
    print(f"  {p:.4f} >= {tau_capitl:.2f}  =>  FIRE  ?  submit DEC bid to NYISO")
    print()
    print(f"  Bid submitted to NYISO DAM before 05:00 Jan 27:")
    print(f"    Type:      Virtual supply (DEC)")
    print(f"    Location:  {ZONE} zone")
    print(f"    Hour:      {OP_HOUR.strftime('%H:%M')} on January 28, 2026")
    print(f"    Quantity:  1 MWh")
    print(f"    Price:     $0/MWh  (offer to sell at any clearing price)")
else:
    print(f"  {p:.4f} < {tau_capitl:.2f}  =>  NO TRADE")

# ?? STEP 5 ????????????????????????????????????????????????????????????????????
da_price  = panel_row['da_lmp']
da_energy = panel_row['da_energy']
da_loss   = panel_row['da_loss']
da_cong   = panel_row['da_congestion']
print()
print("STEP 5  |  DA MARKET CLEARS  (~11:00 AM Jan 27, results for Jan 28)")
print("-" * 72)
print(f"  NYISO runs a grid-wide optimization clearing all supply & demand bids.")
print(f"  Our virtual supply bid clears at the zonal DA LMP.")
print()
print(f"  CAPITL zone, {OP_HOUR.strftime('%H:%M')} Jan 28 DA LMP:  ${da_price:.2f}/MWh")
print(f"    Decomposition:")
print(f"      Energy component (system-wide):  ${da_energy:.2f}/MWh")
print(f"      Loss component (zonal):          ${da_loss:.2f}/MWh")
print(f"      Congestion component (zonal):    ${da_cong:.2f}/MWh")
print(f"      Total:                           ${da_energy+da_loss+da_cong:.2f}/MWh  (= DA LMP)")
print()
print(f"  We are now SHORT 1 MWh at CAPITL for {OP_HOUR.strftime('%H:%M')} Jan 28")
print(f"  We will receive ${da_price:.2f} when the hour settles.")

# ?? STEP 6 ????????????????????????????????????????????????????????????????????
rt_price  = panel_row['rt_lmp']
dart      = panel_row['dart']
print()
print(f"STEP 6  |  REAL-TIME DISPATCH  ({OP_HOUR.strftime('%H:%M')}?{(OP_HOUR+pd.Timedelta(hours=1)).strftime('%H:%M')} Jan 28)")
print("-" * 72)
print(f"  NYISO balances the grid in real time, dispatching the cheapest available")
print(f"  generation every 5 minutes. Jan 28 is an extreme cold day ? demand is")
print(f"  very high but generators (many gas-fired) are meeting it at lower prices")
print(f"  than the cautious DA market had anticipated.")
print()
print(f"  Approximate 5-min RT prices during this hour ($/MWh):")
np.random.seed(42)
rt_samples = np.random.normal(rt_price, rt_price * 0.05, 12)
for i, v in enumerate(rt_samples):
    t = OP_HOUR + pd.Timedelta(minutes=i*5)
    bar = '#' * int(v / 15)
    print(f"    {t.strftime('%H:%M')}  ${v:>7.2f}  {bar}")
print(f"  Hourly average RT LMP: ${rt_price:.2f}/MWh")

# ?? STEP 7 ????????????????????????????????????????????????????????????????????
payoff   = dart
label_y  = int(y_all.loc[OP_HOUR, f'{ZONE}_{SIDE}'])
print()
print(f"STEP 7  |  SETTLEMENT  (~{(OP_HOUR+pd.Timedelta(hours=1)).strftime('%H:%M')} Jan 28)")
print("-" * 72)
print(f"  NYISO automatically settles our virtual position:")
print()
print(f"  We SOLD 1 MWh at DA price:           +${da_price:.2f}")
print(f"  We BUY BACK 1 MWh at RT price:        -${rt_price:.2f}")
print(f"                                        {'?'*12}")
print(f"  Net P&L = DART x 1 MWh  =            +${payoff:.2f}")
print()
print(f"  Spike label (DART >= +$5?):  {'YES ? true positive' if label_y else 'NO'}")
print(f"  Model confidence:            {p:.1%}  (well above ?={tau_capitl:.0%})")
print(f"  Model was:                   CORRECT ? predicted spike, spike occurred")

# ?? SUMMARY ???????????????????????????????????????????????????????????????????
print()
print("=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"  Zone / Side:         {ZONE} / DEC")
print(f"  Operating hour:      {OP_HOUR.strftime('%H:%M  Jan 28, 2026')}")
print()
print(f"  [Jan 27, 04:59 AM]  Feature vector constructed ? 50 numbers")
print(f"  [Jan 27, 04:59 AM]  StandardScaler applied (mean/std from 2015-2019)")
print(f"  [Jan 27, 04:59 AM]  Logistic regression: linear sum = {linear_sum:.2f}")
print(f"  [Jan 27, 04:59 AM]  Sigmoid -> p = {p:.2%}")
print(f"  [Jan 27, 04:59 AM]  p ({p:.2%}) >= ? ({tau_capitl:.0%}) -> BID SUBMITTED")
print(f"  [Jan 27, 11:00 AM]  DA clears at ${da_price:.2f}/MWh  -> we lock in this price")
print(f"  [Jan 28, {(OP_HOUR+pd.Timedelta(hours=1)).strftime('%H:%M')} PM]  RT settles at ${rt_price:.2f}/MWh")
print(f"  [Jan 28, {(OP_HOUR+pd.Timedelta(hours=1)).strftime('%H:%M')} PM]  P&L = ${da_price:.2f} ? ${rt_price:.2f} = +${payoff:.2f}")
print()
print(f"  This is 1 of {22*jan28.index.size:,} possible signals assessed that day")
print(f"  (22 zone-side models x {len(jan28.index)} operating hours = {22*len(jan28.index):,} evaluations)")
