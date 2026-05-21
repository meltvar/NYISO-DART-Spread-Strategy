import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')

trades = pd.read_parquet('results/naive/trades.parquet')

print('=== PAYOFF DISTRIBUTION (all 9,244 trades, 2022-2025) ===')
print(trades['payoff'].describe().round(2).to_string())
print()

buckets = [
    (0,    1,    'under $1   (below NYISO fee)'),
    (1,    2,    '$1 - $2    (breaks even at $1 cost)'),
    (2,    5,    '$2 - $5    (thin margin after cost)'),
    (5,   15,    '$5 - $15   (small win)'),
    (15,  50,    '$15 - $50  (moderate win)'),
    (50, 500,    '$50 - $500 (large win)'),
    (500, 1e9,   'over $500  (extreme event)'),
]
winning = trades[trades['payoff'] > 0]
print('=== WINNING TRADES BY PAYOFF BUCKET (n=%d) ===' % len(winning))
print('   %-30s  %5s  %4s   %12s   %6s' % ('Range','Count','Pct','Total PnL','Of total'))
print('   ' + '-'*70)
total_pnl = winning['payoff'].sum()
for lo, hi, label in buckets:
    sub = winning[(winning['payoff'] >= lo) & (winning['payoff'] < hi)]
    n = len(sub); pnl = sub['payoff'].sum()
    if n == 0: continue
    print('   %-30s  %5d  %4.1f%%  %12,.2f   %6.1f%%' % (label, n, 100*n/len(winning), pnl, 100*pnl/total_pnl))
print()

total_gross = trades['payoff'].sum()
n_total = len(trades)
print('=== TRANSACTION COST SCENARIOS (unit size: 1 MWh per trade) ===')
print('   Gross P&L: $%s  |  Trades: %d' % (f'{total_gross:,.2f}', n_total))
print()
print('   %-10s  %-12s  %-12s  %-8s  %-14s' % ('Cost/MWh','Total cost','Net PnL','Drag','Trades turned neg'))
print('   ' + '-'*65)
for tc in [0.50, 1.00, 2.00]:
    net_per = trades['payoff'] - tc
    killed  = (net_per <= 0).sum() - (trades['payoff'] <= 0).sum()
    killed  = max(0, killed)
    total_cost = tc * n_total
    net_pnl    = total_gross - total_cost
    drag       = 100 * total_cost / total_gross
    print('   $%-9.2f  $%-11,.2f  $%-11,.2f  %6.1f%%  %d (%4.1f%%)' % (
        tc, total_cost, net_pnl, drag, killed, 100*killed/n_total))
print()

losing = trades[trades['payoff'] < 0]
print('=== LOSING TRADES: already negative before any cost ===')
print('   Count: %d  |  Total loss: $%s' % (len(losing), f'{losing.payoff.sum():,.2f}'))
print('   Average loss per trade: $%.2f' % losing['payoff'].mean())
print()

print('=== IMPACT ON THE TWO TRACED TRADES ===')
cases = [('Quiet day  Apr 10 CAPITL', 12.20), ('Extreme event Jan 28 CAPITL', 634.45)]
for name, gross in cases:
    print('   %s  gross=$%.2f' % (name, gross))
    for tc in [0.50, 1.00, 2.00]:
        net  = gross - tc
        drag = 100 * tc / gross
        print('     tc=$%.2f  ->  net=$%.2f  (drag=%.1f%%)' % (tc, net, drag))
    print()

print('=== BREAK-EVEN ANALYSIS ===')
print('   A trade must earn MORE than the transaction cost to be profitable.')
for tc in [0.50, 1.00, 2.00]:
    below_cost = winning[winning['payoff'] < tc]
    print('   At $%.2f/MWh cost: %d winning trades (%.1f%%) become breakeven or loss' % (
        tc, len(below_cost), 100*len(below_cost)/len(winning)))
print()

print('=== NYISO VIRTUAL BID CHARGES (approximate, from NYISO tariffs) ===')
charges = [
    ('Market Administration Charge',   '~$0.10-0.30/MWh', 'charged on all energy schedules'),
    ('Real-Time Uplift allocation',     '~$0.10-1.50/MWh', 'variable, elevated in stressed periods'),
    ('Financial security requirement',  'margin posted',   'capital tied up, opportunity cost'),
    ('Data / infrastructure',           'fixed overhead',  'gridstatus, compute, ops'),
    ('Total round-trip estimate',       '~$0.50-2.00/MWh', 'practitioner rule of thumb'),
]
for item, cost, note in charges:
    print('   %-35s  %-18s  %s' % (item, cost, note))
