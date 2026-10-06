import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')

fe = df.groupby(df['<DATE>']).agg(
    c=('<CLOSE>', 'last'), h=('<HIGH>', 'max'), l=('<LOW>', 'min')).reset_index()
D = len(fe)
dates = pd.to_datetime(fe['<DATE>'], format='%Y.%m.%d').values
day_close = fe['c'].values.astype(float)
day_high = fe['h'].values.astype(float)
day_low = fe['l'].values.astype(float)
day_ret = np.append([0.0], np.log(day_close[1:] / day_close[:-1]))

def ema_cross(span_fast, span_slow):
    fast = pd.Series(day_close).ewm(span=span_fast, adjust=False).mean().values
    slow = pd.Series(day_close).ewm(span=span_slow, adjust=False).mean().values
    sig = np.sign(fast - slow)
    return sig

def donchian(N):
    sig = np.zeros(D)
    pos = 0.0
    for i in range(D):
        if i >= N:
            hh = day_high[i - N:i].max()
            ll = day_low[i - N:i].min()
            if day_close[i] > hh:
                pos = 1.0
            elif day_close[i] < ll:
                pos = -1.0
        sig[i] = pos
    return sig

def apply(sig, cost):
    pos = np.zeros(D)
    pos[1:] = sig[:-1]
    pnl = np.zeros(D)
    flips = 0
    for t in range(1, D):
        delta = abs(pos[t] - pos[t - 1])
        pnl[t] = pos[t] * day_ret[t] - cost * delta
        flips += delta > 0
    eq = np.cumprod(1 + pnl)
    return eq, pnl, flips

def metrics(eq, pnl, flips, tail_days):
    full = eq[-1]
    sharpe = np.mean(pnl[1:]) / (np.std(pnl[1:]) + 1e-12) * np.sqrt(365)
    peak = np.maximum.accumulate(eq)
    maxdd = np.max((peak - eq) / peak)
    tail_eq = eq[-tail_days:] / eq[max(0, D - tail_days - 1)]
    return full, sharpe, maxdd, flips, tail_eq[-1]

variants = [
    ('ema 20/60', ema_cross(20, 60)),
    ('donch 20', donchian(20)),
    ('donch 55', donchian(55)),
    ('donch 100', donchian(100)),
]

print(f"daily bars: {D} ({str(dates[0])[:10]}..{str(dates[-1])[:10]}), eval tail: last 250 days")
print("=" * 96)
print(f"{'variant':<12}{'cost(pip)':<10}{'final':>9}{'sharpe':>8}{'maxDD%':>8}{'flips':>7}{'tail250d':>10}")
print("-" * 96)

results = []
for name, sig in variants:
    for cost_pips in [0.0, 0.3, 1.0]:
        cost = cost_pips * 0.0001
        eq, pnl, flips = apply(sig, cost)
        final, sharpe, maxdd, flips, tail = metrics(eq, pnl, flips, 250)
        results.append((name, cost_pips, eq, pnl, flips, final, sharpe, maxdd, tail))
        print(f"{name:<12}{cost_pips:<10.1f}{final:>9.3f}{sharpe:>8.2f}{maxdd*100:>7.1f}%{flips:>7}{tail:>9.2f}x")

print("=" * 96)
bh = day_close / day_close[0]
bh_tail = bh[-1] / bh[max(0, D - 250 - 1)]
print(f"{'buy&hold':<12}{0.0:<10}{bh[-1]:>9.3f}{'':>8}{'':>8}{'':>7}{bh_tail:>9.2f}x")

costs0 = [r for r in results if r[1] == 0.3]
best = max(costs0, key=lambda r: r[6])
name, cp, eq, pnl, flips, final, sharpe, maxdd, tail = best
print(f"\nBEST @0.3pip cost: {name}, final={final:.3f}x, sharpe={sharpe:.2f}, maxDD={maxdd*100:.1f}%, flips={flips}")

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(dates, eq, label=f'{name} (final {final:.2f}x)', linewidth=2)
ax.plot(dates, bh, label=f'Buy & Hold ({bh[-1]:.2f}x)', alpha=0.7, linestyle=':')
ax.axhline(1.0, color='black', linewidth=0.6)
ax.set_yscale('log')
ax.legend()
ax.set_title(f'Daily Trend-Following (decision at bar open, signal from prev close) - winner {name}')
plt.savefig('trend_following_weekly_equity.png')
plt.show()