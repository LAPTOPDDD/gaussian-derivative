import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price = df['<CLOSE>'].values
returns = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1)).fillna(0).values
hours = df['<TIME>'].str.split(':').str[0].astype(int).values
N = len(df)

START = 24 * 10
EVAL_N = 6000
EVAL0 = N - EVAL_N

def run(alpha, cost, min_abs=0.0):
    ema = np.zeros(24)
    init_n = np.zeros(24)
    eq = np.ones(N)
    wins = np.zeros(N)
    acted = np.zeros(N)
    last_pos = 0.0
    for t in range(START, N):
        h = hours[t]
        if init_n[h] < 1:
            ema[h] = 0.0
        mean_est = ema[h]
        if abs(mean_est) > min_abs:
            pos = np.sign(mean_est)
        else:
            pos = 0.0
        r = returns[t]
        c = cost * abs(pos - last_pos)
        eq[t] = eq[t - 1] * (1 + pos * r - c)
        if pos != 0:
            acted[t] = 1
            wins[t] = int(pos * r > 0)
        if init_n[h] < 1:
            ema[h] = r
            init_n[h] = 1
        else:
            ema[h] = alpha * r + (1 - alpha) * ema[h]
        last_pos = pos
    return eq, wins, acted

def metrics(eq, wins, acted):
    s0 = max(START, EVAL0) - 1
    seg_e = eq[s0:]
    rpnl = np.diff(seg_e) / seg_e[:-1]
    sharpe = np.mean(rpnl) / (np.std(rpnl) + 1e-12) * np.sqrt(24 * 365)
    peak = np.maximum.accumulate(seg_e)
    maxdd = np.max((peak - seg_e) / peak)
    act = acted[max(START, EVAL0):]
    ww = wins[max(START, EVAL0):]
    trades = int(np.sum(act > 0))
    wr = np.mean(ww[act > 0]) if (act > 0).any() else np.nan
    return seg_e[-1], sharpe, maxdd, trades, wr

print(f"bars={N}, eval region: {max(START, EVAL0)}..{N}")
print("=" * 92)
print(f"{'alpha':<7}{'min|mean|':<10}{'costpip':<8}{'final':>9}{'sharpe':>8}{'maxDD%':>8}{'trades':>8}{'winrate%':>9}")
print("-" * 92)

best = None
for alpha in [0.01, 0.02, 0.05]:
    for min_abs in [0.0, 3e-5, 6e-5]:
        for cost in [0.0, 0.00003, 0.0001]:
            eq, wins, acted = run(alpha, cost, min_abs)
            final, sharpe, maxdd, trades, wr = metrics(eq, wins, acted)
            if best is None or final > best[0]:
                best = (final, alpha, cost, min_abs, eq, wins, acted, sharpe, maxdd)
            print(f"{alpha:<7}{min_abs:<10.0e}{int(cost*10000):<8}{final:>9.3f}{sharpe:>8.2f}{maxdd*100:>7.1f}%{trades:>8}{wr*100:>8.1f}%")

final, alpha, cost, min_abs, eq, wins, acted, sharpe, maxdd = best
print("=" * 92)
print(f"BEST: alpha={alpha}, min|mean|={min_abs:.0e}, cost={int(cost*10000)} pips -> final={final:.3f}x, sharpe={sharpe:.2f}, maxDD={maxdd*100:.1f}%")

bh = price[max(START, EVAL0):] / price[max(START, EVAL0)]
print(f"Buy & Hold: {bh[-1]:.3f}x")

ev = np.arange(max(START, EVAL0), N)
seg_eq = eq[ev] / eq[max(START, EVAL0) - 1]
wr_ema = np.full(len(ev), np.nan)
wr = 0.5
for i, t in enumerate(ev):
    if acted[t]:
        wr = 0.02 * wins[t] + 0.98 * wr
    wr_ema[i] = wr

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(np.arange(len(ev)), seg_eq, label=f'Hour-EWMA self-learning ({seg_eq[-1]:.2f}x)', linewidth=2)
ax[0].plot(np.arange(len(ev)), price[ev] / price[ev[0]], label=f'Buy & Hold ({bh[-1]:.2f}x)', alpha=0.7, linestyle=':')
ax[0].axhline(1.0, color='black', linewidth=0.6)
ax[0].set_yscale('log')
ax[0].legend()
ax[0].set_title('Equity: causal hour-of-day self-learning')
ax[1].plot(np.arange(len(ev)), wr_ema, color='green', linewidth=1.5)
ax[1].axhline(0.5, color='black', linewidth=0.6)
ax[1].set_ybound(0.4, 0.6)
ax[1].set_title('Rolling win rate (EWMA)')
plt.tight_layout()
plt.savefig('online_hour_selflearning_equity.png')
plt.show()