import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from sklearn.metrics import r2_score

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price = df['<CLOSE>'].values
returns = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1)).fillna(0).values

sigma = 5
purge_gap = 3 * sigma
test_len = 200
n_windows = 15
window_len = test_len // n_windows
num_lags = 60

window_size = 3 * sigma
t = np.arange(-window_size, window_size + 1)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
gaussian_derivative = -(t / sigma**2) * gaussian

conv_derivative = convolve1d(price, weights=gaussian_derivative, mode='nearest')
y_all = pd.Series(conv_derivative)

X = pd.DataFrame()
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = pd.Series(returns).shift(i)
X = X.fillna(0)

total_len = len(df)
test_start = total_len - test_len
window_bounds = []
for w in range(n_windows):
    ws = test_start + w * window_len
    we = test_start + (w + 1) * window_len
    if w == n_windows - 1:
        we = total_len
    window_bounds.append((ws, we))

pred_all = np.full(total_len, np.nan)
for ws, we in window_bounds:
    train_end = ws - purge_gap
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
    model.fit(X.iloc[num_lags:train_end], y_all.iloc[num_lags:train_end])
    pred_all[ws:we] = model.predict(X.iloc[ws:we])

sig = pred_all[test_start:total_len]
r = returns[test_start:total_len]
sig_std = np.std(sig)

CORR_N = len(sig)
print(f"corr(signal, next-bar return): {np.corrcoef(sig, r)[0,1]:+.3f} (n={CORR_N})")

BAR_PER_YEAR = 24 * 365
COST = 0.0001

def block_strategy(h, th):
    pos_vec, ret_vec, eq_prc = [], [], []
    t = 0
    while t + h <= len(r):
        block_ret = r[t:t + h].sum()
        s = sig[t]
        pos = 0.0
        if abs(s) > th * sig_std:
            pos = np.sign(s)
        cost = COST * abs(pos - (np.sign(sig[t - h]) if t >= h and abs(sig[t - h]) > th * sig_std else 0.0))
        pos_vec.append(pos)
        ret_vec.append(block_ret)
        eq_prc.append(1 + pos * block_ret - cost)
        t += h
    if t < len(r):
        block_ret = r[t:].sum()
        pos = np.sign(sig[t]) if abs(sig[t]) > th * sig_std else 0.0
        pos_vec.append(pos)
        ret_vec.append(block_ret)
        eq_prc.append(1 + pos * block_ret)
    pos_vec = np.array(pos_vec)
    ret_vec = np.array(ret_vec)
    eq_prc = np.array(eq_prc)
    eq = np.cumprod(eq_prc)
    rpnl = pos_vec * ret_vec
    block_years = BAR_PER_YEAR / h
    sharpe = np.mean(rpnl) / (np.std(rpnl) + 1e-12) * np.sqrt(block_years)
    peak = np.maximum.accumulate(eq)
    maxdd = np.max((peak - eq) / peak)
    trades = int(np.sum(np.abs(np.diff(pos_vec)) > 0))
    return eq, sharpe, maxdd, trades

print("=" * 82)
print(f"{'h':<5}{'theta':<7}{'final_mult':>10}{'ret%':>9}{'sharpe':>8}{'maxDD%':>8}{'trades':>7}")
print("-" * 82)

results = []
for h in [1, 3, 5, 8, 10, 15]:
    for th in [0.0, 0.5, 1.0]:
        eq, sharpe, maxdd, trades = block_strategy(h, th)
        results.append((h, th, eq, sharpe, maxdd, trades))
        print(f"{h:<5}{th:<7}{eq[-1]:>10.3f}{100*(eq[-1]-1):>8.2f}%{sharpe:>8.2f}{maxdd*100:>7.1f}%{trades:>7}")

best = max(results, key=lambda x: x[2][-1])
print("=" * 82)
h_b, th_b, eq_b, sh_b, dd_b, tr_b = best
print(f"BEST: h={h_b}, theta={th_b}, final={eq_b[-1]:.3f}x, sharpe={sh_b:.2f}, maxDD={dd_b*100:.1f}%, trades={tr_b}")

bh = price[test_start:total_len]
bh_curve = bh / bh[0]

def to_bars(eq, h):
    out = np.full(len(r), 1.0)
    t = 0
    for i, v in enumerate(eq):
        end = min(t + h, len(r))
        out[t:end] = v
        t = end
    return out

fig, ax = plt.subplots(figsize=(15, 7))
x_axis = np.arange(len(r))
ax.plot(x_axis, bh_curve, label=f'Buy & Hold ({bh_curve[-1]:.3f}x)', alpha=0.6, linestyle=':')
for h in [3, 5, 10, 15]:
    cands = [res for res in results if res[0] == h]
    br = max(cands, key=lambda x: x[2][-1])
    curve = to_bars(br[2], h)
    lab = f'h={h} θ={br[1]} ({curve[-1]:.2f}x)'
    ax.plot(x_axis, curve, label=lab,
            linewidth=2.4 if (h == h_b) else 1.2,
            alpha=0.9 if (h == h_b) else 0.5)
ax.set_yscale('log')
ax.set_xlabel('bar in test')
ax.set_ylabel('equity multiple (log)')
ax.legend()
ax.set_title(f'Equity: block-hold usage of the predicted Gaussian derivative (h-bar hold, walk-forward signal)\n'
             f'Winner h={h_b} θ={th_b} final={eq_b[-1]:.2f}x Sharpe={sh_b:.2f}')
plt.savefig('equity_test_block_hold.png')
plt.show()