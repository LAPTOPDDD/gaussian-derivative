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
N = len(df)

sigma = 5
purge_gap = 3 * sigma
h = 15
num_lags = 60
EVAL_N = 6000
CHUNK = 200
COST = 0.0001

window_size = 3 * sigma
t = np.arange(-window_size, window_size + 1)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
gaussian_derivative = -(t / sigma**2) * gaussian

gauss_target = convolve1d(price, weights=gaussian_derivative, mode='nearest')

cum = np.concatenate([[0.0], np.cumsum(returns)])
fwd_target = np.full(N, np.nan)
fwd_target[:N - h + 1] = cum[h:] - cum[:-h]

X = pd.DataFrame()
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = pd.Series(returns).shift(i)
X = X.fillna(0)

eval_start = max(num_lags, N - EVAL_N)

def walk_forward_predictions(y):
    pred = np.full(N, np.nan)
    last_valid = N - h + 1
    ws = eval_start
    while ws < last_valid:
        we = min(ws + CHUNK, last_valid)
        train_end = ws - purge_gap
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
        model.fit(X.iloc[num_lags:train_end], y.iloc[num_lags:train_end])
        pred[ws:we] = model.predict(X.iloc[ws:we])
        ws = we
    return pred

def strategy_equity(sig, first_t):
    pos = np.sign(sig)
    eq = 1.0
    t = first_t
    last_pos = 0.0
    rpnl = []
    while t + h <= len(returns):
        br = fwd_target[t]
        p = pos[t]
        cost = COST * abs(p - last_pos)
        eq *= 1 + p * br - cost
        rpnl.append(p * br - cost)
        last_pos = p
        t += h
    return eq, np.array(rpnl)

print(f"bars={N}, eval region: {eval_start}..{N-h} ({N-h-eval_start} bars), retrain every {CHUNK}, hold h={h}")
print("=" * 96)

results = {}
eq_curves = {}

for name, y in [('gauss', pd.Series(gauss_target)), ('fwd_return', pd.Series(fwd_target))]:
    pred = walk_forward_predictions(y)
    last_valid = N - h + 1
    idx = np.arange(eval_start, last_valid)
    p = pred[idx]
    f = fwd_target[idx]
    m = ~np.isnan(f)
    p, f = p[m], f[m]

    corr = np.corrcoef(p, f)[0, 1]
    r2 = r2_score(f, p)

    eq, rpnl = strategy_equity(pred, eval_start)
    n_blocks = len(rpnl)
    sharpe = np.mean(rpnl) / (np.std(rpnl) + 1e-12) * np.sqrt(24 * 365 / h)
    blocks = np.arange(n_blocks)
    eq_curve = np.cumprod(1 + rpnl)
    peak = np.maximum.accumulate(eq_curve)
    maxdd = np.max((peak - eq_curve) / peak)
    trades = int(np.sum(np.abs(np.diff(np.sign(np.array([pred[t] for t in range(eval_start, N - h + 1, h)])))) > 0))

    results[name] = (eq, sharpe, maxdd, trades, corr, r2, n_blocks)
    eq_curves[name] = eq_curve
    print(f"{name:<14} final={eq:8.3f}x  sharpe={sharpe:7.2f}  maxDD={maxdd*100:6.1f}%  "
          f"trades={trades:3d}  blocks={n_blocks:3d}  corr(pred,fwd15)={corr:+.3f}  R2(vs fwd15)={r2:+.3f}")

print("=" * 96)

bh = price[eval_start:N] / price[eval_start]

print(f"{'buy&hold':<14} final={bh[-1]:8.3f}x")

fig, ax = plt.subplots(figsize=(15, 7))
x_axis = np.arange(len(eq_curves['gauss']))
for name, eq_curve in eq_curves.items():
    lbl = f"{name} ({eq_curve[-1]:.2f}x, R2={results[name][5]:+.3f})"
    ax.plot(x_axis, eq_curve, label=lbl, linewidth=2)
ax.axhline(bh[-1], color='gray', linestyle=':', linewidth=1.2, label=f'Buy & Hold ({bh[-1]:.2f}x)')
ax.axhline(1.0, color='black', linewidth=0.6)
ax.set_yscale('log')
ax.set_xlabel('block (h=15 bars) in walk-forward test')
ax.set_ylabel('equity multiple (log)')
ax.legend()
ax.set_title('Walk-Forward Equity: Gaussian-derivative target vs Direct forward-return target (h=15)')
plt.savefig('equity_forward_return_target.png')
plt.show()