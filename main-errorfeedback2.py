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

causal_ceiling = 0.506
GATE_T = 1.3
MAX_MEMORY_BARS = 40

oos_resid = np.full(total_len, np.nan)
raw_parts, corrected_parts, true_parts = [], [], []

print(f"total bars: {total_len}, test windows: {n_windows}, bias gate t>={GATE_T}")
print("=" * 88)

for w, (ws, we) in enumerate(window_bounds, start=1):
    train_end = ws - purge_gap
    X_train = X.iloc[num_lags:train_end]
    y_train = y_all.iloc[num_lags:train_end]
    X_test = X.iloc[ws:we]
    y_test = y_all.iloc[ws:we].values

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)

    if train_end < ws:
        band_pred = model.predict(X.iloc[train_end:ws])
        band_true = y_all.iloc[train_end:ws].values
        oos_resid[train_end:ws] = band_true - band_pred

    raw = model.predict(X_test)

    hist_bars = np.where(~np.isnan(oos_resid))[0]
    hist_bars = hist_bars[hist_bars >= ws - MAX_MEMORY_BARS]
    bias_hat = 0.0
    if len(hist_bars) >= 10:
        r = oos_resid[hist_bars]
        n = len(r)
        m = np.mean(r)
        s = np.std(r)
        t_stat = np.sqrt(n) * abs(m) / (s + 1e-12)
        if t_stat > GATE_T:
            js_shrink = 1.0 - GATE_T**2 / t_stat**2
            bias_hat = m * max(js_shrink, 0.0)
        print(f"window {w:2d} [{ws:5d}..{we:5d}] | oos n={n:2d} mean={m:+.2e} std={s:.2e} "
              f"t={t_stat:6.2f} | applied bias={bias_hat:+.2e}")
    else:
        print(f"window {w:2d} [{ws:5d}..{we:5d}] | oos n={len(hist_bars):2d} (too few) | applied bias=0")

    corrected = raw - bias_hat
    oos_resid[ws:we] = y_test - raw

    raw_parts.append(raw)
    corrected_parts.append(corrected)
    true_parts.append(y_test)

y_true_all = np.concatenate(true_parts)
y_raw_all = np.concatenate(raw_parts)
y_corr_all = np.concatenate(corrected_parts)

r2_raw_total = r2_score(y_true_all, y_raw_all)
r2_corr_total = r2_score(y_true_all, y_corr_all)
rmse_raw_total = np.sqrt(np.mean((y_true_all - y_raw_all)**2))
rmse_corr_total = np.sqrt(np.mean((y_true_all - y_corr_all)**2))

print("=" * 88)
print(f"RAW       Walk-Forward R2 on Test: {r2_raw_total:.4f}  RMSE={rmse_raw_total:.6f}")
print(f"CORRECTED Walk-Forward R2 on Test: {r2_corr_total:.4f}  RMSE={rmse_corr_total:.6f}")
print(f"Causal Ceiling: {causal_ceiling:.4f}")
print(f"Efficiency raw / corrected: {r2_raw_total/causal_ceiling:.2%} / {r2_corr_total/causal_ceiling:.2%}")

fig, ax = plt.subplots(figsize=(15, 7))
x_axis = np.arange(test_start, total_len)
ax.plot(x_axis, y_true_all, label='Actual (Non-Causal)', alpha=0.7)
ax.plot(x_axis, y_raw_all, label=f'Raw (R2={r2_raw_total:.3f})', alpha=0.6, linestyle=':')
ax.plot(x_axis, y_corr_all, label=f'Bias-Corrected (R2={r2_corr_total:.3f})', alpha=0.8, linestyle='--')
for ws, we in window_bounds:
    ax.axvline(ws, color='gray', linewidth=0.6, linestyle=':')
ax.axhline(0, color='black', linewidth=0.5)
ax.legend()
ax.set_title(f'Rolling Walk-Forward Gaussian Derivative with Significance-Gated Bias Correction\n'
             f'Ceiling={causal_ceiling:.3f}')
plt.savefig('gaussian_derivative_rolling_bias_corrected.png')
plt.show()