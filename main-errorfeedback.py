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

resid_by_bar = np.full(total_len, np.nan)
raw_parts, corrected_parts, true_parts = [], [], []

print(f"total bars: {total_len}, test windows: {n_windows}")
print("=" * 84)

for w, (ws, we) in enumerate(window_bounds, start=1):
    train_end = ws - purge_gap
    X_train = X.iloc[num_lags:train_end]
    y_train = y_all.iloc[num_lags:train_end]
    X_test = X.iloc[ws:we]
    y_test = y_all.iloc[ws:we].values

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    raw = model.predict(X_test)
    bars = np.arange(ws, we)

    prev_block = np.arange(test_start, ws)
    prev_block = prev_block[~np.isnan(resid_by_bar[prev_block])]
    has_prev = len(prev_block) >= 5

    if has_prev:
        r = resid_by_bar[prev_block]
        n_prev = len(r)
        prev_rmse = np.sqrt(np.mean(r**2))
        prev_mae = np.mean(np.abs(r))
        prev_bias = np.mean(r)
        prev_max = np.max(np.abs(r))
        prev_std = np.std(r)

        slope = 0.0
        if len(prev_block) >= 8 and prev_std > 1e-9:
            slope, _ = np.polyfit(prev_block, r, 1)

        mean_part = 0.6 * prev_bias
        trend_part = 0.5 * slope * (bars - ws)
        r_hat = np.clip(mean_part + trend_part, -prev_max, prev_max)
    else:
        n_prev = prev_rmse = prev_mae = prev_bias = prev_max = prev_std = slope = 0.0
        mean_part = 0.0
        r_hat = np.zeros(len(bars))

    corrected = raw - r_hat

    resid_by_bar[bars] = y_test - corrected

    r2_raw = r2_score(y_test, raw)
    r2_corr = r2_score(y_test, corrected)

    raw_parts.append(raw)
    corrected_parts.append(corrected)
    true_parts.append(y_test)

    if has_prev:
        print(f"window {w:2d} [{ws:5d}..{we:5d}] | prev win analysis: n={n_prev:2d} RMSE={prev_rmse:.4f} "
              f"MAE={prev_mae:.4f} bias={prev_bias:+.4f} slope={slope:+.2e} | corr: mean={mean_part:+.4f} "
              f"trend={0.5*slope:+.2e} | R2 raw={r2_raw:+.3f} corr={r2_corr:+.3f}")
    else:
        print(f"window {w:2d} [{ws:5d}..{we:5d}] | no previous window yet | R2 raw={r2_raw:+.3f} corr={r2_corr:+.3f}")

y_true_all = np.concatenate(true_parts)
y_raw_all = np.concatenate(raw_parts)
y_corr_all = np.concatenate(corrected_parts)

r2_raw_total = r2_score(y_true_all, y_raw_all)
r2_corr_total = r2_score(y_true_all, y_corr_all)
rmse_raw_total = np.sqrt(np.mean((y_true_all - y_raw_all)**2))
rmse_corr_total = np.sqrt(np.mean((y_true_all - y_corr_all)**2))

print("=" * 84)
print(f"RAW      Walk-Forward R2 on Test: {r2_raw_total:.4f}  RMSE={rmse_raw_total:.4f}")
print(f"CORRECTED Walk-Forward R2 on Test: {r2_corr_total:.4f}  RMSE={rmse_corr_total:.4f}")
print(f"Causal Ceiling: {causal_ceiling:.4f}")
print(f"Efficiency raw / corrected: {r2_raw_total/causal_ceiling:.2%} / {r2_corr_total/causal_ceiling:.2%}")

fig, ax = plt.subplots(figsize=(15, 7))
x_axis = np.arange(test_start, total_len)
ax.plot(x_axis, y_true_all, label='Actual (Non-Causal)', alpha=0.7)
ax.plot(x_axis, y_raw_all, label=f'Raw (R2={r2_raw_total:.3f})', alpha=0.6, linestyle=':')
ax.plot(x_axis, y_corr_all, label=f'Error-Corrected (R2={r2_corr_total:.3f})', alpha=0.8, linestyle='--')
for ws, we in window_bounds:
    ax.axvline(ws, color='gray', linewidth=0.6, linestyle=':')
ax.axhline(0, color='black', linewidth=0.5)
ax.legend()
ax.set_title(f'Rolling Walk-Forward Gaussian Derivative with Error Feedback from Previous Window\n'
             f'Ceiling={causal_ceiling:.3f}')
plt.savefig('gaussian_derivative_rolling_errorcorrection.png')
plt.show()