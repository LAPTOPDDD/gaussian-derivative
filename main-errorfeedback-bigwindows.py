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
n_windows = 5
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

band_resid = np.full(total_len, np.nan)
true_parts = []
raw_parts = []
persist_parts = []
meta_parts = []

bb_history = []
wb_history = []
last_win_bias = 0.0
b_meta = 0.0

print(f"total bars: {total_len}, test windows: {n_windows} x {window_len} bars")
print("=" * 100)

for w, (ws, we) in enumerate(window_bounds, start=1):
    train_end = ws - purge_gap
    X_train = X.iloc[num_lags:train_end]
    y_train = y_all.iloc[num_lags:train_end]
    X_test = X.iloc[ws:we]
    y_test = y_all.iloc[ws:we].values

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)

    band_bias = 0.0
    if train_end < ws:
        band_pred = model.predict(X.iloc[train_end:ws])
        band_true = y_all.iloc[train_end:ws].values
        band_resid[train_end:ws] = band_true - band_pred
        band_bias = np.mean(band_resid[train_end:ws])

    raw = model.predict(X_test)

    bias_persist = last_win_bias
    bias_meta = b_meta * band_bias if (len(bb_history) >= 3 and abs(band_bias) > 1e-12) else 0.0

    pred_persist = raw - bias_persist
    pred_meta = raw - bias_meta

    win_bias_true = float(np.mean(y_test - raw))
    bb_history.append(band_bias)
    wb_history.append(win_bias_true)
    last_win_bias = win_bias_true

    if len(bb_history) >= 3:
        bb = np.array(bb_history[:-1])
        wb = np.array(wb_history[:-1])
        denom = np.sum(bb**2)
        b_meta = np.sum(bb * wb) / denom if denom > 1e-12 else 0.0

    true_parts.append(y_test)
    raw_parts.append(raw)
    persist_parts.append(pred_persist)
    meta_parts.append(pred_meta)

    print(f"window {w:2d} [{ws:5d}..{we:5d}] | band_bias={band_bias:+.2e} win_bias={win_bias_true:+.2e} "
          f"| persist_corr={bias_persist:+.2e} meta_corr={bias_meta:+.2e} (b_meta={b_meta:+.3f})")

y_true_all = np.concatenate(true_parts)
y_raw_all = np.concatenate(raw_parts)
y_persist_all = np.concatenate(persist_parts)
y_meta_all = np.concatenate(meta_parts)

bb_arr = np.array(bb_history)
wb_arr = np.array(wb_history)
sign_agree = np.mean(np.sign(bb_arr) == np.sign(wb_arr)) if len(bb_arr) else np.nan
corr_bb_wb = np.corrcoef(bb_arr, wb_arr)[0, 1] if len(bb_arr) > 1 else np.nan

r2_raw = r2_score(y_true_all, y_raw_all)
r2_persist = r2_score(y_true_all, y_persist_all)
r2_meta = r2_score(y_true_all, y_meta_all)
rmse_raw = np.sqrt(np.mean((y_true_all - y_raw_all)**2))
rmse_persist = np.sqrt(np.mean((y_true_all - y_persist_all)**2))
rmse_meta = np.sqrt(np.mean((y_true_all - y_meta_all)**2))

print("=" * 100)
print(f"sign agreement band_bias vs next win_bias: {sign_agree:.2%} | corr={corr_bb_wb:+.3f}")
print(f"RAW        R2={r2_raw:.4f}   RMSE={rmse_raw:.6f}")
print(f"PERSIST    R2={r2_persist:.4f} RMSE={rmse_persist:.6f}  (subtract last window mean error)")
print(f"META-CALIB R2={r2_meta:.4f}  RMSE={rmse_meta:.6f}  (slope-validated correction)")
print(f"Causal Ceiling: {causal_ceiling:.4f}")

fig, ax = plt.subplots(figsize=(15, 7))
x_axis = np.arange(test_start, total_len)
ax.plot(x_axis, y_true_all, label='Actual (Non-Causal)', alpha=0.7)
ax.plot(x_axis, y_raw_all, label=f'Raw (R2={r2_raw:.3f})', alpha=0.6, linestyle=':')
ax.plot(x_axis, y_persist_all, label=f'Persistence (R2={r2_persist:.3f})', alpha=0.7, linestyle='-')
ax.plot(x_axis, y_meta_all, label=f'Meta-Calibrated (R2={r2_meta:.3f})', alpha=0.8, linestyle='--')
for ws, we in window_bounds:
    ax.axvline(ws, color='gray', linewidth=0.6, linestyle=':')
ax.axhline(0, color='black', linewidth=0.5)
ax.legend()
ax.set_title(f'Rolling Walk-Forward, {n_windows} windows x {window_len} bars: Error-Feedback Strategies vs Raw')
plt.savefig('gaussian_derivative_rolling_5windows_errorfeedback.png')
plt.show()