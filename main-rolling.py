import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from sklearn.metrics import r2_score

# 1. Load and Prep Data
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price = df['<CLOSE>'].values
returns = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1)).fillna(0).values

# 2. Parameters
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

# 3. Target (Non-causal)
conv_derivative = convolve1d(price, weights=gaussian_derivative, mode='nearest')
y_all = pd.Series(conv_derivative)

# 4. Features (Causal Lags of Log Returns)
X = pd.DataFrame()
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = pd.Series(returns).shift(i)
X = X.fillna(0)

# If test_len is not a full multiple of n_windows, give the remainder to the last window
total_len = len(df)
test_start = total_len - test_len
window_bounds = []
for w in range(n_windows):
    ws = test_start + w * window_len
    we = test_start + (w + 1) * window_len
    if w == n_windows - 1:
        we = total_len
    window_bounds.append((ws, we))

# 5. Walk-Forward: retrain before every window, predict the window, slide on
y_true_all = []
y_pred_all = []
window_labels = []
causal_ceiling = 0.506

print(f"total bars: {total_len}, test region: {test_start}..{total_len}, {n_windows} windows")
print("-" * 70)

for w, (ws, we) in enumerate(window_bounds, start=1):
    train_end = ws - purge_gap

    X_train = X.iloc[num_lags:train_end]
    y_train = y_all.iloc[num_lags:train_end]

    X_test = X.iloc[ws:we]
    y_test = y_all.iloc[ws:we]

    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2_w = r2_score(y_test, y_pred)

    y_true_all.append(y_test.values)
    y_pred_all.append(y_pred)
    window_labels.append(w)

    print(f"window {w:2d}: bars {ws:5d}..{we:5d} (len={we-ws:3d}) | train_end={train_end} | R2={r2_w:.4f}")

# 6. Overall evaluation
y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
r2_total = r2_score(y_true_all, y_pred_all)

print("-" * 70)
print(f"Overall Walk-Forward R2 on Test: {r2_total:.4f}")
print(f"Causal Ceiling: {causal_ceiling:.4f}")
print(f"Efficiency: {r2_total / causal_ceiling:.2%}")

# 7. Plot
fig, ax = plt.subplots(figsize=(15, 6))
x_axis = np.arange(test_start, total_len)
ax.plot(x_axis, y_true_all, label='Actual (Non-Causal)', alpha=0.7)
ax.plot(x_axis, y_pred_all, label=f'Walk-Forward Prediction (R2={r2_total:.3f})', alpha=0.7, linestyle='--')

ymax = ax.get_ylim()[1]
for w, (ws, we) in enumerate(window_bounds, start=1):
    ax.axvline(ws, color='gray', linewidth=0.6, linestyle=':')
    if we - ws > 0:
        ax.text((ws + we) / 2, ymax, str(w), ha='center',
                va='top', fontsize=8, color='gray')

ax.axhline(0, color='black', linewidth=0.5)
ax.legend()
ax.set_title(f'Rolling (Walk-Forward) Gaussian Derivative Approximation - 15 Windows\n'
             f'Test R2={r2_total:.3f} vs Ceiling={causal_ceiling}')
plt.savefig('gaussian_derivative_rolling_15windows.png')
plt.show()