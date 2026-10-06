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
window_size = 3 * sigma
t = np.arange(-window_size, window_size + 1)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
gaussian_derivative = -(t / sigma**2) * gaussian

# 3. Target (Non-causal)
conv_derivative = convolve1d(price, weights=gaussian_derivative, mode='nearest')

# 4. Features (Causal Lags of Log Returns)
num_lags = 60
X = pd.DataFrame()
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = pd.Series(returns).shift(i)
X = X.fillna(0)

# 5. Split and Purge
total_len = len(df)
train_end = total_len - test_len - purge_gap

X_train = X.iloc[num_lags:train_end]
y_train = pd.Series(conv_derivative).iloc[num_lags:train_end]

X_test = X.iloc[-test_len:]
y_test = pd.Series(conv_derivative).iloc[-test_len:]
test_returns = returns[-test_len:]

# 6. Train
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Analysis
# Correlation between model output and actual returns
correlation = np.corrcoef(y_pred, test_returns)[0, 1]

# Strategy: Signal based on threshold
threshold = np.std(y_pred)
signals = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
# Assume we trade on the current candle's return
trade_returns = signals * test_returns
cumulative_profit = np.cumsum(trade_returns)

print(f"Correlation (Model Output vs Actual Returns): {correlation:.4f}")
print(f"Strategy: Trade if |pred| > std({threshold:.6f})")
print(f"Cumulative Profit (Log Returns): {cumulative_profit[-1]:.6f}")

# 9. Visualization
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(y_test.values, label='Actual Derivative', alpha=0.5)
plt.plot(y_pred, label='Model Prediction', alpha=0.5, linestyle='--')
plt.title('Derivative Approximation')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(cumulative_profit, label='Cumulative Strategy Profit')
plt.title(f'Strategy Performance (Correlation={correlation:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('analysis_signals.png')
plt.show()
