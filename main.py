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
# Train: all data EXCEPT test_len + purge_gap
# Test: last test_len bars
total_len = len(df)
train_end = total_len - test_len - purge_gap

X_train = X.iloc[num_lags:train_end]
y_train = pd.Series(conv_derivative).iloc[num_lags:train_end]

X_test = X.iloc[-test_len:]
y_test = pd.Series(conv_derivative).iloc[-test_len:]

# 6. Train
model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, verbose=-1, n_jobs=-1)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
r2_model = r2_score(y_test, y_pred)
causal_ceiling = 0.506

print(f"Model R2 on Test: {r2_model:.4f}")
print(f"Causal Ceiling: {causal_ceiling:.4f}")
print(f"Efficiency: {r2_model / causal_ceiling:.2%}")

# 8. Plot
plt.figure(figsize=(15, 6))
plt.plot(y_test.values, label='Actual (Non-Causal)', alpha=0.7)
plt.plot(y_pred, label=f'Model Prediction (R2={r2_model:.3f})', alpha=0.7, linestyle='--')
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.title(f'Strict Causal Gaussian Derivative Approximation (Test R2={r2_model:.3f} vs Ceiling={causal_ceiling})')
plt.savefig('gaussian_derivative_approximation_v3.png')
plt.show()
