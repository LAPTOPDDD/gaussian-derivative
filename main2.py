import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

# Load data
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price_series = df['<CLOSE>']
price = price_series.values

# Parameters
sigma = 5
window_size = 3 * sigma
t = np.arange(-window_size, window_size + 1)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
gaussian_derivative = -(t / sigma**2) * gaussian

# Calculate target
conv_derivative = convolve1d(price, weights=gaussian_derivative, mode='nearest')

# Improved Features: Causal only
X = pd.DataFrame()
# Lags
num_lags = 60
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = price_series.shift(i)

# Differences
X['diff_1'] = price_series.diff(1).shift(1)
X['diff_5'] = price_series.diff(5).shift(1)

# Moving Averages / Volatility
X['sma_10'] = price_series.rolling(10).mean().shift(1)
X['std_10'] = price_series.rolling(10).std().shift(1)

# Alignment
y = pd.Series(conv_derivative)
data = pd.concat([X, y.rename('target')], axis=1).dropna()
X_train = data.drop(columns=['target'])
y_train = data['target']

# Train LightGBM (Increased n_estimators)
model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, verbose=-1, n_jobs=-1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_train)

# Visualization
test_len = 200
plt.figure(figsize=(15, 6))
plt.plot(y_train.values[-test_len:], label='Actual (Non-Causal Gaussian Derivative)', alpha=0.7)
plt.plot(y_pred[-test_len:], label='Model Prediction', alpha=0.7, linestyle='--')
plt.legend()
plt.title('Improved Gaussian Derivative Approximation (Test Segment)')
plt.savefig('gaussian_derivative_approximation_v2.png')
plt.show()

print("Model trained and plot saved as gaussian_derivative_approximation_v2.png")
