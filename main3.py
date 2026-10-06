import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

# ============================================================
# Версия 4: разбиение на train/test, целевая функция не изменена
# ============================================================

# Загрузка данных
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price_series = df['<CLOSE>']
price = price_series.values

# Параметры
sigma = 5
window_size = 3 * sigma
t = np.arange(-window_size, window_size + 1)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma**2))
gaussian_derivative = -(t / sigma**2) * gaussian

# Целевая функция: производная гаусса (НЕ ИЗМЕНЕНА, остаётся non-causal)
conv_derivative = convolve1d(price, weights=gaussian_derivative, mode='nearest')

# Фичи: только причинные (лаги и прошлые значения)
X = pd.DataFrame()
# Лаги цены закрытия
num_lags = 60
for i in range(1, num_lags + 1):
    X[f'lag_{i}'] = price_series.shift(i)

# Разности
X['diff_1'] = price_series.diff(1).shift(1)
X['diff_5'] = price_series.diff(5).shift(1)

# Скользящие средние / волатильность
X['sma_10'] = price_series.rolling(10).mean().shift(1)
X['std_10'] = price_series.rolling(10).std().shift(1)

# Сборка датасета
y = pd.Series(conv_derivative)
data = pd.concat([X, y.rename('target')], axis=1).dropna()
X_all = data.drop(columns=['target'])
y_all = data['target']

# --------------------------------------------
# Разбиение на train / test (по времени)
# --------------------------------------------
split_idx = int(len(data) * 0.8)
X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

print(f"Train: {len(X_train)} строк, Test: {len(X_test)} строк")

# Обучение LightGBM только на train
model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, verbose=-1, n_jobs=-1)
model.fit(X_train, y_train)

# Предсказание на test (данные, которые модель не видела)
y_pred_test = model.predict(X_test)

# Визуализация test-сегмента
test_len = 200
plt.figure(figsize=(15, 6))
plt.plot(y_test.values[-test_len:], label='Actual (Non-Causal Gaussian Derivative)', alpha=0.7)
plt.plot(y_pred_test[-test_len:], label='Model Prediction (test)', alpha=0.7, linestyle='--')
plt.legend()
plt.title('Gaussian Derivative Approximation v4 (Test Segment, out-of-sample)')
plt.savefig('gaussian_derivative_approximation_v4.png')
plt.show()

print("Model trained and plot saved as gaussian_derivative_approximation_v4.png")