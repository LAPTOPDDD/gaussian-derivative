import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Попытка импорта LightGBM. Если его нет, используем быстрый аналог из sklearn
try:
    import lightgbm as lgb
    ModelClass = lgb.LGBMRegressor
    # verbose=-1 чтобы не спамил в консоль процессом обучения
    model_params = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42, 'verbose': -1}
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    ModelClass = HistGradientBoostingRegressor
    model_params = {'max_iter': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42}

# 1. Загрузка данных
filename = 'EURUSD_H1_202412270000_202609092300.csv'
df = pd.read_csv(filename, sep='\t')

# ИСПРАВЛЕНИЕ: Убираем угловые скобки из названий столбцов (<CLOSE> -> CLOSE)
df.columns = [col.strip('<>') for col in df.columns]

print(f"Загружено {len(df)} строк из {filename}")
print(f"Названия столбцов: {df.columns.tolist()}")

prices = df['CLOSE'].values

# 2. Параметры ядра Гаусса и его первой производной
sigma = 10.0           # Ширина гауссиана (в барах)
radius = int(3 * sigma) # Радиус окна (30 баров)
x = np.arange(-radius, radius + 1)

# Формула первой производной гауссиана: -x / sigma^2 * exp(-x^2 / (2 * sigma^2))
kernel = (-x / (sigma**2)) * np.exp(-x**2 / (2 * sigma**2))

# 3. Формирование таргета (Учитель)
# mode='same' означает, что ядро центрировано на текущем баре t.
# Это НЕКАЗУАЛЬНЫЙ сигнал: для вычисления значения в момент t используются цены от t-radius до t+radius.
target = np.convolve(prices, kernel, mode='same')

# 4. Формирование казуальных фичей
# Мы используем только цены от текущего момента t и вглубь прошлого до t - 2*radius
max_lag = 2 * radius  # 60 баров
feature_dict = {}
for i in range(max_lag + 1):
    # np.roll сдвигает массив вправо, поэтому на позиции t оказывается цена t-i (строго казуально!)
    feature_dict[f'price_lag_{i}'] = np.roll(prices, i)

feature_df = pd.DataFrame(feature_dict)

# ВАЖНО: Удаляем первые max_lag строк, чтобы избавиться от "заглядывания в будущее" 
# (в начале массива np.roll подставляет значения с конца ряда, что является утечкой данных)
feature_df = feature_df.iloc[max_lag:].reset_index(drop=True)
target_series = pd.Series(target).iloc[max_lag:].reset_index(drop=True)

# 5. Разделение на обучающую и тестовую выборки
test_size = 200
X_train, X_test = feature_df.values[:-test_size], feature_df.values[-test_size:]
y_train, y_test = target_series.values[:-test_size], target_series.values[-test_size:]

# 6. Обучение модели
print("Обучение модели...")
model = ModelClass(**model_params)
model.fit(X_train, y_train)
print("Обучение завершено!")

# 7. Предсказание на тестовом участке
y_pred = model.predict(X_test)

# 8. Оценка качества
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
corr = np.corrcoef(y_test, y_pred)[0, 1]

print(f"\nМетрики на тестовом участке ({test_size} баров):")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2:  {r2:.4f}")
print(f"Корреляция Пирсона: {corr:.4f}")

# 9. Визуализация
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Истинная производная (Неказуальный учитель)', color='blue', alpha=0.8, linewidth=2)
plt.plot(y_pred, label='Предсказание модели (Казуальные фичи)', color='red', linestyle='--', alpha=0.8, linewidth=2)
plt.title(f'Аппроксимация первой производной свертки Гаусса\n(Последние {test_size} баров)\nR2: {r2:.4f}, Корреляция: {corr:.4f}', fontsize=14)
plt.xlabel('Индекс бара (Тестовая выборка)', fontsize=12)
plt.ylabel('Значение производной', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Сохраняем график в папку со скриптом
plt.savefig('gaussian_derivative_approximation.png', dpi=150)
print("\nГрафик сохранен как 'gaussian_derivative_approximation.png'")
plt.show()