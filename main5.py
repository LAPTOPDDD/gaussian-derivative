import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 6: модель строит свечи последнего торгового дня
# Обучение НЕ включает последний день (24 бара = 1 день H1)
# Чёрные свечи — настоящие, зелёные — смоделированные моделью
# ============================================================

# Загрузка данных
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')

D = df[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['open', 'high', 'low', 'close']
D = D.reset_index(drop=True)

# Последний день = последние 24 бара (H1)
day_len = 24
n = len(D)
n_test = day_len
n_train = n - n_test

print(f"Всего баров: {n}, Train: {n_train}, Последний день (test): {n_test} баров")

# --------------------------------------------
# Фичи: лаги OHLC (только прошлые бары)
# --------------------------------------------
LAGS = 30
X = pd.DataFrame()
for lag in range(1, LAGS + 1):
    for col in ['open', 'high', 'low', 'close']:
        X[f'{col}_{lag}'] = D[col].shift(lag)

# Целевые значения: OHLC текущего бара
y = D[['open', 'high', 'low', 'close']]

# Объединяем и убираем первые LAGS строк (пустые из-за лагов)
data = pd.concat([X, y], axis=1).dropna()

# Train: всё, кроме последнего дня; Test: последний день
X_train = data.iloc[:-n_test].drop(columns=['open', 'high', 'low', 'close'])
y_train = data.iloc[:-n_test][['open', 'high', 'low', 'close']]

print(f"Фактических строк обучения: {len(X_train)}")

# --------------------------------------------
# Обучение 4 моделей (open, high, low, close)
# --------------------------------------------
models = {}
for col in ['open', 'high', 'low', 'close']:
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(X_train, y_train[col])
    models[col] = m

# --------------------------------------------
# Рекурсивный прогноз последнего дня
# Модель видит только прошлое; прогнозы подставляются как история дальше
# --------------------------------------------
hist = {
    'open': list(D['open'].values[:n_train]),
    'high': list(D['high'].values[:n_train]),
    'low': list(D['low'].values[:n_train]),
    'close': list(D['close'].values[:n_train]),
}

def build_row(hist):
    """Один вектор фич из последних LAGS баров истории."""
    row = {}
    for lag in range(1, LAGS + 1):
        for col in ['open', 'high', 'low', 'close']:
            idx = -lag
            val = hist[col][idx] if len(hist[col]) >= lag else hist[col][0]
            row[f'{col}_{lag}'] = val
    return row

pred = {col: [] for col in ['open', 'high', 'low', 'close']}
for i in range(n_test):
    feat = pd.DataFrame([build_row(hist)])
    for col in ['open', 'high', 'low', 'close']:
        p = models[col].predict(feat)[0]
        pred[col].append(p)
    # подставляем предсказанный бар в историю для следующих шагов
    hist['open'].append(pred['open'][-1])
    hist['high'].append(pred['high'][-1])
    hist['low'].append(pred['low'][-1])
    hist['close'].append(pred['close'][-1])

# Настоящие свечи последнего дня
actual = {
    'open': D['open'].values[n_train:],
    'high': D['high'].values[n_train:],
    'low': D['low'].values[n_train:],
    'close': D['close'].values[n_train:],
}

# --------------------------------------------
# Сравнение (MAE по каждому полю)
# --------------------------------------------
for col in ['open', 'high', 'low', 'close']:
    mae = np.mean(np.abs(np.array(pred[col]) - actual[col]))
    print(f"MAE {col}: {mae:.6f}")

# --------------------------------------------
# Отрисовка свечей (чёрные — настоящие, зелёные — модельные)
# --------------------------------------------
def draw_candles(ax, o, h, l, c, color, alpha=1.0, width=0.6):
    for i in range(len(o)):
        x = i + 0.5
        ax.plot([x, x], [l[i], h[i]], color=color, alpha=alpha, linewidth=1)
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        ax.bar(x, body_hi - body_lo, bottom=body_lo, width=width, color=color, alpha=alpha)

fig, ax = plt.subplots(figsize=(15, 6))
draw_candles(ax, actual['open'], actual['high'], actual['low'], actual['close'], color='black', alpha=1.0)
draw_candles(ax, pred['open'], pred['high'], pred['low'], pred['close'], color='green', alpha=0.65)

ax.set_title('Последний день: чёрные — настоящие свечи, зелёные — смоделированные моделью (v6)')
ax.set_xlabel('Бар внутри дня (0..23)')
ax.set_ylabel('Цена')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('candle_last_day_v6.png')
plt.show()

print("Model trained and plot saved as candle_last_day_v6.png")