import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 7: прогноз максимума и минимума последнего дня
# Плюс построение всех свечей последнего дня (рекурсивно)
# Смотрим, как свечи ложатся на предсказанные min/max
# ============================================================

# Загрузка данных
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')

D = df[['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['date', 'time', 'open', 'high', 'low', 'close']
D['dt'] = pd.to_datetime(D['date'] + ' ' + D['time'], format='%Y.%m.%d %H:%M:%S')
D = D.reset_index(drop=True)

day_len = 24
n = len(D)
n_test = day_len
n_train = n - n_test

# --------------------------------------------
# Часть 1. Дневная модель: максимум и минимум дня
# --------------------------------------------
D['dur_date'] = D['date']
day_agg = D.groupby('dur_date').agg(
    o=('open', 'first'),
    h=('high', 'max'),
    l=('low', 'min'),
    c=('close', 'last'),
    nbars=('close', 'count'),
).reset_index()
day_agg['range'] = day_agg['h'] - day_agg['l']
day_agg['body'] = day_agg['c'] - day_agg['o']
day_agg['gap'] = day_agg['o'] - day_agg['c'].shift(1)

# Полные дни (для обучения) и последний день отдельно
last_day = day_agg.iloc[-1]
full_days = day_agg[day_agg['nbars'] >= 20]

# Фичи дневной модели: показатели предыдущих дней (только прошлое)
P = 10
Xday = pd.DataFrame()
for lag in range(1, P + 1):
    for col in ['o', 'h', 'l', 'c', 'range', 'body', 'gap']:
        Xday[f'prev{lag}_{col}'] = day_agg[col].shift(lag)

# Роллинг-окна
Xday['roll_mean_h5'] = day_agg['h'].rolling(5).mean().shift(1)
Xday['roll_max_h5'] = day_agg['h'].rolling(5).max().shift(1)
Xday['roll_mean_l5'] = day_agg['l'].rolling(5).mean().shift(1)
Xday['roll_min_l5'] = day_agg['l'].rolling(5).min().shift(1)
Xday['roll_mean_r5'] = day_agg['range'].rolling(5).mean().shift(1)

yd = day_agg[['h', 'l']]
data_day = pd.concat([Xday, yd], axis=1).dropna()

# Train: все дни, кроме последнего; Test: последний день
Xday_train = data_day.iloc[:-1].drop(columns=['h', 'l'])
yday_train = data_day.iloc[:-1][['h', 'l']]

day_models = {}
for col in ['h', 'l']:
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(Xday_train, yday_train[col])
    day_models[col] = m

# Последний ряд фич для последнего дня
feat_last = data_day.iloc[-1:].drop(columns=['h', 'l'])
pred_h = day_models['h'].predict(feat_last)[0]
pred_l = day_models['l'].predict(feat_last)[0]

act_h = last_day['h']
act_l = last_day['l']
print(f"Последний день {last_day['dur_date']}:")
print(f"  Прогноз  HIGH: {pred_h:.5f}  vs факт: {act_h:.5f}")
print(f"  Прогноз  LOW : {pred_l:.5f}  vs факт: {act_l:.5f}")

# --------------------------------------------
# Часть 2. Построение всех свечей последнего дня (рекурсивно)
# --------------------------------------------
LAGS = 30
Xh = pd.DataFrame()
for lag in range(1, LAGS + 1):
    for col in ['open', 'high', 'low', 'close']:
        Xh[f'{col}_{lag}'] = D[col].shift(lag)

yh = D[['open', 'high', 'low', 'close']]
data_h = pd.concat([Xh, yh], axis=1).dropna()

Xh_train = data_h.iloc[:-n_test].drop(columns=['open', 'high', 'low', 'close'])
yh_train = data_h.iloc[:-n_test][['open', 'high', 'low', 'close']]

amb_models = {}
for col in ['open', 'high', 'low', 'close']:
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(Xh_train, yh_train[col])
    amb_models[col] = m

hist = {
    'open': list(D['open'].values[:n_train]),
    'high': list(D['high'].values[:n_train]),
    'low': list(D['low'].values[:n_train]),
    'close': list(D['close'].values[:n_train]),
}

def build_row(hist):
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
        p = amb_models[col].predict(feat)[0]
        pred[col].append(p)
    hist['open'].append(pred['open'][-1])
    hist['high'].append(pred['high'][-1])
    hist['low'].append(pred['low'][-1])
    hist['close'].append(pred['close'][-1])

actual = {
    'open': D['open'].values[n_train:],
    'high': D['high'].values[n_train:],
    'low': D['low'].values[n_train:],
    'close': D['close'].values[n_train:],
}

# Сколько предсказанных свечей укладывается в предсказанный диапазон [pred_l, pred_h]
sim_close = np.array(pred['close'])
inside = ((sim_close >= pred_l) & (sim_close <= pred_h)).sum()
last_close_val = pred['close'][-1]
print(f"Свечей внутрь прогнозного диапазона: {inside} из {n_test}")
print(f"Закрытие дня (модель): {last_close_val:.5f}")

# --------------------------------------------
# Отрисовка
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

# Прогнозные min/max дня
ax.axhline(pred_h, color='red', linestyle='--', linewidth=1.5, label=f'Прогноз HIGH = {pred_h:.5f}')
ax.axhline(pred_l, color='blue', linestyle='--', linewidth=1.5, label=f'Прогноз LOW = {pred_l:.5f}')
ax.axhline(act_h, color='red', linestyle=':', alpha=0.6, label=f'Факт HIGH = {act_h:.5f}')
ax.axhline(act_l, color='blue', linestyle=':', alpha=0.6, label=f'Факт LOW = {act_l:.5f}')

ax.set_title(f'Версия 7: свечи последнего дня ({last_day["dur_date"]}) и прогнозные min/max')
ax.set_xlabel('Бар внутри дня (0..23)')
ax.set_ylabel('Цена')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('candle_last_day_v7.png')
plt.show()

print("Model trained and plot saved as candle_last_day_v7.png")