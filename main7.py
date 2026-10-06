import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 8: бэктест на 30 тестовых днях
# Модель предсказывает максимум/минимум дня
#  - касание предсказанного минимума -> покупка, выход у предсказанного максимума
#  - касание предсказанного максимума -> продажа, выход у предсказанного минимума
# ============================================================

# Загрузка данных
file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')

D = df[['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['date', 'time', 'open', 'high', 'low', 'close']
D = D.reset_index(drop=True)

# --------------------------------------------
# Дневная агрегация
# --------------------------------------------
day_agg = D.groupby('date').agg(
    o=('open', 'first'),
    h=('high', 'max'),
    l=('low', 'min'),
    c=('close', 'last'),
    nbars=('close', 'count'),
).reset_index()
day_agg['range'] = day_agg['h'] - day_agg['l']
day_agg['body'] = day_agg['c'] - day_agg['o']
day_agg['gap'] = day_agg['o'] - day_agg['c'].shift(1)

n_test_days = 30

# Строим фичи для всех дней заранее (индекс соответствует day_agg)
P = 10
F = pd.DataFrame()
for lag in range(1, P + 1):
    for col in ['o', 'h', 'l', 'c', 'range', 'body', 'gap']:
        F[f'prev{lag}_{col}'] = day_agg[col].shift(lag)

F['roll_mean_h5'] = day_agg['h'].rolling(5).mean().shift(1)
F['roll_max_h5'] = day_agg['h'].rolling(5).max().shift(1)
F['roll_mean_l5'] = day_agg['l'].rolling(5).mean().shift(1)
F['roll_min_l5'] = day_agg['l'].rolling(5).min().shift(1)
F['roll_mean_r5'] = day_agg['range'].rolling(5).mean().shift(1)

Y = day_agg[['h', 'l']]

valid = F.ffill().copy()  # заполняем первые дни, чтобы оставить максимум строк
combined = pd.concat([valid, Y], axis=1).dropna()

# Первые n_test_days дней выкидываем из обучения ещё и для чистоты (мало лагов)
# Тестовый период = последние 30 дней, обучение = всё до него
split = len(combined) - n_test_days
Xt = combined.iloc[:split].drop(columns=['h', 'l'])
Yt = combined.iloc[:split][['h', 'l']]
Xt_test = combined.iloc[split:].drop(columns=['h', 'l'])
test_days = combined.index[split:]
test_dates = day_agg.iloc[test_days]['date'].values
Y_actual = combined.iloc[split:][['h', 'l']]

print(f"Тестовый период: {test_dates[0]} .. {test_dates[-1]} ({n_test_days} дней)")
print(f"Обучение: {len(Xt)} дней")

# --------------------------------------------
# Обучаем 2 модели: максимум и минимум дня
# --------------------------------------------
models = {}
for col in ['h', 'l']:
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(Xt, Yt[col])
    models[col] = m

pred_h = models['h'].predict(Xt_test)
pred_l = models['l'].predict(Xt_test)

# --------------------------------------------
# Бэктест по внутридневным свечам
# --------------------------------------------
# Правила (по статусу внутри дня):
#  flat  : касание pred_l -> LONG  по pred_l; иначе касание pred_h -> SHORT по pred_h
#  LONG  : касание pred_h -> выход по pred_h
#  SHORT : касание pred_l -> выход по pred_l
#  Конец дня: открытая позиция закрывается по последнему close дня

def touches_timed(day_rows, pred_hi, pred_lo):
    """Возвращает порядок касаний уровней в течение дня (по свечам)."""
    events = []
    for r in day_rows:
        if r['low'] <= pred_lo:
            events.append(('low', r))
        if r['high'] >= pred_hi:
            events.append(('high', r))
    return events

trades = []
equity = 0.0
equity_curve = []
daily_rows = {d: g for d, g in D.groupby('date')}

for idx, day in zip(range(len(test_dates)), test_dates):
    rows = daily_rows[day]
    state = 0        # 0 - flat, +1 - LONG, -1 - SHORT
    entry = None
    day_pnl_pips = 0.0
    n_trades = 0

    for r in rows.itertuples(index=False):
        if state == 0:
            if r.low <= pred_l[idx]:
                state = 1
                entry = pred_l[idx]
                n_trades += 1
            elif r.high >= pred_h[idx]:
                state = -1
                entry = pred_h[idx]
                n_trades += 1
        elif state == 1:
            if r.high >= pred_h[idx]:
                day_pnl_pips += (pred_h[idx] - entry) * 10000
                trades.append((day, 'LONG', entry, pred_h[idx], (pred_h[idx] - entry) * 10000))
                state = 0
        elif state == -1:
            if r.low <= pred_l[idx]:
                day_pnl_pips += (entry - pred_l[idx]) * 10000
                trades.append((day, 'SHORT', entry, pred_l[idx], (entry - pred_l[idx]) * 10000))
                state = 0

    if state != 0:
        last_close = rows['close'].iloc[-1]
        if state == 1:
            pnl = (last_close - entry) * 10000
            day_pnl_pips += pnl
            trades.append((day, 'LONG(dayclose)', entry, last_close, pnl))
        else:
            pnl = (entry - last_close) * 10000
            day_pnl_pips += pnl
            trades.append((day, 'SHORT(dayclose)', entry, last_close, pnl))
        state = 0

    equity += day_pnl_pips
    equity_curve.append(equity)
    print(f"{day}: pred_L={pred_l[idx]:.5f} act_L={Y_actual.iloc[idx]['l']:.5f} | "
          f"pred_H={pred_h[idx]:.5f} act_H={Y_actual.iloc[idx]['h']:.5f} | day_nl={day_pnl_pips:+.1f} pips | total={equity:+.1f}")

total_pnl = equity
n_wins = sum(1 for t in trades if t[4] > 0)
print("\n=== ИТОГ ===")
print(f"Сделок: {len(trades)}, прибыльных: {n_wins} ({100*n_wins/len(trades):.0f}%)" if trades else "Сделок: 0")
print(f"Итог за {n_test_days} дней: {total_pnl:+.1f} пипс")

# --------------------------------------------
# График
# --------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1]})

axs[0].plot(range(len(test_dates)), equity_curve, marker='o', color='black', label='Кумулятивн. PnL (пипс)')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].set_title('Бэктест v8: стратегия касания предсказанных min/max (30 тестовых дней)')
axs[0].set_ylabel('PnL, пипс')
axs[0].grid(alpha=0.3)
axs[0].legend()

daily_pnl = [0.0] * len(test_dates)
for idx, day in enumerate(test_dates):
    day_pnl = 0.0
    for t in trades:
        if t[0] == day:
            day_pnl += t[4]
    daily_pnl[idx] = day_pnl

colors = ['green' if v >= 0 else 'red' for v in daily_pnl]
axs[1].bar(range(len(test_dates)), daily_pnl, color=colors)
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].set_ylabel('PnL за день, пипс')
axs[1].set_xlabel('Тестовый день')
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_30d_v8.png')
plt.show()

print("Plot saved as backtest_30d_v8.png")