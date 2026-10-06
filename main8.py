import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 9: ежедневное переобучение (walk-forward)
# День отторговали -> переобучаем на всех днях, включая его -> торгуем следующий
# Модель дня N обучается ТОЛЬКО на днях < N (без утечки)
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

# --------------------------------------------
# Причинные фичи (сдвиги/окна смотрят только в прошлое)
# --------------------------------------------
P = 10
F = pd.DataFrame(index=day_agg.index)
for lag in range(1, P + 1):
    for col in ['o', 'h', 'l', 'c', 'range', 'body', 'gap']:
        F[f'prev{lag}_{col}'] = day_agg[col].shift(lag)
F['roll_mean_h5'] = day_agg['h'].rolling(5).mean().shift(1)
F['roll_max_h5'] = day_agg['h'].rolling(5).max().shift(1)
F['roll_mean_l5'] = day_agg['l'].rolling(5).mean().shift(1)
F['roll_min_l5'] = day_agg['l'].rolling(5).min().shift(1)
F['roll_mean_r5'] = day_agg['range'].rolling(5).mean().shift(1)

days = day_agg.reset_index(drop=True)
F = F.reset_index(drop=True)
yy = days[['h', 'l']]
feat_valid = F.notna().all(axis=1)

n_test_days = 30
test_indices = list(range(len(days)))[-n_test_days:]
test_dates = days['date'].iloc[test_indices].values
daily_rows = {d: g for d, g in D.groupby('date')}

print(f"Тестовый период: {test_dates[0]} .. {test_dates[-1]} ({n_test_days} дней)")

# --------------------------------------------
# Walk-forward: переобучение каждый день
# --------------------------------------------
cand_models = {'h': None, 'l': None}

def fit_till(before_idx):
    """Обучает модели на всех днях с индексом < before_idx (фичи при этом причинные)."""
    mask = (days.index < before_idx) & feat_valid
    Xt = F[mask]
    Yt = yy[mask]
    for col in ['h', 'l']:
        m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1)
        m.fit(Xt, Yt[col])
        cand_models[col] = m

trades = []
equity = 0.0
equity_curve = []

for pos, ti in enumerate(test_indices):
    # 1) Переобучение на всех днях до текущего (типа «после вчерашней торговли»)
    fit_till(ti)

    # 2) Прогноз min/max сегодняшнего дня
    ph = cand_models['h'].predict(F.iloc[ti:ti + 1])[0]
    pl = cand_models['l'].predict(F.iloc[ti:ti + 1])[0]

    # 3) Торгуем день по свечам
    rows = daily_rows[test_dates[pos]]
    state = 0        # 0 - flat, +1 - LONG, -1 - SHORT
    entry = None
    day_pnl_pips = 0.0
    n_trades = 0

    for r in rows.itertuples(index=False):
        if state == 0:
            if r.low <= pl:
                state = 1
                entry = pl
                n_trades += 1
            elif r.high >= ph:
                state = -1
                entry = ph
                n_trades += 1
        elif state == 1:
            if r.high >= ph:
                day_pnl_pips += (ph - entry) * 10000
                trades.append((test_dates[pos], 'LONG', entry, ph, (ph - entry) * 10000))
                state = 0
        elif state == -1:
            if r.low <= pl:
                day_pnl_pips += (entry - pl) * 10000
                trades.append((test_dates[pos], 'SHORT', entry, pl, (entry - pl) * 10000))
                state = 0

    if state != 0:
        last_close = rows['close'].iloc[-1]
        if state == 1:
            pnl = (last_close - entry) * 10000
            trades.append((test_dates[pos], 'LONG(dayclose)', entry, last_close, pnl))
        else:
            pnl = (entry - last_close) * 10000
            trades.append((test_dates[pos], 'SHORT(dayclose)', entry, last_close, pnl))
        day_pnl_pips += pnl
        state = 0

    equity += day_pnl_pips
    equity_curve.append(equity)
    print(f"{test_dates[pos]}: pred_L={pl:.5f} act_L={yy['l'].iloc[ti]:.5f} | "
          f"pred_H={ph:.5f} act_H={yy['h'].iloc[ti]:.5f} | day_nl={day_pnl_pips:+.1f} | total={equity:+.1f}")

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
axs[0].set_title('Бэктест v9: ежедневное переобучение (30 тестовых дней)')
axs[0].set_ylabel('PnL, пипс')
axs[0].grid(alpha=0.3)
axs[0].legend()

daily_pnl = [0.0] * len(test_dates)
for idx, day in enumerate(test_dates):
    daily_pnl[idx] = sum(t[4] for t in trades if t[0] == day)

colors = ['green' if v >= 0 else 'red' for v in daily_pnl]
axs[1].bar(range(len(test_dates)), daily_pnl, color=colors)
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].set_ylabel('PnL за день, пипс')
axs[1].set_xlabel('Тестовый день')
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_30d_v9.png')
plt.show()

print("Plot saved as backtest_30d_v9.png")