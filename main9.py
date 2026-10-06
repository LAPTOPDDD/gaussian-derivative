import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 10: стратегия на предсказанном диапазоне (fade полосы)
#  - одна модель предсказывает диапазон R = H - L дня
#  - полоса: [open - k*R/2, open + k*R/2]  (якорь = дневной open)
#  - вход: откат к нижней границе  -> LONG, TP = open (центр), SL за полосой
#  - вход: откат к верхней границе -> SHORT, TP = open (центр), SL за полосой
#  - торгуем только если R_pred > медианы диапазонов прошлых дней
#  - размер позиции нормализован по предсказанному R (постоянный риск)
#  - walk-forward: каждый день переобучаемся на днях < текущего
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
# Причинные фичи (только прошлое)
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
feat_valid = F.notna().all(axis=1)

n_test_days = 30
test_indices = list(range(len(days)))[-n_test_days:]
test_dates = days['date'].iloc[test_indices].values
daily_rows = {d: g for d, g in D.groupby('date')}

# Параметры стратегии
K = 1.0          # множитель полуширины полосы: [open - K*R/2, open + K*R/2]
SL_EXTRA = 0.25  # стоп на 25% диапазона ЗА границей полосы
TP_AT_OPEN = True  # цель = центр полосы (открытие дня)

print(f"Тестовый период: {test_dates[0]} .. {test_dates[-1]} ({n_test_days} дней)")

trades = []
equity_pips = 0.0
equity_scaled = 0.0
curve_pips = []
curve_scaled = []
rng_model = None

def fit_range_till(before_idx):
    """Обучает модель диапазона на всех днях < before_idx."""
    global rng_model
    mask = (days.index < before_idx) & feat_valid
    Xt = F[mask]
    yt = day_agg['range'].iloc[mask]
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(Xt, yt)
    rng_model = m

for pos, ti in enumerate(test_indices):
    # 1) Переобучение на днях < ti
    fit_range_till(ti)

    # 2) Прогноз диапазона и медиана прошлых диапазонов (порог входа)
    r_pred = rng_model.predict(F.iloc[ti:ti + 1])[0]
    med_r = day_agg['range'].iloc[:ti].median()

    day_open = day_agg['o'].iloc[ti]
    act_r = day_agg['range'].iloc[ti]

    # 3) Полоса от прогнозного диапазона
    if r_pred > med_r:
        do_trade = True
        band_hi = day_open + K * r_pred / 2.0
        band_lo = day_open - K * r_pred / 2.0
        sl_lo = band_lo - SL_EXTRA * r_pred   # для LONG
        sl_hi = band_hi + SL_EXTRA * r_pred   # для SHORT
    else:
        do_trade = False

    rows = daily_rows[test_dates[pos]]
    state = 0        # 0 - flat, +1 - LONG (от нижней границы), -1 - SHORT (от верхней)
    entry = None
    day_pnl_pips = 0.0
    n_trades = 0

    if do_trade:
        for r in rows.itertuples(index=False):
            if state == 0:
                if r.low <= band_lo:
                    state = 1
                    entry = band_lo
                    n_trades += 1
                elif r.high >= band_hi:
                    state = -1
                    entry = band_hi
                    n_trades += 1
            elif state == 1:
                if r.low < sl_lo:  # стоп за нижней полосой
                    loss = (sl_lo - entry) * 10000
                    day_pnl_pips += loss
                    trades.append((test_dates[pos], 'LONG(SL)', entry, sl_lo, loss))
                    state = 0
                elif TP_AT_OPEN and r.high >= day_open:  # цель = центр (open)
                    win = (day_open - entry) * 10000
                    day_pnl_pips += win
                    trades.append((test_dates[pos], 'LONG(TP)', entry, day_open, win))
                    state = 0
            elif state == -1:
                if r.high > sl_hi:  # стоп за верхней полосой
                    loss = (entry - sl_hi) * 10000
                    day_pnl_pips += loss
                    trades.append((test_dates[pos], 'SHORT(SL)', entry, sl_hi, loss))
                    state = 0
                elif TP_AT_OPEN and r.low <= day_open:  # цель = центр (open)
                    win = (entry - day_open) * 10000
                    day_pnl_pips += win
                    trades.append((test_dates[pos], 'SHORT(TP)', entry, day_open, win))
                    state = 0

        if state != 0:  # закрытие по последнему close дня
            last_close = rows['close'].iloc[-1]
            if state == 1:
                pnl = (last_close - entry) * 10000
                trades.append((test_dates[pos], 'LONG(dayclose)', entry, last_close, pnl))
            else:
                pnl = (entry - last_close) * 10000
                trades.append((test_dates[pos], 'SHORT(dayclose)', entry, last_close, pnl))
            day_pnl_pips += pnl
            state = 0

    # 4) Нормализация риска: размер позиции ~ med_r / r_pred
    w = med_r / r_pred if r_pred > 0 else 1.0
    day_pnl_scaled = day_pnl_pips * w if do_trade else 0.0

    equity_pips += day_pnl_pips
    equity_scaled += day_pnl_scaled
    curve_pips.append(equity_pips)
    curve_scaled.append(equity_scaled)

    flag = "" if do_trade else "  (пропуск: R_pred < медианы)"
    print(f"{test_dates[pos]}: R_pred={r_pred:.5f} act_R={act_r:.5f} | "
          f"open={day_open:.5f} | day_pnl={day_pnl_pips:+.1f} pips{flag}")

n_wins = sum(1 for t in trades if t[4] > 0)
print("\n=== ИТОГ ===")
print(f"Сделок: {len(trades)}, прибыльных: {n_wins} ({100*n_wins/len(trades):.0f}%)" if trades else "Сделок: 0")
print(f"Итог (сырые пипс): {equity_pips:+.1f}")
print(f"Итог (с нормировкой риска): {equity_scaled:+.1f}")

# --------------------------------------------
# График
# --------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1]})

axs[0].plot(range(len(test_dates)), curve_pips, marker='o', color='black', label='PnL (сырые пипс)')
axs[0].plot(range(len(test_dates)), curve_scaled, marker='o', color='blue', label='PnL (риск-нормированные пипс)')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].legend()
axs[0].set_title('Бэктест v10: fade предсказанной полосы диапазона (30 тестовых дней)')
axs[0].set_ylabel('PnL, пипс')
axs[0].grid(alpha=0.3)

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
plt.savefig('backtest_30d_v10.png')
plt.show()

print("Plot saved as backtest_30d_v10.png")