import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 12 (пункт 4): range recycler
# Вход по открытию дня, TP = open + 0.5*R_pred, SL = open - 0.5*R_pred
# Направление не предсказываем (чистый тест: достаточно ли точного R_pred)
# Не торгуем, если R_pred слишком мал (ширина TP/SL меньше затрат).
# ============================================================

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
D = df[['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['date', 'time', 'open', 'high', 'low', 'close']
D = D.reset_index(drop=True)

day_agg = D.groupby('date').agg(
    o=('open', 'first'), h=('high', 'max'), l=('low', 'min'),
    c=('close', 'last'), nbars=('close', 'count'),
).reset_index()
day_agg['range'] = day_agg['h'] - day_agg['l']
day_agg['body'] = day_agg['c'] - day_agg['o']
day_agg['gap'] = day_agg['o'] - day_agg['c'].shift(1)

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

TP_FRAC = 0.5   # цель = 0.5 * R_pred
SL_FRAC = 0.5   # стоп = 0.5 * R_pred
MIN_R = 0.002   # минимум ~20 pips диапазона, ниже не торгуем

print(f"Тестовый период: {test_dates[0]} .. {test_dates[-1]}")

trades = []
equity_pips = 0.0
equity_scaled = 0.0
curve_pips = []
curve_scaled = []
rng_model = None

def fit_range_till(before_idx):
    global rng_model
    mask = (days.index < before_idx) & feat_valid
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(F[mask], day_agg['range'].iloc[mask].values)
    rng_model = m

for pos, ti in enumerate(test_indices):
    fit_range_till(ti)
    r_pred = rng_model.predict(F.iloc[ti:ti + 1])[0]
    med_r = day_agg['range'].iloc[:ti].median()
    day_open = day_agg['o'].iloc[ti]
    act_r = day_agg['range'].iloc[ti]

    do_trade = r_pred > MIN_R
    rows = daily_rows[test_dates[pos]]
    state = 0
    entry = None
    day_pnl_pips = 0.0
    n_trades = 0

    if do_trade:
        tp = day_open + TP_FRAC * r_pred
        sl = day_open - SL_FRAC * r_pred
        state = 1
        entry = day_open
        n_trades += 1
        for r in rows.itertuples(index=False):
            if state == 1:
                if r.low <= sl:
                    pnl = (sl - entry) * 10000
                    day_pnl_pips += pnl
                    trades.append((test_dates[pos], 'LONG(SL)', entry, sl, pnl))
                    state = 0
                elif r.high >= tp:
                    pnl = (tp - entry) * 10000
                    day_pnl_pips += pnl
                    trades.append((test_dates[pos], 'LONG(TP)', entry, tp, pnl))
                    state = 0
        if state != 0:
            last_close = rows['close'].iloc[-1]
            pnl = (last_close - entry) * 10000
            trades.append((test_dates[pos], 'LONG(dayclose)', entry, last_close, pnl))
            day_pnl_pips += pnl
            state = 0

    w = med_r / r_pred if r_pred > 0 else 1.0
    day_pnl_scaled = day_pnl_pips * w if do_trade else 0.0
    equity_pips += day_pnl_pips
    equity_scaled += day_pnl_scaled
    curve_pips.append(equity_pips)
    curve_scaled.append(equity_scaled)

    why = "" if do_trade else "  (R_pred < MIN_R)"
    print(f"{test_dates[pos]}: R_pred={r_pred:.5f} act_R={act_r:.5f} "
          f"open={day_open:.5f} day_pnl={day_pnl_pips:+.1f}{why}")

n_wins = sum(1 for t in trades if t[4] > 0)
daily = [sum(t[4] for t in trades if t[0] == d) for d in test_dates]
dd = 0.0; peak = -1e9
for v in curve_pips:
    peak = max(peak, v); dd = min(dd, v - peak)
sharpe = (np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252)

print("\n=== ИТОГ (v12, range recycler) ===")
print(f"Сделок: {len(trades)}, прибыльных: {n_wins} ({100*n_wins/(len(trades) or 1):.0f}%)")
print(f"Итог: {equity_pips:+.1f} pips (норм: {equity_scaled:+.1f})")
print(f"Max DD: {dd:.1f} pips, Sharpe(daily): {sharpe:.2f}")

fig, axs = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1]})
axs[0].plot(range(len(test_dates)), curve_pips, marker='o', color='black', label='PnL pips')
axs[0].plot(range(len(test_dates)), curve_scaled, marker='o', color='blue', label='PnL norm')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].legend()
axs[0].set_title('backtest v12: range recycler (вход по open, TP/SL = 0.5*R_pred)')
axs[0].set_ylabel('PnL, pips'); axs[0].grid(alpha=0.3)
colors = ['green' if v >= 0 else 'red' for v in daily]
axs[1].bar(range(len(test_dates)), daily, color=colors)
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].set_ylabel('PnL за день'); axs[1].set_xlabel('День'); axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_30d_v12.png')
plt.show()
print("Plot saved as backtest_30d_v12.png")