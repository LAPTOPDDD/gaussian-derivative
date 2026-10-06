import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# Версия 13: 5 направлений для v12 (range recycler)
# Все варианты используют тот же ход вперёд (переобучение ежедневно) и те же 30 тестовых дней.
#   V0 base   : long, вход в open, TP=+0.5R, SL=-0.5R, каждый день            (эталон)
#   V1 dir    : направление по вчерашнему телу (body>=0 -> long, иначе short)
#   V2 time   : отложенный вход (2-й бар), вход по open того бара
#   V3 async  : асимметрия TP=0.6R / SL=0.5R (ближе, чаще, меньше)
#   V4 vol    : торгуем только дни с R_pred > медианы прошлых диапазонов
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
bodies = day_agg['body']

n_test_days = 30
test_indices = list(range(len(days)))[-n_test_days:]
test_dates = days['date'].iloc[test_indices].values
daily_rows = {d: g for d, g in D.groupby('date')}

MIN_R = 0.002

# ---------- расчёт R_pred и med_r на все тестовые дни (общий для всех вариантов) ----------
r_preds = {}
meds = {}
rng_model = None

def fit_range_till(before_idx):
    global rng_model
    mask = (days.index < before_idx) & feat_valid
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(F[mask], day_agg['range'].iloc[mask].values)
    rng_model = m

for ti in test_indices:
    fit_range_till(ti)
    r_preds[ti] = rng_model.predict(F.iloc[ti:ti + 1])[0]
    meds[ti] = day_agg['range'].iloc[:ti].median()

# ---------- симуляция дня для конфигурации ----------
def simulate_day(rows, day_open, r_pred, med_r, prev_body, cfg):
    """Возвращает pnl_pips, число сделок, lable исходов."""
    do_trade = r_pred > MIN_R
    if cfg.get('vol_filter') and do_trade:
        do_trade = r_pred > med_r
    if not do_trade:
        return 0.0, 0, 0

    tp_f = cfg.get('tp', 0.5); sl_f = cfg.get('sl', 0.5)
    delay = cfg.get('delay', 0)
    mode = cfg.get('mode', 'longonly')

    if mode == 'dir_body':
        direc = 1 if prev_body >= 0 else -1
    else:
        direc = 1

    bars = list(rows.itertuples(index=False))
    entry = day_open
    if delay > 0 and len(bars) > delay:
        entry = bars[delay].open

    tp = entry + direc * tp_f * r_pred
    sl = entry - direc * sl_f * r_pred
    state = 1 if direc > 0 else -1
    n_tr = 1

    for bar in bars[delay:]:
        if state == 1:
            if bar.low <= sl:
                pnl = (sl - entry) * 10000
                return pnl, n_tr, 'SL'
            elif bar.high >= tp:
                pnl = (tp - entry) * 10000
                return pnl, n_tr, 'TP'
        else:
            if bar.high >= sl:
                pnl = (entry - sl) * 10000
                return pnl, n_tr, 'SL'
            elif bar.low <= tp:
                pnl = (entry - tp) * 10000
                return pnl, n_tr, 'TP'

    if state == 1:
        pnl = (bars[-1].close - entry) * 10000
        return pnl, n_tr, 'EOF'
    else:
        pnl = (entry - bars[-1].close) * 10000
        return pnl, n_tr, 'EOF'

VARIANTS = {
    'V0 base   ': {'mode': 'longonly', 'tp': 0.5, 'sl': 0.5, 'delay': 0, 'vol_filter': False},
    'V1 dir    ': {'mode': 'dir_body', 'tp': 0.5, 'sl': 0.5, 'delay': 0, 'vol_filter': False},
    'V2 time   ': {'mode': 'longonly', 'tp': 0.5, 'sl': 0.5, 'delay': 2, 'vol_filter': False},
    'V3 async  ': {'mode': 'longonly', 'tp': 0.6, 'sl': 0.5, 'delay': 0, 'vol_filter': False},
    'V4 vol    ': {'mode': 'longonly', 'tp': 0.5, 'sl': 0.5, 'delay': 0, 'vol_filter': True},
}

results = {}
fig, ax = plt.subplots(figsize=(14, 7))

for name, cfg in VARIANTS.items():
    trades = []
    equity_pips = 0.0
    equity_scaled = 0.0
    curve = []
    daily = []
    outs = {'TP': 0, 'SL': 0, 'EOF': 0}

    for pos, ti in enumerate(test_indices):
        rows = daily_rows[test_dates[pos]]
        day_open = day_agg['o'].iloc[ti]
        r_pred = r_preds[ti]
        med_r = meds[ti]
        prev_body = bodies.iloc[ti - 1]

        pnl, ntr, out = simulate_day(rows, day_open, r_pred, med_r, prev_body, cfg)
        if ntr:
            outs[out] += 1
            trades.append((test_dates[pos], pnl, out))
        day_pnl = pnl if ntr else 0.0

        w = med_r / r_pred if r_pred > 0 else 1.0
        day_pnl_scaled = day_pnl * w
        equity_pips += day_pnl
        equity_scaled += day_pnl_scaled
        curve.append(equity_pips)
        daily.append(day_pnl)

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t[1] > 0)
    dd = 0.0; peak = -1e9
    for v in curve:
        peak = max(peak, v); dd = min(dd, v - peak)
    sharpe = (np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252)

    results[name] = {
        'pips': equity_pips, 'scaled': equity_scaled, 'tr': n_trades,
        'win': (100 * n_wins / n_trades if n_trades else 0),
        'sharpe': sharpe, 'dd': dd, 'outs': outs,
    }
    ax.plot(range(len(test_dates)), curve, marker='o', label=f"{name} -> {equity_pips:+.0f} pips")

ax.axhline(0, color='gray', linestyle='--')
ax.legend()
ax.set_title('v12: 5 направлений улучшения (кумулятивный PnL, 30 тестовых дней)')
ax.set_ylabel('PnL, pips')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('v12_variants_backtest.png')
plt.show()

# ---------- таблица ----------
print("\nВерсия      | pips   | норм  | сделки | win%  | Sharpe | MaxDD  | TP/SL/EOF")
for name, r in results.items():
    o = r['outs']
    print(f"{name} | {r['pips']:+7.1f} | {r['scaled']:+7.1f} | {r['tr']:5d} | {r['win']:5.0f}% | {r['sharpe']:6.2f} | {r['dd']:7.1f} | {o['TP']}/{o['SL']}/{o['EOF']}")

print("\nPlot saved as v12_variants_backtest.png")