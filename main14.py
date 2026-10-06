import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# v14: валидация recycler на 4-х окнах по 30 дней
# + кандидат V5: V3 (TP=0.6R/SL=0.5R) + каузальная производная Гаусса
#    как фильтр направления (long if deriv>0, short if deriv<0)
# Каузальная производная (вариант B): smooth[t] - smooth[t-sigma],
#    smooth - одностороннее гаусс-сглаживание по лагам 0..3*sigma (только прошлое)
# ============================================================

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
D = df[['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['date', 'time', 'open', 'high', 'low', 'close']
D = D.reset_index(drop=True)

close = D['close'].values

# ---------- Каузальное гаусс-сглаживание и производная ----------
SIG_G = 5
W = 3 * SIG_G
s = np.arange(0, W + 1)
w = np.exp(-s * s / (2 * SIG_G * SIG_G))
w = w / w.sum()

n = len(close)
smooth_full = np.full(n, np.nan)
for t in range(W, n):
    smooth_full[t] = np.dot(w, close[t - W:t + 1][::-1])
deriv_full = np.full(n, np.nan)
for t in range(SIG_G, n):
    deriv_full[t] = smooth_full[t] - smooth_full[t - SIG_G]

# Даты -> индекс первой свечи дня
first_idx_by_date = D.groupby('date')['open'].apply(lambda g: g.index[0]).to_dict()

# ---------- Дневная агрегация ----------
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

MIN_R = 0.002
TOTAL = len(days)

# 4 окна по 30 дней
windows = {
    f"W1 {days['date'].iloc[TOTAL-30]}..{days['date'].iloc[-1]}":  list(range(TOTAL-30, TOTAL)),
    f"W2 {days['date'].iloc[TOTAL-60]}..{days['date'].iloc[TOTAL-31]}": list(range(TOTAL-60, TOTAL-30)),
    f"W3 {days['date'].iloc[TOTAL-90]}..{days['date'].iloc[TOTAL-61]}": list(range(TOTAL-90, TOTAL-60)),
    f"W4 {days['date'].iloc[TOTAL-120]}..{days['date'].iloc[TOTAL-91]}": list(range(TOTAL-120, TOTAL-90)),
}

# ---------- Ход вперёд: R_pred на все тестовые дни ----------
all_test = sorted(set().union(*windows.values()))
rng_model = None
per_day = {}

def fit_range_till(before_idx):
    global rng_model
    mask = (days.index < before_idx) & feat_valid
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1, n_jobs=-1)
    m.fit(F[mask], day_agg['range'].iloc[mask].values)
    rng_model = m

for ti in all_test:
    fit_range_till(ti)
    b = first_idx_by_date[days['date'].iloc[ti]] - 1  # последний завершённый бар перед днём
    per_day[ti] = {
        'r_pred': rng_model.predict(F.iloc[ti:ti + 1])[0],
        'med_r': day_agg['range'].iloc[:ti].median(),
        'body_prev': bodies.iloc[ti - 1],
        'deriv': deriv_full[b],
    }

# ---------- Симуляция ----------
def simulate_day(rows, day_open, r_pred, med_r, cfg, direc):
    do_trade = r_pred > MIN_R
    if cfg.get('vol_filter') and do_trade:
        do_trade = r_pred > med_r
    if not do_trade:
        return 0.0, 0

    tp_f = cfg.get('tp', 0.6)
    sl_f = cfg.get('sl', 0.5)
    delay = cfg.get('delay', 0)
    m = cfg.get('dir_mode', 'long')

    d = 1
    if m == 'gauss':
        d = direc if direc != 0 else 1
    elif m == 'body':
        d = 1 if cfg_prev_body >= 0 else -1   # pragma: no cover (не используется)

    bars = list(rows.itertuples(index=False))
    entry = day_open
    if delay > 0 and len(bars) > delay:
        entry = bars[delay].open

    tp = entry + d * tp_f * r_pred
    sl = entry - d * sl_f * r_pred
    state = 1 if d > 0 else -1

    for bar in bars[delay:]:
        if state == 1:
            if bar.low <= sl:
                return (sl - entry) * 10000, 1
            elif bar.high >= tp:
                return (tp - entry) * 10000, 1
        else:
            if bar.high >= sl:
                return (entry - sl) * 10000, 1
            elif bar.low <= tp:
                return (entry - tp) * 10000, 1

    if state == 1:
        return (bars[-1].close - entry) * 10000, 1
    return (entry - bars[-1].close) * 10000, 1

cfg_prev_body = 0  # заглушка, вариант 'body' не запускаем

VARIANTS = {
    'V0 base ': {'tp': 0.5, 'sl': 0.5, 'delay': 0, 'dir_mode': 'long'},
    'V3 async': {'tp': 0.6, 'sl': 0.5, 'delay': 0, 'dir_mode': 'long'},
    'V5 gauss': {'tp': 0.6, 'sl': 0.5, 'delay': 0, 'dir_mode': 'gauss'},
}

summary = {v: [] for v in VARIANTS}

for wname, widx in windows.items():
    print(f"\n=== {wname} ===")
    for vname, cfg in VARIANTS.items():
        trades = []
        equity_pips = 0.0
        equity_scaled = 0.0
        curve = []
        daily = []
        nlong = 0
        nshort = 0
        for ti in widx:
            pd_ = per_day[ti]
            rows = D[D['date'] == days['date'].iloc[ti]]
            day_open = day_agg['o'].iloc[ti]
            r_pred = pd_['r_pred']
            med_r = pd_['med_r']
            direc = pd_['deriv']
            d = 1 if direc > 0 else (-1 if direc < 0 else 1)
            if cfg['dir_mode'] == 'gauss':
                nlong += (d > 0)
                nshort += (d < 0)
            pnl, ntr = simulate_day(rows, day_open, r_pred, med_r, cfg, direc)
            if ntr:
                trades.append((days['date'].iloc[ti], pnl))
            w = med_r / r_pred if r_pred > 0 else 1.0
            equity_pips += pnl if ntr else 0.0
            equity_scaled += (pnl * w) if ntr else 0.0
            curve.append(equity_pips)
            daily.append(pnl if ntr else 0.0)

        nt = len(trades)
        nw = sum(1 for t in trades if t[1] > 0)
        dd = 0.0; peak = -1e9
        for v in curve:
            peak = max(peak, v); dd = min(dd, v - peak)
        sh = np.std(daily)
        shp = (np.mean(daily) / (sh + 1e-12)) * np.sqrt(252)
        print(f"  {vname}: {equity_pips:+7.1f} pips (норм {equity_scaled:+6.1f}) | сделок {nt:2d} | win {100*nw/(nt or 1):3.0f}% | "
              f"Sharpe {shp:5.2f} | DD {dd:6.1f}" + (f" | L/S={nlong}/{nshort}" if cfg['dir_mode'] == 'gauss' else ""))
        summary[vname].append(equity_pips)

# ---------- Итоговый рейтинг ----------
print("\n=== ИТОГ ПО ВСЕМ 4 ОКНАМ (сумма pips по окнам) ===")
for vname in VARIANTS:
    vals = summary[vname]
    pos = sum(1 for x in vals if x > 0)
    print(f"  {vname}: сумма {sum(vals):+8.1f} | плюсовых окон {pos}/4 | по окнам {[f'{x:+.0f}' for x in vals]}")

# ---------- График: V3 vs V5 по 4 окнам ----------
xt = np.arange(4)
w = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
for i, vname in enumerate(VARIANTS):
    ax.bar(xt + i * w, summary[vname], width=w, label=vname)
ax.set_xticks(xt + w)
ax.set_xticklabels(list(windows.keys()), rotation=15, ha='right')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title('v14: recycler на 4 окнах (V0 база, V3 асимметрия, V5 гаусс-фильтр)')
ax.set_ylabel('PnL за окно, pips')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('v14_four_windows.png')
plt.show()

print("\nPlot saved as v14_four_windows.png")