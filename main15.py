import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# ============================================================
# v15: делаем V3 устойчивее к режиму
# Дрейфовый гейт: каузальная гаусс-производная дневных закрытий (sigma=3д)
#   V6a: long только когда drift>0, иначе пропуск дня
#   V6b: long когда drift>0, short когда drift<0
# Сравнение с V3 (long всегда) на тех же 4 окнах
# ============================================================

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
D = df[['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']].copy()
D.columns = ['date', 'time', 'open', 'high', 'low', 'close']
D = D.reset_index(drop=True)

# ---------- каузальная производная по ЧАСОВЫМ (для справки) ----------
SIG_G = 5
W = 3 * SIG_G
s = np.arange(0, W + 1)
w = np.exp(-s * s / (2 * SIG_G * SIG_G)); w = w / w.sum()
close = D['close'].values
n = len(close)
smooth_full = np.full(n, np.nan)
for t in range(W, n):
    smooth_full[t] = np.dot(w, close[t - W:t + 1][::-1])

# ---------- Дневная агрегация ----------
day_agg = D.groupby('date').agg(
    o=('open', 'first'), h=('high', 'max'), l=('low', 'min'),
    c=('close', 'last'), nbars=('close', 'count'),
).reset_index()
day_agg['range'] = day_agg['h'] - day_agg['l']
day_agg['body'] = day_agg['c'] - day_agg['o']
day_agg['gap'] = day_agg['o'] - day_agg['c'].shift(1)

# ---------- каузальная производная по ДНЕВНЫМ закрытиям (drift-gate) ----------
SIG_D = 3
WD = 3 * SIG_D
sd = np.arange(0, WD + 1)
wd = np.exp(-sd * sd / (2 * SIG_D * SIG_D)); wd = wd / wd.sum()
dclose = day_agg['c'].values
nd = len(dclose)
smooth_d = np.full(nd, np.nan)
for t in range(WD, nd):
    smooth_d[t] = np.dot(wd, dclose[t - WD:t + 1][::-1])
drift_day = np.full(nd, np.nan)
for t in range(SIG_D, nd):
    drift_day[t] = smooth_d[t] - smooth_d[t - SIG_D]

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

MIN_R = 0.002
TOTAL = len(days)

windows = {
    f"W1 {days['date'].iloc[TOTAL-30]}..{days['date'].iloc[-1]}":        list(range(TOTAL-30, TOTAL)),
    f"W2 {days['date'].iloc[TOTAL-60]}..{days['date'].iloc[TOTAL-31]}": list(range(TOTAL-60, TOTAL-30)),
    f"W3 {days['date'].iloc[TOTAL-90]}..{days['date'].iloc[TOTAL-61]}": list(range(TOTAL-90, TOTAL-60)),
    f"W4 {days['date'].iloc[TOTAL-120]}..{days['date'].iloc[TOTAL-91]}": list(range(TOTAL-120, TOTAL-90)),
}

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
    per_day[ti] = {
        'r_pred': rng_model.predict(F.iloc[ti:ti + 1])[0],
        'med_r': day_agg['range'].iloc[:ti].median(),
        'drift': drift_day[ti - 1],
    }

def simulate_day(rows, day_open, r_pred, med_r, tp_f=0.6, sl_f=0.5, direc=1):
    bars = list(rows.itertuples(index=False))
    entry = day_open
    tp = entry + direc * tp_f * r_pred
    sl = entry - direc * sl_f * r_pred
    state = 1 if direc > 0 else -1
    for bar in bars:
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

# V3: long всегда; V6a: long только drift>0, иначе пропуск; V6b: long/short по drift
def make_trade(ti, cfg):
    pd_ = per_day[ti]
    r_pred = pd_['r_pred']
    med_r = pd_['med_r']
    day_open = day_agg['o'].iloc[ti]
    rows = D[D['date'] == days['date'].iloc[ti]]
    drift = pd_['drift']
    mode = cfg
    if mode == 'V3':
        return simulate_day(rows, day_open, r_pred, med_r)
    elif mode == 'V6a':
        if drift > 0:
            return simulate_day(rows, day_open, r_pred, med_r)
        return 0.0, 0
    elif mode == 'V6b':
        d = 1 if drift > 0 else -1
        if drift == 0:
            return 0.0, 0
        return simulate_day(rows, day_open, r_pred, med_r, direc=d)

summary = {}
for mode in ['V3', 'V6a', 'V6b']:
    summary[mode] = []
    print(f"\n=== {mode} ===")
    for wname, widx in windows.items():
        trades = []
        eq = 0.0; eqs = 0.0; curve = []; daily = []
        nl = ns = nskip = 0
        for ti in widx:
            pd_ = per_day[ti]
            if mode == 'V6a' and pd_['drift'] <= 0:
                nskip += 1
            pnl, ntr = make_trade(ti, mode)
            if ntr:
                trades.append(pnl)
                if mode != 'V3':
                    if pd_['drift'] > 0: nl += 1
                    else: ns += 1
            w = pd_['med_r'] / pd_['r_pred'] if pd_['r_pred'] > 0 else 1.0
            eq += pnl if ntr else 0.0
            eqs += (pnl * w) if ntr else 0.0
            curve.append(eq); daily.append(pnl if ntr else 0.0)
        nt = len(trades)
        nw = sum(1 for t in trades if t > 0)
        dd = 0.0; peak = -1e9
        for v in curve:
            peak = max(peak, v); dd = min(dd, v - peak)
        sh = (np.mean(daily) / (np.std(daily) + 1e-12)) * np.sqrt(252)
        extra = f" | skip {nskip}, L/S={nl}/{ns}" if mode != 'V3' else ""
        print(f"  {wname}: {eq:+7.1f} pips (н {eqs:+6.1f}) | {nt:2d} сд | win {100*nw/(nt or 1):3.0f}% | Sh {sh:5.2f} | DD {dd:6.1f}{extra}")
        summary[mode].append(eq)

print("\n=== ИТОГ (сумма по 4 окнам) ===")
for mode in summary:
    vals = summary[mode]
    pos = sum(1 for x in vals if x > 0)
    print(f"  {mode}: {sum(vals):+8.1f} pips | плюс. окон {pos}/4 | {[f'{x:+.0f}' for x in vals]}")

xt = np.arange(4); w = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
colors = {'V3': 'gray', 'V6a': 'green', 'V6b': 'blue'}
for i, mode in enumerate(['V3', 'V6a', 'V6b']):
    ax.bar(xt + i * w, summary[mode], width=w, color=colors[mode], label=mode)
ax.set_xticks(xt + w); ax.set_xticklabels(list(windows.keys()), rotation=15, ha='right')
ax.axhline(0, color='gray', linestyle='--')
ax.set_title('v15: V3 vs дрейфовый гейт (V6a пропуск, V6b long/short)')
ax.set_ylabel('PnL за окно, pips'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('v15_drift_gate.png'); plt.show()
print("\nPlot saved as v15_drift_gate.png")