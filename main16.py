import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna

# ============================================================
# v16: LightGBM - направление следующих 4 свечей + Optuna
#  Валидация Оптуны: месяц W2 (30 дней перед тестом), критерий - прибыль/шарп/просадка
#  Отбираем 15 чемпионов, прогоняем на ТЕСТОВОМ месяце W1 (2026.07.30..09.09)
#  и рисуем 15 эквити одной картинкой
# ============================================================

H = 4          # горизонт: направление следующих 4 свечей
SEED = 42

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
D = pd.read_csv(file_path, sep='\t')
D = D.rename(columns={'<OPEN>': 'open', '<HIGH>': 'high', '<LOW>': 'low', '<CLOSE>': 'close',
                      '<DATE>': 'date', '<TIME>': 'time'})
D = D.reset_index(drop=True)

close = D['close'].values
open_ = D['open'].values
n = len(D)

# ---------- фичи (только прошлое) ----------
X = pd.DataFrame()
X['r1'] = D['close'].pct_change()
X['r4'] = D['close'].pct_change(4)
for lag in range(1, 31):
    X[f'lag_{lag}'] = D['close'].shift(lag)
for lag in range(1, 11):
    X[f'rlag_{lag}'] = D['close'].pct_change().shift(lag)
X['sma20'] = D['close'].rolling(20).mean()
X['std20'] = D['close'].rolling(20).std()
X['sma5_dev'] = (D['close'] - D['close'].rolling(5).mean())
X['hh20'] = D['high'].rolling(20).max()
X['ll20'] = D['low'].rolling(20).min()

# время суток (циклические признаки)
hour = pd.to_datetime(D['date'] + ' ' + D['time'], format='%Y.%m.%d %H:%M:%S').dt.hour
X['h_sin'] = np.sin(2 * np.pi * hour / 24)
X['h_cos'] = np.cos(2 * np.pi * hour / 24)

# целевая: знак изменения цены за следующие H свечей
y = (D['close'].shift(-H) > D['close']).astype(int)

data = pd.concat([X, y.rename('target')], axis=1).dropna()
Xa = data.drop(columns=['target'])
ya = data['target'].astype(int).values
Xa = Xa.values  # для скорости

# ---------- границы месяцев ----------
def first_idx(date_str):
    return int(np.where(D['date'] == date_str)[0][0])

W1_start = first_idx('2026.07.30')   # тест
W2_start = first_idx('2026.06.18')   # валидация Оптуны
split_tr = first_idx('2026.06.10')   # начало «прогрева» для обучения (не критично)

# Сессии для бэктеста: решения каждые H баров (не пересекаются)
def sessions_idxs(start_bar, end_bar):
    return list(range(start_bar, end_bar - H))

def get_proba(model, start_bar, end_bar):
    hi = min(end_bar - H, len(Xa))
    idx = np.arange(start_bar, hi)
    return idx, model.predict_proba(Xa[idx])[:, 1]

def run_backtest(idx, proba, margin=0.03):
    sessions = []
    eq = 0.0
    curve = []
    for k, i in enumerate(idx):
        p = proba[k]
        if abs(p - 0.5) < margin:
            sessions.append(0.0)
            curve.append(eq)
            continue
        pred = 1 if p >= 0.5 else -1
        pnl = pred * (close[i + H] - open_[i + 1]) * 10000
        eq += pnl
        sessions.append(pnl)
        curve.append(eq)
    return eq, np.array(sessions), np.array(curve)

def metrics(curve, sessions):
    dd = 0.0; peak = -1e9
    for v in curve:
        peak = max(peak, v); dd = min(dd, v - peak)
    if np.std(sessions) < 1e-12:
        sh = 0.0
    else:
        sh = (np.mean(sessions) / (np.std(sessions) + 1e-12)) * np.sqrt(252 * 24 / H)
    return dd, sh

def make_model(trial):
    params = dict(
        objective='binary',
        n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        num_leaves=trial.suggest_int('num_leaves', 8, 128, step=4),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=SEED,
        verbose=-1, n_jobs=-1,
    )
    return lgb.LGBMClassifier(**params)

# Обучаемся на строках ДО валидационного/тестового окна (с запасом в H баров на таргет)
tr_mask = np.arange(split_tr, W2_start - H)

def objective(trial):
    m = make_model(trial)
    m.fit(Xa[tr_mask], ya[tr_mask])
    idx, proba = get_proba(m, W2_start, W1_start)
    eq, sessions, curve = run_backtest(idx, proba)
    dd, sh = metrics(curve, sessions)
    ntr = int(np.sum(np.abs(sessions) > 1e-12))
    trial.set_user_attr('pnl', eq)
    trial.set_user_attr('dd', dd)
    trial.set_user_attr('sharpe', sh)
    trial.set_user_attr('ntr', ntr)
    return sh

print("=== Optuna: подбор параметров (валидация = месяц W2) ===")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=80, show_progress_bar=False)

# чемпионы: прибыльные, с реальными сделками, лучший шарп (при равном - меньшая просадка)
champ = [t for t in study.trials if t.user_attrs.get('pnl', -1) > 0 and t.user_attrs.get('ntr', 0) > 0]
champ.sort(key=lambda t: (t.user_attrs['sharpe'], -t.user_attrs['dd']), reverse=True)
champions = champ[:15]

print(f"Оптуна завершила {len(study.trials)} проб, прибыльных с сделками: {len(champ)}")
print("Чемпионы (валидация W2):")
for i, t in enumerate(champions):
    print(f"  #{i+1}: trial {t.number:3d} | pnl {t.user_attrs['pnl']:+6.1f} | sharpe {t.user_attrs['sharpe']:5.2f} | DD {t.user_attrs['dd']:6.1f} | сделок {t.user_attrs['ntr']:3d}")

# ---------- тест на W1: все 15 чемпионов ----------
print("\n=== Тест на W1 (2026.07.30..09.09) ===")

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(champions)))
test_rows = []

for i, t in enumerate(champions):
    params = t.params
    m = lgb.LGBMClassifier(objective='binary', random_state=SEED, verbose=-1, n_jobs=-1, **params)
    m.fit(Xa[:W1_start - H], ya[:W1_start - H])
    idx, proba = get_proba(m, W1_start, n)
    eq, sessions, curve = run_backtest(idx, proba)
    dd, sh = metrics(curve, sessions)
    n_wins = int(np.sum(sessions > 0))
    test_rows.append((eq, dd, sh, len(sessions), n_wins))
    ax.plot(np.arange(len(curve)), curve, color=colors[i], alpha=0.8,
            label=f"trial {t.number}: {eq:+.0f}p, Sh {sh:.2f}, DD {dd:.0f}")

ax.axhline(0, color='gray', linestyle='--')
ax.set_title(f'v16: тест на месяце W1 - 15 чемпионов Optuna (LightGBM, направление на {H} свечей)')
ax.set_xlabel('Сессия (4-баровые)'); ax.set_ylabel('PnL, pips')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('v16_optuna_15_champions.png')
plt.show()

print("\nЭквити 15 чемпионов на W1 (pnl, DD, sharpe, сделок, wins):")
for i, (eq, dd, sh, ns, nw) in enumerate(test_rows):
    print(f"  #{i+1}: pnl {eq:+7.1f} | DD {dd:6.1f} | Sh {sh:5.2f} | сессий {ns} | win {100*nw/max(ns,1):.0f}%")
print("\nPlot saved as v16_optuna_15_champions.png")