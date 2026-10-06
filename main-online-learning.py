import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "D:\\PYTHON\\Первая-производная-гаусса\\EURUSD_H1_202412270000_202609092300.csv"
df = pd.read_csv(file_path, sep='\t')
price = df['<CLOSE>'].values
returns = np.log(df['<CLOSE>'] / df['<CLOSE>'].shift(1)).fillna(0).values
hours = df['<TIME>'].str.split(':').str[0].astype(int).values
tickvol = df['<TICKVOL>'].values.astype(float)
spread = df['<SPREAD>'].values.astype(float)
N = len(df)

cum = np.concatenate([[0.0], np.cumsum(returns)])
MOM_K = [1, 2, 3, 5, 8, 13, 21, 34, 55]
BASE = len(MOM_K) + 5
N_FEAT = BASE + 24
FEAT_NAMES = [f'mom{k}' for k in MOM_K] + ['rev(3_55)', 'vol', 'tvol', 'tvol_chg', 'spread'] + \
             [f'hr{hh}' for hh in range(24)]

START = 600
EVAL_N = 6000
EVAL0 = N - EVAL_N
ALPHA_STAT = 0.01
ALPHA_VOL = 0.01
ALPHA_TV = 0.01
EPS = 1e-3
LR = 0.02

def build_features(t, vol_prev, tv_prev):
    f = [cum[t] - cum[t - k] for k in MOM_K]
    f.append(f[2] - f[8])
    f.append(vol_prev)
    f.append(tickvol[t])
    f.append(tickvol[t] - tv_prev)
    f.append(spread[t])
    hd = np.zeros(24)
    hd[hours[t]] = 1.0
    f = np.concatenate([np.array(f, dtype=float), hd])
    return f

def run_online(dz, cost):
    w = np.zeros(N_FEAT)
    rms = np.full(N_FEAT, 1e-2)
    f_mean = np.zeros(N_FEAT)
    f_var = np.ones(N_FEAT)
    vol_ema = np.std(returns[:300]) + 1e-9
    tv_ema = np.mean(tickvol[:300]) + 1e-9
    score_mean = 0.0
    score_var = 1.0

    eq = np.ones(N)
    wins = np.zeros(N)
    acted = np.zeros(N)
    win_cnt = np.zeros(24)
    win_sum = np.zeros(24)

    last_pos = 0.0

    for t in range(START, N):
        x = build_features(t, vol_ema, tv_ema)
        xn = (x - f_mean) / (np.sqrt(f_var) + EPS)
        score = float(w @ xn)

        score_mean = ALPHA_STAT * score + (1 - ALPHA_STAT) * score_mean
        score_var = ALPHA_STAT * score**2 + (1 - ALPHA_STAT) * score_var
        score_std = np.sqrt(max(score_var - score_mean**2, 1e-12))

        if abs(score) > dz * score_std:
            pos = np.sign(score)
        else:
            pos = 0.0

        r = returns[t]
        c = cost * abs(pos - last_pos)
        pnl = pos * r - c
        eq[t] = eq[t - 1] * (1 + pnl)

        if pos != 0:
            wins[t] = int(pos * r > 0)
            acted[t] = 1
            hh = hours[t]
            win_cnt[hh] += 1
            win_sum[hh] += wins[t]

        u = float(np.clip(r / (vol_ema + EPS), -3, 3))
        err = score - u
        rms = 0.9 * rms + 0.1 * xn**2
        w -= LR * err * xn / (np.sqrt(rms) + EPS)

        f_mean = ALPHA_STAT * x + (1 - ALPHA_STAT) * f_mean
        f_var = ALPHA_STAT * x**2 + (1 - ALPHA_STAT) * f_var
        vol_ema = ALPHA_VOL * np.abs(r) + (1 - ALPHA_VOL) * vol_ema
        tv_ema = ALPHA_TV * tickvol[t] + (1 - ALPHA_TV) * tv_ema
        last_pos = pos

    return eq, wins, acted, w, win_cnt, win_sum

def metrics(eq, wins, acted):
    s0 = max(START, EVAL0) - 1
    seg_e = eq[s0:]
    rpnl = np.diff(seg_e) / seg_e[:-1]
    sharpe = np.mean(rpnl) / (np.std(rpnl) + 1e-12) * np.sqrt(24 * 365)
    peak = np.maximum.accumulate(seg_e)
    maxdd = np.max((peak - seg_e) / peak)
    act = acted[max(START, EVAL0):]
    ww = wins[max(START, EVAL0):]
    trades = int(np.sum(act > 0))
    winrate = np.mean(ww[act > 0]) if (act > 0).any() else np.nan
    return seg_e[-1], sharpe, maxdd, trades, winrate

print(f"bars={N}, online from t={START}, eval region: {max(START, EVAL0)}..{N}, features={N_FEAT}")
print("=" * 96)
print(f"{'deadzone':<9}{'cost(pip)':<10}{'final':>9}{'sharpe':>8}{'maxDD%':>8}{'trades':>8}{'winrate%':>9}")
print("-" * 96)

best = None
for dz in [0.0, 0.5, 1.0]:
    for cost_pips in [0.0, 0.3, 1.0]:
        cost = cost_pips * 0.0001
        eq, wins, acted, w, win_cnt, win_sum = run_online(dz, cost)
        final, sharpe, maxdd, trades, winrate = metrics(eq, wins, acted)
        if best is None or final > best[0]:
            best = (final, dz, cost, eq, wins, acted, w, win_cnt, win_sum, sharpe, maxdd)
        print(f"{dz:<9}{int(cost_pips):<10}{final:>9.3f}{sharpe:>8.2f}{maxdd*100:>7.1f}%{trades:>8}{winrate*100:>8.1f}%")

final, dz, cost, eq, wins, acted, w, win_cnt, win_sum, sharpe, maxdd = best
print("=" * 96)
print(f"BEST: deadzone={dz}, cost={int(cost*10000)} pips, final={final:.3f}x, sharpe={sharpe:.2f}, maxDD={maxdd*100:.1f}%")

ev = np.arange(max(START, EVAL0), N)
seg_eq = eq[ev] / eq[max(START, EVAL0) - 1]
seg_bh = price[ev] / price[ev[0]]

winrate_ewma = np.full(len(ev), np.nan)
wr = 0.5
for i, t in enumerate(ev):
    if acted[t]:
        wr = 0.02 * wins[t] + 0.98 * wr
    winrate_ewma[i] = wr

hr_wr = np.where(win_cnt > 0, win_sum / np.maximum(win_cnt, 1), np.nan)
hr_counts = win_cnt

order = np.argsort(-np.abs(w))
top_feats = [(FEAT_NAMES[i], w[i]) for i in order[:10]]

fig, ax = plt.subplots(2, 2, figsize=(16, 10))
ax[0, 0].plot(np.arange(len(ev)), seg_eq, label=f'Online self-learning ({seg_eq[-1]:.2f}x)', linewidth=2)
ax[0, 0].plot(np.arange(len(ev)), seg_bh, label=f'Buy & Hold ({seg_bh[-1]:.2f}x)', alpha=0.7, linestyle=':')
ax[0, 0].axhline(1.0, color='black', linewidth=0.6)
ax[0, 0].set_yscale('log')
ax[0, 0].set_title('Equity (online, walk-forward)')
ax[0, 0].legend()
ax[0, 1].plot(np.arange(len(ev)), winrate_ewma, color='green', linewidth=1.5)
ax[0, 1].axhline(0.5, color='black', linewidth=0.6)
ax[0, 1].set_ybound(0.35, 0.65)
ax[0, 1].set_title('Rolling win rate (EWMA)')
ax[1, 0].bar(range(24), hr_wr, color='steelblue')
ax[1, 0].set_xticks(range(24))
ax[1, 0].set_title('Win rate by hour of day')
hax = ax[1, 0].twinx()
hax.bar(range(24), hr_counts / max(hr_counts.max(), 1), alpha=0.3, color='gray')
hax.set_ylabel('trade density')
ax[1, 1].barh(range(len(top_feats)), [x[1] for x in top_feats], color='steelblue')
ax[1, 1].set_yticks(range(len(top_feats)))
ax[1, 1].set_yticklabels([x[0] for x in top_feats])
ax[1, 1].set_title('Top |w| at the end')
plt.suptitle('Self-Learning System: decide at each bar open, learn from realized errors/successes')
plt.tight_layout()
plt.savefig('online_selflearning_equity.png')
plt.show()