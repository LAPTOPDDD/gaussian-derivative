"""
Аппроксиматор первой производной свёртки Гаусса  d/dt (G_sigma * log P)
на LightGBM по каузальным фичам.  EURUSD H1.

Учитель  : НЕкаузальный (центрированное ядро +-4*sigma) - сознательно разрешено.
Фичи     : строго каузальные (прошлое + текущий бар).
Тест     : последние 200 баров, отделён purge-разрывом = 4*sigma,
           иначе цель-учитель на конце трейна заглядывала бы в тест.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

np.random.seed(42)
SIGMA, TRUNC, TEST_N, SCALE = 10.0, 4.0, 200, 1e4
# путь к CSV можно передать первым аргументом:  python gauss_deriv_lgb.py <file.csv>
import sys
PATH = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_H1_202412270000_202609092300.csv"

# ============================================================ 1. данные
df = pd.read_csv(PATH, sep="\t")
df.columns = [c.strip("<>") for c in df.columns]
df["dt"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="%Y.%m.%d %H:%M:%S")
df = df.sort_values("dt").reset_index(drop=True)
for c in ["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL"]:
    df[c] = df[c].astype(float)
c_, n = df["CLOSE"].values, len(df)
lp = np.log(c_)
print(f"H1 баров: {n}   {df.dt.iloc[0]} .. {df.dt.iloc[-1]}")

# ========================================= 2. учитель: d/dt (G_sigma * lp)
R = int(np.ceil(TRUNC * SIGMA))
k = np.arange(-R, R + 1)
g = np.exp(-0.5 * (k / SIGMA) ** 2); g /= g.sum()
dg = -(k / SIGMA ** 2) * g
print(f"ядро: {len(k)} отсчётов, окно цели = +-{R} баров (4 sigma)")

def conv(x, kern):
    return np.convolve(np.pad(x, (R, R), mode="reflect"), kern, mode="valid")

smoothed = conv(lp, g)
target   = conv(lp, dg)          # (G' * p) - аналитическая производная
y = target * SCALE               # bp / бар
df["smooth_px"] = np.exp(smoothed); df["y"] = y

# разложение цели на вклад прошлых / будущих ДОХОДНОСТЕЙ (точное: sum_k w[k] = 0).
# ВАЖНО: np.convolve разворачивает ядро -> target[i] = sum_k w[k]*p[i-k],
# поэтому k>0 даёт ПРОШЛОЕ, k<0 - БУДУЩЕЕ.
Pp = np.pad(lp, (R, R), mode="reflect")
past_part = np.zeros(n); fut_part = np.zeros(n)
for j, kk in enumerate(k):
    if kk == 0: continue
    seg = Pp[R - kk: R - kk + n] - lp
    if kk > 0: past_part += dg[j] * seg
    else:      fut_part  += dg[j] * seg
print(f"[проверка] max|(past+future) - target| = {np.max(np.abs(past_part + fut_part - target)):.2e}")
fp, ppart = fut_part * SCALE, past_part * SCALE

# ================================================== 3. каузальные фичи
close, high, low, tvol = df["CLOSE"], df["HIGH"], df["LOW"], df["TICKVOL"]
ret1 = np.log(close).diff()
F = pd.DataFrame(index=df.index)
for w in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
    F[f"ret_{w}"] = np.log(close).diff(w)
for w in range(1, 21):                       # покомпонентные лаги доходности
    F[f"lag_{w}"] = ret1.shift(w)
for s in [5, 10, 20, 40, 80]:
    F[f"ema{s}_dev"] = np.log(close) - np.log(close).ewm(span=s, adjust=False).mean()
for a, b in [(5, 20), (10, 40), (20, 80), (3, 13)]:
    F[f"emadiff_{a}_{b}"] = (np.log(close).ewm(span=a, adjust=False).mean()
                             - np.log(close).ewm(span=b, adjust=False).mean())
for s in [5, 10, 30]:
    F[f"ret_ema{s}"] = ret1.ewm(span=s, adjust=False).mean()
for w in [10, 30, 60]:
    sd = ret1.rolling(w).std()
    F[f"vol_{w}"] = sd
    F[f"mom_norm_{w}"] = np.log(close).diff(w) / (sd * np.sqrt(w) + 1e-12)
d = close.diff()
up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
F["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-12))
F["hl_range"] = (high - low) / close
tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
F["atr14"] = tr.ewm(alpha=1/14, adjust=False).mean() / close
for w in [20, 60]:
    F[f"pos_in_range_{w}"] = (close - close.rolling(w).min()) / (close.rolling(w).max() - close.rolling(w).min() + 1e-12)
for w in [10, 30]:
    F[f"up_frac_{w}"] = (d > 0).rolling(w).mean()
F["tvol_log"] = np.log1p(tvol)
F["tvol_ratio"] = tvol / (tvol.rolling(50).mean() + 1e-9)
F["hour"] = df.dt.dt.hour; F["dow"] = df.dt.dt.dayofweek
F["h_sin"] = np.sin(2 * np.pi * F.hour / 24); F["h_cos"] = np.cos(2 * np.pi * F.hour / 24)
FEATS = list(F.columns); X = F.copy()
# базис лагов доходностей для "виннеровского" (оптимального линейного因果ного) фильтра
Wl = pd.concat([ret1.rename("lag0")] + [ret1.shift(i).rename(f"lag{i}") for i in range(1, 61)], axis=1)
print(f"фич: {len(FEATS)}  (+61 лаг доходности для линейного фильтра)")

# ================================================ 4. сплит с purge по цели
PURGE = R
test_i0  = n - TEST_N
valid_i1 = test_i0 - PURGE - 1
valid_i0 = valid_i1 - TEST_N
train_i, valid_i, test_i = np.arange(0, valid_i0), np.arange(valid_i0, valid_i1), np.arange(test_i0, n)
good = X.notna().all(axis=1).values & Wl.notna().all(axis=1).values & np.isfinite(y)
tr = train_i[good[train_i]]; va = valid_i[good[valid_i]]; te = test_i[good[test_i]]
trv = np.concatenate([tr, va])

# ---- теоретические потолки каузальной модели на тесте -------------------
var_t, var_p, var_f = np.var(y[te]), np.var(ppart[te]), np.var(fp[te])
CEIL      = 1 - var_f / var_t                                   # потолок R²
CORR_CEIL = (var_p + np.cov(ppart[te], fp[te])[0, 1]) / (np.std(y[te]) * np.std(ppart[te]))
print(f"train {len(tr)} | valid {len(va)} | purge-gap {PURGE} | test {len(te)}")

# ============================ 5. подбор гиперпараметров по walk-forward
def mkparams(**kw):
    P = dict(objective="regression_l2", metric="l2", learning_rate=0.03, num_leaves=31,
             min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=5.0, verbosity=-1, num_threads=8, seed=42)
    P.update(kw); return P

folds, L = [], len(trv)
for e in np.linspace(0.35, 1.0, 5):
    cut = int(L * e)
    a = np.arange(0, max(cut - PURGE, 10)); b = np.arange(cut, min(cut + TEST_N, L))
    if len(b) >= 60 and len(a) >= 500: folds.append((a, b))

GRID = [dict(num_leaves=31, learning_rate=0.03, min_data_in_leaf=60, lambda_l2=5.0),
        dict(num_leaves=15, learning_rate=0.03, min_data_in_leaf=150, lambda_l2=10.0),
        dict(num_leaves=7,  learning_rate=0.05, min_data_in_leaf=250, lambda_l2=20.0),
        dict(num_leaves=63, learning_rate=0.02, min_data_in_leaf=40,  lambda_l2=2.0)]
print("\nwalk-forward подбор (медиана по фолдам):")
best_cfg, best_sc, best_it = None, -9, None
for cfg in GRID:
    P = mkparams(**cfg); r2s, cs, its = [], [], []
    for a, b in folds:
        m = lgb.train(P, lgb.Dataset(X.iloc[trv[a]], y[trv[a]]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X.iloc[trv[b]], y[trv[b]])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pr = m.predict(X.iloc[trv[b]])
        r2s.append(r2_score(y[trv[b]], pr)); cs.append(pearsonr(y[trv[b]], pr)[0]); its.append(m.best_iteration)
    sc = np.median(r2s) + 0.5 * np.median(cs)
    print(f"  {str(cfg):72s} medR2={np.median(r2s):+.3f} medCorr={np.median(cs):.3f} iters={int(np.median(its))}")
    if sc > best_sc: best_sc, best_cfg, best_it = sc, cfg, int(np.median(its))
print(f"  -> выбрано: {best_cfg}  rounds={best_it}")

P = mkparams(**best_cfg)
m0 = lgb.train(P, lgb.Dataset(X.iloc[trv], y[trv]), num_boost_round=3000,
               valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
NIT = max(best_it, 50)
mf = lgb.train(P, lgb.Dataset(X.iloc[trv], y[trv]), num_boost_round=NIT)
p_te, p_tr = mf.predict(X.iloc[te]), mf.predict(X.iloc[trv])

# ===================================================== 6. бейзлайны
def metrics(yy, pp, name):
    return dict(model=name, R2=r2_score(yy, pp),
                R2_vs_zero=1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2),
                corr=pearsonr(yy, pp)[0], spearman=spearmanr(yy, pp)[0],
                MAE=mean_absolute_error(yy, pp), RMSE=np.sqrt(mean_squared_error(yy, pp)),
                sign_acc=np.mean(np.sign(yy) == np.sign(pp)),
                R2_доля_потолка=r2_score(yy, pp) / CEIL if pp.std() > 0 else np.nan,
                corr_доля_потолка=pearsonr(yy, pp)[0] / CORR_CEIL if pp.std() > 0 else np.nan)

rows = [metrics(y[te], p_te, "LightGBM (каузальные фичи)"),
        metrics(y[trv], p_tr, "LightGBM train (in-sample)"),
        metrics(y[te], np.zeros(len(te)), "константа 0"),
        metrics(y[te], np.full(len(te), y[trv].mean()), "константа = mean(train)")]
cand = {f: r2_score(y[te], X[f].iloc[te].values * SCALE) for f in FEATS
        if f.startswith(("emadiff", "ret_", "lag_"))}
bf = max(cand, key=cand.get)
rows.append(metrics(y[te], X[bf].iloc[te].values * SCALE, f"наив без калибровки: {bf}"))
xb = X[bf].iloc[tr].values; sl = np.dot(xb, y[tr]) / np.dot(xb, xb)
rows.append(metrics(y[te], X[bf].iloc[te].values * sl, f"наив откалиброванный: {bf}"))

# виннеровский фильтр: оптимальный линейный каузальный фильтр по лагам доходностей
muW, sdW = Wl.iloc[tr].mean(), Wl.iloc[tr].std() + 1e-12
Ztr, Zva, Zte = (Wl.iloc[tr] - muW) / sdW, (Wl.iloc[va] - muW) / sdW, (Wl.iloc[te] - muW) / sdW
best_a = min([(r2_score(y[va], Ridge(alpha=a).fit(Ztr, y[tr]).predict(Zva)), a) for a in [0.1, 1, 10, 100, 1000]])[1]
Ztrv = (Wl.iloc[trv] - muW) / sdW
rw = Ridge(alpha=best_a).fit(Ztrv, y[trv])
rows.append(metrics(y[te], rw.predict(Zte), f"Виннер: Ridge по 61 лагу r (a={best_a})"))
mu, sd = X.iloc[trv].mean(), X.iloc[trv].std() + 1e-12
rg = Ridge(alpha=10).fit(((X.iloc[trv] - mu) / sd), y[trv])
rows.append(metrics(y[te], rg.predict((X.iloc[te] - mu) / sd), "Ridge по всем фичам"))

Fo = X.copy()
for w in [1, 2, 3, 5, 8, 13, 21, 34]:
    Fo[f"fut_ret_{w}"] = np.log(close).shift(-w) - np.log(close)
mo = lgb.train(P, lgb.Dataset(Fo.iloc[trv], y[trv]), num_boost_round=NIT)
rows.append(metrics(y[te], mo.predict(Fo.iloc[te]), "ОРАКУЛ (будущее доступно) = потолок"))

res = pd.DataFrame(rows)
pd.set_option("display.width", 220, "display.max_columns", 50)
print("\n" + "=" * 118)
print(f"ТЕСТ: последние {TEST_N} баров H1  ({df.dt.iloc[te[0]]} .. {df.dt.iloc[te[-1]]})")
print(f"цель = d/dt(G_sigma * log P), sigma={SIGMA:.0f} баров, единицы bp/бар;  std(цели на тесте)={np.std(y[te]):.2f}")
print("=" * 118)
print(res.round(4).to_string(index=False))

var_t, var_p, var_f = np.var(y[te]), np.var(ppart[te]), np.var(fp[te])
# Лучший каузальный прогноз = past_part + E[fut_part | прошлое].
# Если будущие доходности непрогнозируемы, E[fut|past] = 0 и невязка = var(fut_part):
CEIL = 1 - var_f / var_t
print(f"\nдостижимость цели (тест): var(цель)={var_t:.4f}")
print(f"  вклад ПРОШЛЫХ доходностей (принципиально достижимо) var={var_p:.4f}  ({var_p/var_t:.1%})")
print(f"  вклад БУДУЩИХ доходностей (недостижимо каузально)   var={var_f:.4f}  ({var_f/var_t:.1%})")
print(f"  corr(прошлая, будущая часть) = {pearsonr(ppart[te], fp[te])[0]:+.3f}")
print(f"  ==> ТЕОРЕТИЧЕСКИЙ ПОТОЛОК R² каузальной модели ~ {CEIL:.3f} "
      f"(если будущие доходности непрогнозируемы)")

# =============================== 7. walk-forward оценка стабильности
print(f"\nwalk-forward CV лучшей конфигурации ({len(folds)} фолда):")
cvs = []
for a, b in folds:
    mm = lgb.train(P, lgb.Dataset(X.iloc[trv[a]], y[trv[a]]), num_boost_round=NIT)
    pr = mm.predict(X.iloc[trv[b]])
    bb = trv[b]
    cl = 1 - np.var(fp[bb]) / np.var(y[bb])
    cvs.append((r2_score(y[bb], pr), pearsonr(y[bb], pr)[0], np.std(y[bb]), cl))
for i, (q, c1, s1, cl) in enumerate(cvs, 1):
    print(f"  fold {i}: R2={q:+.3f} (потолок {cl:.3f}, доля {q/cl:6.1%})  corr={c1:.3f}  "
          f"std(цели)={s1:.2f}  n_train={len(folds[i-1][0])}")
print(f"  median R2={np.median([c[0] for c in cvs]):.3f} | median corr={np.median([c[1] for c in cvs]):.3f} | "
      f"median доля потолка={np.median([c[0]/c[3] for c in cvs]):.1%}")

# ---- с каким лагом каузальная модель догоняет некаузального учителя ----
lags = np.arange(-25, 26); cc = []
for Lg in lags:
    a = y[te][Lg:] if Lg >= 0 else y[te][:len(te) + Lg]
    b = p_te[:len(te) - Lg] if Lg >= 0 else p_te[-Lg:]
    cc.append(pearsonr(a, b)[0])
cc = np.array(cc); bl = lags[cc.argmax()]
print(f"\nкросс-корреляция: max corr={cc.max():.3f} при сдвиге {bl:+d} бар "
      f"(corr при нулевом сдвиге {cc[lags == 0][0]:.3f})")
if bl < 0:
    yy, pp_ = y[te][:len(te) + bl], p_te[-bl:]
    print(f"  => каузальная оценка воспроизводит учителя с отставанием ~{abs(bl)} бар")
    print(f"  R² при совмещении по лагу {bl}: {r2_score(yy, pp_):.3f} "
          f"(было {r2_score(y[te], p_te):.3f});  std(учитель)={np.std(y[te]):.3f} "
          f"std(модель)={np.std(p_te):.3f}")
else:
    print("  => заметного отставания нет")
pd.DataFrame({"dt": df.dt.iloc[te].values, "target": y[te], "pred_lgb": p_te,
              "pred_wiener": rw.predict(Zte), "close": c_[te]}).to_csv("pred_test.csv", index=False)

# =========================================================== 8. графики
te_t = df.dt.iloc[te].values
R2te, CORte = r2_score(y[te], p_te), pearsonr(y[te], p_te)[0]
fig, ax = plt.subplots(5, 1, figsize=(15, 16))
ax[0].plot(te_t, c_[te], lw=.9, color="#8fa3bf", label="close")
ax[0].plot(te_t, np.exp(smoothed[te]), lw=2, color="#e8623d", label=f"$G_\\sigma * P$, $\\sigma$={SIGMA:.0f}")
ax[0].set_title("Тестовые 200 баров H1: цена и гауссово сглаживание"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(te_t, y[te], lw=2.2, color="#1f4e79", label="УЧИТЕЛЬ (некаузальный)")
ax[1].plot(te_t, p_te, lw=1.5, color="#e8623d", ls="--", label="LightGBM (каузальные фичи)")
ax[1].plot(te_t, rw.predict(Zte), lw=1.2, color="#2e8b57", ls=":", label="Виннер Ridge (линейный)")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set_title(f"Первая производная: учитель vs модели   R²={R2te:.3f}  corr={CORte:.3f}  "
                f"sign-acc={np.mean(np.sign(y[te]) == np.sign(p_te)):.1%}   "
                f"[каузальный потолок R²~{CEIL:.2f}]")
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_ylabel("bp / бар")
ax[2].scatter(y[te], p_te, s=14, alpha=.7, color="#1f4e79", edgecolor="none")
lim = [min(y[te].min(), p_te.min()), max(y[te].max(), p_te.max())]
ax[2].plot(lim, lim, "k--", lw=1, label="идеал")
ax[2].set_xlabel("учитель, bp/бар"); ax[2].set_ylabel("прогноз, bp/бар")
ax[2].set_title("Scatter: факт vs прогноз"); ax[2].legend(); ax[2].grid(alpha=.3)
r = y[te] - p_te
ax[3].bar(te_t, r, color=np.where(r >= 0, "#2e8b57", "#c0392b"), width=.03)
ax[3].axhline(0, color="k", lw=.6)
ax[3].set_title(f"Остаток: MAE={np.mean(abs(r)):.2f}  RMSE={np.sqrt(np.mean(r**2)):.2f} bp/бар  "
                f"(std цели {np.std(y[te]):.2f} -> невязка {np.std(r)/np.std(y[te]):.0%} от сигнала)")
ax[3].grid(alpha=.3)
rc = pd.Series(p_te).rolling(40).corr(pd.Series(y[te]))
ax[4].plot(te_t, rc.values, lw=1.6, color="#1f4e79")
ax[4].axhline(rc.mean(), ls="--", color="#e8623d", label=f"среднее {rc.mean():.2f}")
ax[4].axhline(0, color="k", lw=.6); ax[4].set_ylim(-1, 1)
ax[4].set_title("Скользящая корреляция (окно 40 баров) прогноз vs учитель"); ax[4].legend(); ax[4].grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_test.png", dpi=115); plt.close()

imp = pd.Series(mf.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
plt.figure(figsize=(9, 8)); imp.head(20).sort_values().plot.barh(color="#1f4e79")
plt.title("Top-20 фич по gain (LightGBM)"); plt.tight_layout(); plt.savefig("fig_imp.png", dpi=115); plt.close()

w0 = n - 1600; tt = df.dt.iloc[w0:].values
fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
ax[0].plot(tt, c_[w0:], lw=.7, color="#8fa3bf"); ax[0].plot(tt, np.exp(smoothed[w0:]), lw=1.4, color="#e8623d")
ax[0].axvspan(df.dt.iloc[test_i0], df.dt.iloc[-1], color="orange", alpha=.18)
ax[0].set_title("Контекст 1600 баров (оранжевое = тест)"); ax[0].grid(alpha=.3)
ax[1].plot(tt, y[w0:], lw=1.1, color="#1f4e79", label="учитель")
ax[1].plot(te_t, p_te, lw=1.2, color="#e8623d", ls="--", label="модель (тест)")
ax[1].axvspan(df.dt.iloc[test_i0], df.dt.iloc[-1], color="orange", alpha=.18)
ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_ylabel("bp/бар")
plt.tight_layout(); plt.savefig("fig_ctx.png", dpi=115); plt.close()

plt.figure(figsize=(8, 4.5))
plt.plot(lags, cc, lw=1.8, color="#1f4e79"); plt.axvline(0, color="k", lw=.6)
plt.axvline(bl, color="#e8623d", ls="--", label=f"макс при {bl:+d} бар")
plt.xlabel("сдвиг: corr(учитель[i+L], модель[i])"); plt.ylabel("корреляция")
plt.title("Кросс-корреляция учителя и каузальной модели"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_lag.png", dpi=115); plt.close()

plt.figure(figsize=(8, 4.5))
plt.hist(r, bins=30, color="#1f4e79", alpha=.85, label=f"остаток (std={np.std(r):.2f})")
plt.hist(y[te], bins=30, color="#e8623d", alpha=.45, label=f"цель (std={np.std(y[te]):.2f})")
plt.legend(); plt.title("Распределение цели и остатка, bp/бар"); plt.tight_layout()
plt.savefig("fig_resid.png", dpi=115); plt.close()

res.to_csv("metrics.csv", index=False)
print("\nсохранено: fig_test.png fig_imp.png fig_ctx.png fig_resid.png fig_lag.png metrics.csv")
