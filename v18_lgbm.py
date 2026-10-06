#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v18: простая система на LightGBM (Gradient Boosting) в движке реактивного v17.
# Шаги:
#   1) optimize — сетка гиперпараметров LightGBM на EURUSD (полная история data/),
#      метрика ранжирования = Total R walk-forward трейдов (exit_rule=ev, size_mode=steep).
#   2) equity — прогон лучших параметров, файлы трейдов + график эквити
#      (вся история + увеличение по последним ~2 месяцам).
import os
import sys
import time
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

spec_v17 = importlib.util.spec_from_file_location("v17", "reactive_v17.py")
v17 = importlib.util.module_from_spec(spec_v17)
spec_v17.loader.exec_module(v17)
rs = v17.rs
FEATURES = v17.FEATURES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v17_basket8 import get_point

_P = print
def print(*a, **k):
    k.setdefault("flush", True)
    _P(*a, **k)


WARM = 1500

def load_sym(sym, tail_days=None):
    path = os.path.join("data", f"{sym}_H1.csv")
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["open", "high", "low", "close"])
    df["spread_price"] = df["spread"] * get_point(path)
    if tail_days is not None:
        last = df.index.max()
        t0 = last - pd.Timedelta(days=tail_days)
        idx = df.index
        lo = idx[np.searchsorted(idx.values, np.datetime64(t0))]
        df = df.loc[lo:].copy()
    df = rs.add_indicators(df)
    events = rs.simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    states = rs.build_states(df, events, horizon=12)
    return df, events, states


def make_clf(cfg):
    """Возвращает новую модель: LightGBM или sklearn GBC (сверка с v17)."""
    if cfg.get("kind") == "gbc":
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42)
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=cfg["n_estimators"], num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"], subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1)


def wf18(df, states, features, cfg, rr=2.0, train_frac=0.4, step=125,
         max_bars=36, exit_rule="ev", size_mode="steep", vg=False):
    """Walk-forward как в v17, но классификатор — LightGBM (или GBC для сверки)."""
    states = states.dropna(subset=features + ["target"]).copy()
    states = states.sort_values("bar_time").reset_index(drop=True)

    n = len(states)
    start = int(n * 0.2)
    split = int(n * train_frac)
    fold = 0

    vg_cutoff = None
    if vg:
        entry_regs = states.loc[states["bars_in_pos"] == 1, "regime"].dropna()
        vg_cutoff = float(np.quantile(entry_regs.values, 2.0 / 3.0))

    model_trades, base_trades = [], []
    reliability_p, reliability_y, fold_aucs = [], [], []

    def build():
        return make_clf(cfg)

    while split < n:
        test_end = min(split + step, n)
        train, test = states.iloc[start:split], states.iloc[split:test_end]
        if len(train) < 200 or len(test) < 50:
            split = test_end
            continue

        model = build()
        model.fit(train[features].values, train["target"].values)

        if test["target"].nunique() > 1:
            p_test = model.predict_proba(test[features].values)[:, 1]
            fold_aucs.append(roc_auc_score(test["target"], p_test))
            reliability_p.extend(p_test.tolist())
            reliability_y.extend(test["target"].tolist())

        for e_idx in test["entry_idx"].unique():
            pos = states[states["entry_idx"] == e_idx].sort_values("bar_idx")
            if len(pos) == 0:
                continue

            direction = pos["direction"].iloc[0]
            entry_price = pos["entry_price"].iloc[0]
            R = pos["R"].iloc[0]
            actual_exit_idx = int(pos["actual_exit_idx"].iloc[0])
            entry_spread = df["spread_price"].values[int(e_idx)]

            X = pos[features].values
            P = model.predict_proba(X)[:, 1]

            base_pnl = direction * (df["close"].values[actual_exit_idx] - entry_price) / R
            base_pnl -= (entry_spread + df["spread_price"].values[actual_exit_idx]) / R
            base_trades.append({"pnl_r": base_pnl, "exit_time": df.index[actual_exit_idx]})

            exit_idx, exit_reason, bars_held = None, None, 1
            for k, (_, row) in enumerate(pos.iterrows()):
                p = P[k]
                current_r = row["pnl_r"]
                if exit_rule == "prob":
                    take = p < 0.5
                elif exit_rule == "ev":
                    ev_hold = (rr + 1.0) * p - 1.0
                    take = ev_hold <= current_r
                else:
                    raise ValueError(exit_rule)
                if row["bars_in_pos"] >= max_bars:
                    exit_idx, exit_reason, bars_held = int(row["bar_idx"]), "timeout", row["bars_in_pos"]
                    break
                if take:
                    exit_idx, exit_reason, bars_held = int(row["bar_idx"]), exit_rule, row["bars_in_pos"]
                    break

            if exit_idx is None:
                exit_idx, exit_reason = actual_exit_idx, "actual"
                bars_held = actual_exit_idx - int(e_idx) + 1

            exit_spread = df["spread_price"].values[exit_idx]
            move_r = direction * (df["close"].values[exit_idx] - entry_price) / R
            spread_r = (entry_spread + exit_spread) / R
            pnl_raw = move_r - spread_r

            p_entry = float(P[0])
            size = v17.size_fn(p_entry, size_mode)
            if vg and pos["regime"].iloc[0] > vg_cutoff:
                size = 0.0

            model_trades.append({
                "pnl_r": pnl_raw * size, "pnl_r_raw": pnl_raw,
                "move_r": move_r, "spread_r": spread_r,
                "size": size, "p_entry": p_entry,
                "exit_reason": exit_reason, "bars_held": bars_held,
                "direction": direction, "regime_at_entry": float(pos["regime"].iloc[0]),
                "hit_target_actual": int(pos["actual_exit_reason"].iloc[0] == "target"),
                "fold": fold, "target_entry": int(pos["target"].iloc[0]),
                "exit_time": df.index[exit_idx],
            })
        fold += 1
        split = test_end

    trades = pd.DataFrame(model_trades)
    base = pd.DataFrame(base_trades)
    rel = pd.DataFrame({"p": reliability_p, "y": reliability_y})
    return trades, base, rel, fold_aucs


GRID = []
for num_leaves in [7, 15, 31]:
    for learning_rate in [0.03, 0.05, 0.10]:
        for n_estimators in [150, 300]:
            GRID.append(dict(num_leaves=num_leaves, learning_rate=learning_rate,
                             n_estimators=n_estimators))
GRID.append(dict(num_leaves=63, learning_rate=0.03, n_estimators=150))
GRID.append(dict(num_leaves=3, learning_rate=0.05, n_estimators=300))


def optimize(df, events, states):
    print("=" * 72, flush=True)
    print("OPTIMIZE: EURUSD (data/) полная история, walk-forward, exit=ev, size=steep", flush=True)
    print(f"позиций={len(events)} состояний={len(states)} P(target)={states['target'].mean():.3f}", flush=True)

    rows = []
    best_total = -1e9
    best_cfg = None

    # контрольная точка: стандартный GBC из v17
    cfg0 = dict(kind="gbc")
    t0 = time.time()
    tr, base, rel, aucs = wf18(df, states, FEATURES, cfg0)
    m = v17.metrics(tr)
    rows.append({"kind": "gbc(v17)", "num_leaves": "-", "learning_rate": 0.05,
                 "n_estimators": 300, "n": m["n"], "total_r": round(m["total_r"], 2),
                 "sharpe": round(m["sharpe"], 2), "max_dd": round(m["max_dd"], 2),
                 "pf": round(m["pf"], 2), "win": round(m["win_rate"], 3),
                 "sec": round(time.time() - t0, 1)})
    print(f"[0/{(len(GRID))}] gbc(v17): Total={m['total_r']:+.2f} Sharpe={m['sharpe']:.2f} "
          f"PF={m['pf']:.2f} n={m['n']}", flush=True)

    for i, cfg in enumerate(GRID, 1):
        t0 = time.time()
        tr, base, rel, aucs = wf18(df, states, FEATURES, cfg)
        m = v17.metrics(tr)
        rows.append({"kind": "lgbm", "num_leaves": cfg["num_leaves"],
                     "learning_rate": cfg["learning_rate"],
                     "n_estimators": cfg["n_estimators"], "n": m["n"],
                     "total_r": round(m["total_r"], 2), "sharpe": round(m["sharpe"], 2),
                     "max_dd": round(m["max_dd"], 2), "pf": round(m["pf"], 2),
                     "win": round(m["win_rate"], 3),
                     "auc": round(float(np.mean(aucs)), 3) if aucs else np.nan,
                     "sec": round(time.time() - t0, 1)})
        print(f"[{i}/{len(GRID)}] LGBM nl={cfg['num_leaves']:2d} lr={cfg['learning_rate']:.2f} "
              f"ne={cfg['n_estimators']:3d}: Total={m['total_r']:+.2f} Sharpe={m['sharpe']:.2f} "
              f"PF={m['pf']:.2f} n={m['n']} ({rows[-1]['sec']}s)", flush=True)
        if m["total_r"] > best_total:
            best_total = m["total_r"]
            best_cfg = dict(cfg, kind="lgbm")

    tab = pd.DataFrame(rows)
    tab.to_csv("v18_opt.csv", index=False)
    print("\nТаблица сохранена: v18_opt.csv", flush=True)
    print("\nТОП-5 по Total R:", flush=True)
    print(tab.sort_values("total_r", ascending=False).head(5).to_string(index=False), flush=True)
    return best_cfg


def build_equity(df, events, states, best_cfg):
    print("\n" + "=" * 72, flush=True)
    print(f"EQUITY: лучшие параметры {best_cfg}", flush=True)
    tr, base, rel, aucs = wf18(df, states, FEATURES, best_cfg)
    print(f"полная история: n={len(tr)} Total R={tr['pnl_r'].sum():+.2f}", flush=True)

    last = df.index.max()
    t2m = last - pd.Timedelta(days=62)
    tr_m = tr[tr["exit_time"] >= t2m]
    base_m = base[base["exit_time"] >= t2m]
    mm, mb = v17.metrics(tr_m), v17.metrics(base_m)
    print(f"последние ~2 месяца [{t2m:%Y-%m-%d} .. {last:%Y-%m-%d}]:", flush=True)
    print(f"  модель: n={mm['n']:3d} Total={mm['total_r']:+.2f}R Sharpe={mm['sharpe']:.2f} "
          f"DD={mm['max_dd']:+.2f} PF={mm['pf']:.2f} win={mm['win_rate']*100:.0f}%", flush=True)
    print(f"  baseline: n={mb['n']:3d} Total={mb['total_r']:+.2f}R", flush=True)

    x, y = tr["exit_time"], tr["pnl_r"].cumsum()
    xb, yb = base["exit_time"], base["pnl_r"].cumsum()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    ax = axes[0]
    ax.plot(xb, yb, color="gray", alpha=0.6, lw=1.0, label="baseline")
    ax.plot(x, y, color="steelblue", lw=1.6, label="LGBM")
    ax.axvspan(t2m, last, color="orange", alpha=0.15)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{best_cfg} | полная история: Total={y.iloc[-1]:+.1f}R "
                 f"(оранжевое — последние ~2 мес) | EURUSD H1", fontsize=12)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if not base_m.empty:
        ax2.plot(base_m["exit_time"], base_m["pnl_r"].cumsum(), color="gray",
                 alpha=0.6, lw=1.2, label="baseline")
    ax2.plot(tr_m["exit_time"], tr_m["pnl_r"].cumsum(), color="steelblue", lw=2.0,
             label="LGBM")
    ax2.axhline(0, color="black", lw=0.7)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(f"последние ~2 месяца: Total={mm['total_r']:+.1f}R "
                  f"baseline={mb['total_r']:+.1f}R  n={mm['n']}", fontsize=12)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("v18_equity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("сохранено v18_equity.png", flush=True)

    trf = tr.copy()
    trf["last2m"] = trf["exit_time"] >= t2m
    trf.to_csv("v18_best_trades.csv", index=False)
    print("сохранено v18_best_trades.csv", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    df, events, states = load_sym("EURUSD", tail_days=tail)
    print(f"данные: EURUSD, последние {tail} дней, баров={len(df)}, "
          f"позиций={len(events)} состояний={len(states)}", flush=True)
    if stage in ("all", "optimize"):
        best = optimize(df, events, states)
        print(f"\nлучшие параметры: {best}", flush=True)
        if stage == "optimize":
            build_equity(df, events, states, best)
    if stage == "equity":
        tab = pd.read_csv("v18_opt.csv")
        best_row = tab[tab["kind"] == "lgbm"].sort_values("total_r", ascending=False).iloc[0]
        best = {"kind": "lgbm", "num_leaves": int(best_row["num_leaves"]),
                "learning_rate": float(best_row["learning_rate"]),
                "n_estimators": int(best_row["n_estimators"])}
        build_equity(df, events, states, best)
    print("ГОТОВО", flush=True)