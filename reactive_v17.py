#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v17: EV-выход (decision-theoretic) + sizing-варианты + vol-gate, портфель инструментов.
# Одна и та же конфигурация (rr=2.0, step=125, lookback=20, stop 1.5ATR, горизонт 12,
# max_bars=36, train_frac=0.4, GBC 300/3/0.05) НА ВСЕ СИМВОЛЫ. Без подбора под пару.
import sys
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("rs", "Qwen_python_20260922_mfzwne72j____3333333333.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
FEATURES = rs.FEATURES

_P = print
def print(*a, **k):
    k.setdefault("flush", True)
    _P(*a, **k)


def size_fn(p, mode):
    if mode == "orig":
        return float(np.clip(0.5 + (p - 0.2) / 0.6, 0.5, 1.5))
    if mode == "cap12":
        return float(np.clip(0.5 + (p - 0.2) / 0.6, 0.5, 1.2))
    if mode == "steep":
        return float(np.clip(0.5 + (p - 0.2) / 0.4, 0.5, 1.5))
    raise ValueError(mode)


def wf17(df, states, features, rr=2.0, train_frac=0.4, step=125,
         max_bars=36, exit_rule="prob", threshold=0.5,
         size_mode="orig", vg=False):
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

    while split < n:
        test_end = min(split + step, n)
        train, test = states.iloc[start:split], states.iloc[split:test_end]
        if len(train) < 200 or len(test) < 50:
            split = test_end
            continue

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42)
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
                    take = p < threshold
                elif exit_rule == "ev":
                    ev_hold = (rr + 1.0) * p - 1.0
                    take = ev_hold <= current_r
                else:
                    raise ValueError(exit_rule)
                if row["bars_in_pos"] >= max_bars:
                    exit_idx, exit_reason, bars_held = int(row["bar_idx"]), "timeout", row["bars_in_pos"]
                    break
                if take:
                    exit_idx, exit_reason, bars_held = int(row["bar_idx"]), "ev_exit" if exit_rule == "ev" else "prob_exit", row["bars_in_pos"]
                    break

            if exit_idx is None:
                exit_idx, exit_reason = actual_exit_idx, "actual"
                bars_held = actual_exit_idx - int(e_idx) + 1

            exit_spread = df["spread_price"].values[exit_idx]
            move_r = direction * (df["close"].values[exit_idx] - entry_price) / R
            spread_r = (entry_spread + exit_spread) / R
            pnl_raw = move_r - spread_r

            p_entry = float(P[0])
            size = size_fn(p_entry, size_mode)
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


def metrics(trades):
    if len(trades) == 0:
        return dict(n=0, total_r=0.0, win_rate=0.0, sharpe=0.0, max_dd=0.0, pf=0.0)
    eq = trades["pnl_r"].cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    gains = trades.loc[trades["pnl_r"] > 0, "pnl_r"].sum()
    losses = -trades.loc[trades["pnl_r"] < 0, "pnl_r"].sum()
    return {
        "n": int(len(trades)),
        "total_r": float(trades["pnl_r"].sum()),
        "win_rate": float((trades["pnl_r"] > 0).mean()),
        "sharpe": float(trades["pnl_r"].mean() / (trades["pnl_r"].std() + 1e-9) * np.sqrt(len(trades))),
        "max_dd": float(dd.min()),
        "pf": float(gains / losses) if losses > 0 else float("inf"),
    }


def summarize(tag, trades, base=None):
    m = metrics(trades)
    taken = trades[trades["size"] > 0]
    top10 = trades["pnl_r"].nlargest(10).sum()
    line = (f"[{tag}] n={taken['size'].count()} (взято {len(trades)}) Total R={m['total_r']:+.2f} "
            f"Sharpe={m['sharpe']:.2f} DD={m['max_dd']:+.2f} PF={m['pf']:.2f} "
            f"win={m['win_rate']*100:.1f}% avg_size={trades['size'].mean():.3f} "
            f"bars={trades['bars_held'].mean():.1f}  top10={top10/trades['pnl_r'].sum()*100:.0f}% equity")
    print(line, flush=True)
    if base is not None:
        print(f"    baseline: Total R={metrics(base)['total_r']:+.2f}", flush=True)
    return m


def pipeline(df, events, exit_rule="prob", threshold=0.5, size_mode="orig", vg=False,
             train_frac=0.4):
    states = rs.build_states(df, events, horizon=12)
    trades, base, rel, aucs = wf17(
        df, states, FEATURES, rr=2.0, train_frac=train_frac, step=125,
        max_bars=36, exit_rule=exit_rule, threshold=threshold,
        size_mode=size_mode, vg=vg)
    return trades, base, rel, aucs


def load(sym):
    df = rs.load_data(f"{sym}_H1.csv")
    df = rs.add_indicators(df)
    events = rs.simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    return df, events


def phase1():
    print("=" * 72, flush=True)
    print("PHASE 1  EURUSD H1: сравнение конфигураций выхода/sizing", flush=True)
    df, events = load("EURUSD")
    states = rs.build_states(df, events, horizon=12)
    print(f"позиций={len(events)} состояний={len(states)} P(target)={states['target'].mean():.3f}", flush=True)

    base, _, _, _ = pipeline(df, events)
    print("  ШАГ 1 control BASE (prob@0.5, sizing orig)", flush=True)
    summarize("BASE   ", base)

    cfg = [
        ("EV       ", dict(exit_rule="ev", size_mode="orig")),
        ("EV_CAP   ", dict(exit_rule="ev", size_mode="cap12")),
        ("EV_STEEP ", dict(exit_rule="ev", size_mode="steep")),
        ("EV_VG    ", dict(exit_rule="ev", size_mode="orig", vg=True)),
    ]
    res = {}
    for tag, kw in cfg:
        print(f"  шаг: {tag}", flush=True)
        t, b, r, a = pipeline(df, events, **kw)
        summarize(tag, t, b)
        res[tag] = (t, b, r, a)
    trades = {k: v[0] for k, v in res.items()}
    pd.concat(trades, names=["cfg"]).to_csv("v17_phase1_trades.csv")
    print("сравнение размеров принимаемых (size>0):", flush=True)
    for tag, kw in cfg:
        t, _, _, _ = pipeline(df, events, **kw)
        print(f"    {tag}: n={len(t)} size0={int((t['size']==0).sum())}", flush=True)


def phase2():
    print("=" * 72, flush=True)
    print("PHASE 2  портфель: BASE / EV / EV_STEEP по EUR/GBP/JPY (одна конфигурация)", flush=True)
    cfgs = [("BASE", dict(exit_rule="prob", size_mode="orig")),
            ("EV", dict(exit_rule="ev", size_mode="orig")),
            ("EV_STEEP", dict(exit_rule="ev", size_mode="steep"))]
    rows = []
    portfolios = {c: [] for c, _ in cfgs}
    for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
        df, events = load(sym)
        for tag, kw in cfgs:
            t, _, _, _ = pipeline(df, events, **kw)
            m = metrics(t)
            rows.append({"sym": sym, "cfg": tag, "n": m["n"], "total_r": m["total_r"],
                         "sharpe": m["sharpe"], "max_dd": m["max_dd"], "pf": m["pf"]})
            tt = t.copy()
            tt["sym"] = sym
            portfolios[tag].append(tt)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False), flush=True)
    tab.to_csv("v17_phase2_table.csv", index=False)

    for tag, sets in portfolios.items():
        all_t = pd.concat(sets, ignore_index=True)
        mm = metrics(all_t)
        loo = []
        for s in ["EURUSD", "GBPUSD", "USDJPY"]:
            rest = all_t[all_t["sym"] != s]
            loo.append((f"-{s}", metrics(rest)["total_r"]))
        print(f"ПОРТФЕЛЬ {tag}: Total R={mm['total_r']:+.2f} Sharpe={mm['sharpe']:.2f} "
              f"DD={mm['max_dd']:+.2f} n={mm['n']}", flush=True)
        for k, v in loo:
            print(f"    leave {k:8s}: Total={v:+.2f}", flush=True)


def phase3():
    print("=" * 72, flush=True)
    print("PHASE 3  аудит EV_STEEP как ПОРТФЕЛЯ (EUR+GBP+JPY, одна конфигурация)", flush=True)
    cfgs = [("BASE", dict(exit_rule="prob", size_mode="orig")),
            ("EV_STEEP", dict(exit_rule="ev", size_mode="steep"))]
    syms = ["EURUSD", "GBPUSD", "USDJPY"]
    dfs, evs = {}, {}
    for s in syms:
        dfs[s], evs[s] = load(s)

    for tag, kw in cfgs:
        legs = {}
        for s in syms:
            t, _, _, _ = pipeline(dfs[s], evs[s], **kw)
            legs[s] = t
        all_t = pd.concat(legs, ignore_index=True)
        mm = metrics(all_t)
        print(f"\n[{tag}] ПОРТФЕЛЬ n={mm['n']} Total R={mm['total_r']:+.2f} "
              f"Sharpe={mm['sharpe']:.2f} DD={mm['max_dd']:+.2f} PF={mm['pf']:.2f} "
              f"win={mm['win_rate']*100:.1f}%", flush=True)
        for s in syms:
            m = metrics(legs[s])
            l = legs[s]
            tot_s = l["pnl_r"].sum()
            top10 = (l["pnl_r"].nlargest(10).sum() / tot_s * 100) if tot_s > 0 else float("nan")
            top10s = f"{top10:.0f}%" if np.isfinite(top10) else "neg"
            print(f"    {s}: Total={m['total_r']:+.2f} Sharpe={m['sharpe']:.2f} "
                  f"top10={top10s} equity, avg_size={l['size'].mean():.3f}", flush=True)
        if tag == "EV_STEEP":
            all_t.to_csv("v17_portfolio_steeper_trades.csv", index=False)

    t = pd.concat([pipeline(dfs[s], evs[s], exit_rule="ev", size_mode="steep")[0] for s in syms],
                  ignore_index=True)
    print("спред-чувствительность портфеля EV_STEEP:", flush=True)
    for mult in [0.0, 1.0, 2.0, 3.0, 5.0]:
        print(f"    x{mult:.0f}: Total={((t['move_r'] - mult*t['spread_r'])*t['size']).sum():.2f}", flush=True)

    print("bootstrap портфеля, 2000:", flush=True)
    pnls = t["pnl_r"].values
    rng = np.random.default_rng(42)
    totals = np.array([rng.choice(pnls, size=len(pnls), replace=True).sum() for _ in range(2000)])
    print(f"    P(profit>0)={(totals>0).mean():.3f}  5%={np.percentile(totals,5):+.2f} "
          f"med={np.median(totals):+.2f}", flush=True)

    print("\npermutation портфеля (flip направлений во всех 3 ногах, 12 прогонов):", flush=True)
    rng = np.random.default_rng(123)
    null = []
    for it in range(12):
        pool = []
        for s in syms:
            ee = evs[s].copy()
            flips = rng.integers(0, 2, size=len(ee), dtype=np.int8).astype(float) * 2 - 1
            ee["direction"] = ee["direction"] * flips
            tr, *_ = pipeline(dfs[s], ee, exit_rule="ev", size_mode="steep")
            pool.append(tr["pnl_r"].sum())
        null.append(sum(pool))
        print(f"    perm-run {it+1}/12: Total={null[-1]:+.2f}", flush=True)
    null = np.array(null)
    real = t["pnl_r"].sum()
    print(f"    реальный портфель EV_STEEP = {real:+.2f}", flush=True)
    print(f"    P(perm >= real) = {(null >= real).mean():.3f}", flush=True)
    print(f"    null: 5%={np.percentile(null,5):+.2f} med={np.median(null):+.2f} "
          f"95%={np.percentile(null,95):+.2f} max={null.max():+.2f}", flush=True)
    np.save("v17_perm_totals.npy", null)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "phase1"
    {"phase1": phase1, "phase2": phase2, "phase3": phase3}[which]()