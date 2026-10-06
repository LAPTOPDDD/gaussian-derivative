#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Воспроизведение реактивной системы 1:1 + исследования (декомпозиция, калибровка, sizing, спред)
import numpy as np
import pandas as pd
import importlib.util
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("rs", "Qwen_python_20260922_mfzwne72j____3333333333.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

FEATURES = rs.FEATURES
RNG02 = np.random.default_rng(42)


def walk_forward_ext(df, states, features, rr=2.0, train_frac=0.4, step=125,
                     max_bars=36, threshold=0.5):
    states = states.dropna(subset=features + ["target"]).copy()
    states = states.sort_values("bar_time").reset_index(drop=True)

    n = len(states)
    start = int(n * 0.2)
    split = int(n * train_frac)
    fold = 0

    model_trades = []
    base_trades = []
    reliability_p, reliability_y = [], []
    fold_aucs, test_end_times = [], []

    while split < n:
        test_end = min(split + step, n)
        train, test = states.iloc[start:split], states.iloc[split:test_end]
        if len(train) < 200 or len(test) < 50:
            split = test_end
            continue

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(train[features].values, train["target"].values)
        test_end_times.append(states.iloc[test_end - 1]["bar_time"])

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
            actual_exit_reason = pos["actual_exit_reason"].iloc[0]
            entry_spread = df["spread_price"].values[int(e_idx)]

            X = pos[features].values
            P = model.predict_proba(X)[:, 1]

            exit_close = df["close"].values[actual_exit_idx]
            exit_spread = df["spread_price"].values[actual_exit_idx]
            base_pnl = direction * (exit_close - entry_price) / R
            base_pnl -= (entry_spread + exit_spread) / R
            base_trades.append({"pnl_r": base_pnl,
                                "exit_time": df.index[actual_exit_idx]})

            exit_idx, exit_reason = None, None
            bars_held = 1
            for k, (_, row) in enumerate(pos.iterrows()):
                p = P[k]
                if row["bars_in_pos"] >= max_bars:
                    exit_idx, exit_reason = int(row["bar_idx"]), "timeout"
                    bars_held = row["bars_in_pos"]
                    break
                if p < threshold:
                    exit_idx, exit_reason = int(row["bar_idx"]), "prob_exit"
                    bars_held = row["bars_in_pos"]
                    break
            if exit_idx is None:
                exit_idx = actual_exit_idx
                exit_reason = "actual"
                bars_held = exit_idx - int(e_idx) + 1

            exit_close = df["close"].values[exit_idx]
            exit_spread = df["spread_price"].values[exit_idx]
            move_r = direction * (exit_close - entry_price) / R
            spread_r = (entry_spread + exit_spread) / R
            pnl_raw = move_r - spread_r

            p_entry = float(P[0])
            size = float(np.clip(0.5 + (p_entry - 0.2) / 0.6, 0.5, 1.5))

            model_trades.append({
                "pnl_r": pnl_raw * size,
                "pnl_r_raw": pnl_raw,
                "move_r": move_r,
                "spread_r": spread_r,
                "size": size,
                "p_entry": p_entry,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "direction": direction,
                "regime_at_entry": float(pos["regime"].iloc[0]),
                "hit_target_actual": int(actual_exit_reason == "target"),
                "fold": fold,
                "target_entry": int(pos["target"].iloc[0]),
                "exit_time": df.index[exit_idx],
            })
        fold += 1
        split = test_end

    trades = pd.DataFrame(model_trades)
    base = pd.DataFrame(base_trades)
    rel = pd.DataFrame({"p": reliability_p, "y": reliability_y})
    return trades, base, rel, fold_aucs


def metrics(trades):
    eq = trades["pnl_r"].cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    gains = trades.loc[trades["pnl_r"] > 0, "pnl_r"].sum()
    losses = -trades.loc[trades["pnl_r"] < 0, "pnl_r"].sum()
    return {"n": int(len(trades)),
            "total_r": float(trades["pnl_r"].sum()),
            "avg_r": float(trades["pnl_r"].mean()),
            "win_rate": float((trades["pnl_r"] > 0).mean()),
            "sharpe": float(trades["pnl_r"].mean() / (trades["pnl_r"].std() + 1e-9) * np.sqrt(len(trades))),
            "max_dd": float(dd.min()),
            "pf": float(gains / losses) if losses > 0 else float("inf")}


def main():
    df = rs.load_data("EURUSD_H1.csv")
    df = rs.add_indicators(df)
    events = rs.simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    states = rs.build_states(df, events, horizon=12)
    print(f"Позиций {len(events)}  Состояний {len(states)}  P(target)={states['target'].mean():.3f}")

    trades, base, rel, aucs = walk_forward_ext(
        df, states, FEATURES, rr=2.0, train_frac=0.4, step=125,
        max_bars=36, threshold=0.5)
    m = metrics(trades)
    b = metrics(base)
    print("-" * 70)
    print(f"ВОСПРОИЗВЕДЕНО: n={m['n']} Total R={m['total_r']:+.2f} Sharpe={m['sharpe']:.2f} "
          f"DD={m['max_dd']:+.2f} PF={m['pf']:.2f} win={m['win_rate']*100:.1f}% "
          f"avg_size={trades['size'].mean():.3f} bars={trades['bars_held'].mean():.1f}")
    print(f"BASELINE:       Total R={b['total_r']:+.2f} Sharpe={b['sharpe']:.2f} DD={b['max_dd']:+.2f}")
    print(f"AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  folds={len(aucs)}")

    # ---- Сверка: фильтр без sizing = сумма raw ----
    raw_sum = trades['pnl_r_raw'].sum()
    print(f"\n[СВЕРКА] фильтр без sizing (sum pnl_r_raw): {raw_sum:+.2f} R  (заявлено -3.59)")

    # ---- STUDY 1: декомпозиция по size-бакетам и причинам выхода ----
    print("\n" + "=" * 70)
    print("STUDY 1a: по бакету размера")
    buckets = pd.cut(trades['size'], [0.4, 0.6, 0.8, 1.0, 1.2, 1.6],
                     labels=['0.5-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5'])
    g = trades.groupby(buckets, observed=True).agg(
        n=('pnl_r', 'size'), r_sized=('pnl_r', 'sum'), r_raw=('pnl_r_raw', 'sum'),
        avg_raw=('pnl_r_raw', 'mean'), avg_p=('p_entry', 'mean'),
        win=('pnl_r', lambda x: (x > 0).mean()))
    print(g.to_string())
    g.to_csv("study1_size_buckets.csv")

    print("\nSTUDY 1b: по причине выхода")
    g2 = trades.groupby('exit_reason').apply(
        lambda d: pd.Series({'n': len(d), 'r_sized': d['pnl_r'].sum(),
                             'r_raw': d['pnl_r_raw'].sum(),
                             'avg_raw': d['pnl_r_raw'].mean(),
                             'avg_bars': d['bars_held'].mean(),
                             'win': (d['pnl_r'] > 0).mean(),
                             'avg_p': d['p_entry'].mean()})) 
    print(g2.to_string())
    g2.to_csv("study1_exit_reason.csv")

    held = trades[trades['bars_held'] > 1]
    bar1 = trades[trades['bars_held'] == 1]
    print(f"\nbar1-закрытия: n={len(bar1)} R_sized={bar1['pnl_r'].sum():+.2f} "
          f"(доля sized {bar1['pnl_r'].sum()/trades['pnl_r'].sum()*100:.0f}%)")
    print(f"удержанные >1 бар: n={len(held)} R_sized={held['pnl_r'].sum():+.2f} R_raw={held['pnl_r_raw'].sum():+.2f}")

    big = trades[trades['size'] >= 1.0]
    small = trades[trades['size'] < 1.0]
    print(f"size>=1.0: n={len(big)} R_sized={big['pnl_r'].sum():+.2f} avg_raw={big['pnl_r_raw'].mean():+.3f}")
    print(f"size<1.0 : n={len(small)} R_sized={small['pnl_r'].sum():+.2f} avg_raw={small['pnl_r_raw'].mean():+.3f}")

    # ---- STUDY 2: калибровка ----
    print("\n" + "=" * 70)
    print("STUDY 2a: калибровка pooled (все состояния теста) по децилям")
    binp = pd.qcut(rel['p'], 10, duplicates='drop')
    cal_all = rel.groupby(binp, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    print(cal_all.round(3).to_string())

    print("\nSTUDY 2b: калибровка на баре ВХОДА (P_entry vs target_entry)")
    bp = pd.qcut(trades['p_entry'], 10, duplicates='drop')
    cal_entry = trades.groupby(bp, observed=True).agg(p=('p_entry', 'mean'),
                                                      y=('target_entry', 'mean'),
                                                      n=('target_entry', 'size'),
                                                      avg_raw_r=('pnl_r_raw', 'mean'),
                                                      size=('size', 'mean'),
                                                      r_sized=('pnl_r', 'sum'))
    print(cal_entry.round(4).to_string())
    cal_entry.to_csv("study2_entry_calibration.csv")

    # ---- STUDY 3: монотонность raw-R по P_entry ----
    print("\n" + "=" * 70)
    print("STUDY 3: монотонность avg raw-R по P_entry (оправдан ли sizing)")
    corr_p = trades['p_entry'].corr(trades['pnl_r_raw'])
    stats = trades.groupby(bp, observed=True).agg(
        n=('pnl_r_raw', 'size'), avg_raw=('pnl_r_raw', 'mean'),
        avg_move=('move_r', 'mean'), avg_cost=('spread_r', 'mean'))
    print(stats.round(4).to_string())
    print(f"corr(P_entry, raw R) = {corr_p:.4f}")
    big = trades[trades['size'] >= 1.2]
    print(f"size>=1.2 (самые 'уверенные'): n={len(big)} R={big['pnl_r'].sum():+.2f} "
          f"avg_raw={big['pnl_r_raw'].mean():+.3f} hit_target={big['hit_target_actual'].mean():.2f}")

    # ---- STUDY 4: спред ----
    print("\n" + "=" * 70)
    print("STUDY 4: чувствительность к спреду")
    print(f"средний спред на входе: {trades['spread_r'].mean()/2*1e5/(trades['move_r'].std() or 1):.3f} pips")
    for mult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        tot = ((trades['move_r'] - mult * trades['spread_r']) * trades['size']).sum()
        print(f"  spread x{mult:.1f}: Total R = {tot:+.2f}")

    # ---- Bootstrap ----
    print("\n" + "=" * 70)
    pnls = trades['pnl_r'].values
    it = 2000
    totals = np.array([RNG02.choice(pnls, size=len(pnls), replace=True).sum() for _ in range(it)])
    print(f"Bootstrap: P(profit>0)={ (totals>0).mean():.3f}  "
          f"5%={np.percentile(totals,5):+.2f}  med={np.median(totals):+.2f}")

    # ---- LOO ----
    tq = trades.copy()
    tq['q'] = pd.to_datetime(tq['exit_time']).dt.to_period('Q')
    print("\nLOO по кварталам:")
    for q in sorted(tq['q'].unique()):
        mm = metrics(tq[tq['q'] != q])
        print(f"  без {q}: n={mm['n']} Total={mm['total_r']:+.2f} Sharpe={mm['sharpe']:.2f}")

    trades.to_csv("study_trades_full.csv", index=False)
    print("\n[OK] сохранено: study_trades_full.csv, study1_*.csv, study2_*.csv")


if __name__ == "__main__":
    main()