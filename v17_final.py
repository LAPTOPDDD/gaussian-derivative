#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v17 finale: (1) независимая permutation-серия с per-leg разрезанием,
#             (2) полный анализ: baseline портфеля, sweep train_frac, LOO-кварталы,
#                 месячная сводка, концентрация, график v17_report.png
import sys
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("v17", "reactive_v17.py")
v17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v17)

_P = print
def print(*a, **k):
    k.setdefault("flush", True)
    _P(*a, **k)

SYMS = ["EURUSD", "GBPUSD", "USDJPY"]
REAL = 69.70


def legs_df():
    out = {}
    for s in SYMS:
        df, ev = v17.load(s)
        t, b, _, _ = v17.pipeline(df, ev, exit_rule="ev", size_mode="steep")
        t = t.copy(); t["sym"] = s
        b = b.copy(); b["sym"] = s
        out[s] = (df, ev, t, b)
    return out


def perm():
    print("=" * 72, flush=True)
    print("независимая permutation-серия (seed 1234), по-ногам", flush=True)
    data = legs_df()
    real_sums, base_sums = {}, {}
    for s, (df, ev, t, b) in data.items():
        real_sums[s] = float(t["pnl_r"].sum())
        base_sums[s] = float(b["pnl_r"].sum())
        print(f"  {s}: real={real_sums[s]:+.2f}  baseline(hold)={base_sums[s]:+.2f}  "
              f"n={len(t)}/{len(b)}", flush=True)

    rng = np.random.default_rng(1234)
    runs, legs = 12, len(SYMS)
    M = np.zeros((runs, legs))
    for i in range(runs):
        for j, s in enumerate(SYMS):
            df, ev, _, _ = data[s]
            ee = ev.copy()
            flips = rng.integers(0, 2, size=len(ee), dtype=np.int8).astype(float) * 2 - 1
            ee["direction"] = ee["direction"] * flips
            tr, *_ = v17.pipeline(df, ee, exit_rule="ev", size_mode="steep")
            M[i, j] = tr["pnl_r"].sum()
        print(f"  perm {i+1}/{runs}: " +
              "  ".join(f"{s}={M[i,j]:+.1f}" for j, s in enumerate(SYMS)) +
              f"   ПОРТФЕЛЬ={M[i].sum():+.2f}", flush=True)
        np.save("v17_perm2.npy", M[:i+1])

    tot = M.sum(axis=1)
    p = float((tot >= REAL).mean())
    print(f"\n  реальный портфель = {REAL:+.2f}", flush=True)
    print(f"  P(perm >= real) = {p:.3f}   (по {len(tot)} свежим прогонам)", flush=True)
    print(f"  null портфель: 5%={np.percentile(tot,5):+.2f} med={np.median(tot):+.2f} "
          f"95%={np.percentile(tot,95):+.2f} min={tot.min():+.2f} max={tot.max():+.2f}", flush=True)
    print("  per-leg real vs per-leg null (эмпирический p):", flush=True)
    for j, s in enumerate(SYMS):
        r = real_sums[s]
        nn = M[:, j]
        print(f"    {s}: real={r:+.2f}  null_med={np.median(nn):+.2f}  "
              f"p={(nn >= r).mean():.3f}  (p>null: {(nn>=r).sum()}/{runs})", flush=True)
    np.savez("v17_perms_full.npz", M=M, real=np.array(list(real_sums.values())),
             base=np.array(list(base_sums.values())), syms=np.array(SYMS), real_port=REAL)
    print("  сохранено: v17_perm2.npy, v17_perms_full.npz", flush=True)


def analysis():
    print("=" * 72, flush=True)
    print("ФИНАЛЬНЫЙ АНАЛИЗ v17 (EV_STEEP, одна конфигурация)", flush=True)
    data = legs_df()

    rows = []
    all_t, all_b = [], []
    for s, (df, ev, t, b) in data.items():
        m = v17.metrics(t)
        mb = v17.metrics(b)
        rows.append({"sym": s, "cfg": "STEEP", "n": m["n"], "total": m["total_r"],
                     "sharpe": m["sharpe"], "dd": m["max_dd"], "pf": m["pf"]})
        rows.append({"sym": s, "cfg": "baseline_hold", "n": mb["n"],
                     "total": mb["total_r"], "sharpe": mb["sharpe"],
                     "dd": mb["max_dd"], "pf": mb["pf"]})
        all_t.append(t); all_b.append(b)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False), flush=True)
    tab.to_csv("v17_final_table.csv", index=False)

    pt = pd.concat(all_t, ignore_index=True)
    pb = pd.concat(all_b, ignore_index=True)
    mt, mb = v17.metrics(pt), v17.metrics(pb)
    print(f"\nПОРТФЕЛЬ STEEP: n={mt['n']} Total={mt['total_r']:+.2f} Sharpe={mt['sharpe']:.2f} "
          f"DD={mt['max_dd']:+.2f} PF={mt['pf']:.2f}", flush=True)
    print(f"ПОРТФЕЛЬ baseline-hold: n={mb['n']} Total={mb['total_r']:+.2f} "
          f"Sharpe={mb['sharpe']:.2f} DD={mb['max_dd']:+.2f}", flush=True)
    top10 = pt["pnl_r"].nlargest(10).sum() / pt["pnl_r"].sum() * 100
    top30 = pt["pnl_r"].nlargest(30).sum() / pt["pnl_r"].sum() * 100
    big = pt[pt["size"] >= 1.0]
    print(f"концентрация: top10={top10:.0f}%  top30={top30:.0f}%  size>=1.0: "
          f"n={len(big)} r={big['pnl_r'].sum():+.2f} ({big['pnl_r'].sum()/pt['pnl_r'].sum()*100:.0f}%)",
          flush=True)
    print(f"среднее: bars={pt['bars_held'].mean():.1f} size={pt['size'].mean():.3f} "
          f"win={(pt['pnl_r']>0).mean()*100:.1f}%", flush=True)

    print("\nsweep train_frac (STEEP, все 3 ноги):", flush=True)
    srows = []
    for tf in [0.3, 0.4, 0.5]:
        tot = 0.0
        for s in SYMS:
            df, ev = v17.load(s)
            t, *_ = v17.pipeline(df, ev, exit_rule="ev", size_mode="steep", train_frac=tf)
            tot += t["pnl_r"].sum()
        srows.append({"train_frac": tf, "portfolio_total": tot})
        print(f"    train_frac={tf}: портфель Total={tot:+.2f}", flush=True)
    pd.DataFrame(srows).to_csv("v17_sweep_trainfrac.csv", index=False)

    print("\nLOO по кварталам (портфель):", flush=True)
    pt2 = pt.copy()
    pt2["q"] = pd.to_datetime(pt2["exit_time"]).dt.to_period("Q")
    qtab = []
    for q in sorted(pt2["q"].unique()):
        mm = v17.metrics(pt2[pt2["q"] != q])
        qtab.append({"without": str(q), "total_r": mm["total_r"], "sharpe": mm["sharpe"]})
        print(f"    без {q}: Total={mm['total_r']:+.2f} Sharpe={mm['sharpe']:.2f}", flush=True)
    pd.DataFrame(qtab).to_csv("v17_loo_quarters.csv", index=False)

    print("\nмесячная сводка (портфель):", flush=True)
    pt2["m"] = pd.to_datetime(pt2["exit_time"]).dt.to_period("M")
    mon = pt2.groupby("m").agg(total=("pnl_r", "sum"), n=("pnl_r", "size"))
    print(mon.round(2).to_string(), flush=True)
    mon.to_csv("v17_monthly.csv")

    print("\nэкспорт и график:", flush=True)
    pt.to_csv("v17_portfolio_tagged_trades.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        ax = axes[0, 0]
        ax.plot(pb["exit_time"], pb["pnl_r"].cumsum(), color="gray", alpha=0.7,
                linewidth=1.5, label="baseline hold")
        ax.plot(pt["exit_time"], pt["pnl_r"].cumsum(), color="steelblue",
                linewidth=2.2, label="v17 EV_STEEP portfolio")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_title("Portfolio equity (R)")
        ax = axes[0, 1]
        eq = pt["pnl_r"].cumsum().values
        dd = eq - np.maximum.accumulate(eq)
        ax.plot(pt["exit_time"], dd, color="firebrick", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.7); ax.grid(True, alpha=0.3)
        ax.set_title("Portfolio drawdown (R)")
        ax = axes[1, 0]
        ttl = pt.groupby("sym")["pnl_r"].sum()
        ax.bar(ttl.index, ttl.values, color=["steelblue"] * len(ttl), alpha=0.85)
        for i, v in enumerate(ttl.values):
            ax.text(i, v, f"{v:+.1f}", ha="center", va="bottom")
        ax.axhline(0, color="black", linewidth=0.7); ax.grid(True, alpha=0.3)
        ax.set_title("Total R по ногам")
        ax = axes[1, 1]
        vals = mon["total"].values
        cols = ["steelblue" if v > 0 else "firebrick" for v in vals]
        ax.bar(mon.index.astype(str), vals, color=cols, alpha=0.85)
        ax.tick_params(axis="x", rotation=60)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title("Monthly portfolio PnL (R)")
        plt.tight_layout()
        plt.savefig("v17_report.png", dpi=130, bbox_inches="tight")
        print("  сохранено: v17_report.png", flush=True)
    except Exception as e:
        print(f"  график не построен: {e}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analysis"
    {"perm": perm, "analysis": analysis}[stage]()