#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# STUDY 5: анатомия побед + H4-перенос + перестановочный тест (ужато)
import sys
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("rs", "Qwen_python_20260922_mfzwne72j____3333333333.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
FEATURES = rs.FEATURES
from reactive_study import walk_forward_ext  # noqa

P = print
def print(*a, **k):
    k.setdefault("flush", True)
    P(*a, **k)

def run_pipeline(df, events):
    states = rs.build_states(df, events, horizon=12)
    trades, base, rel, aucs = walk_forward_ext(
        df, states, FEATURES, rr=2.0, train_frac=0.4, step=125,
        max_bars=36, threshold=0.5)
    return trades

def main():
    df = rs.load_data("EURUSD_H1.csv")
    df = rs.add_indicators(df)
    events = rs.simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    trades = run_pipeline(df, events)

    print("=" * 70)
    print("STUDY 5a: анатомия 8 'actual'-побед и 20 size>=1.0", flush=True)
    w = trades[trades['exit_reason'] == 'actual'].sort_values('pnl_r', ascending=False)
    print(w[['exit_time', 'direction', 'p_entry', 'size', 'pnl_r_raw', 'pnl_r',
             'bars_held', 'hit_target_actual']].to_string(), flush=True)
    print(f"\n8 actual: Total={w['pnl_r'].sum():+.2f} "
          f"({w['pnl_r'].sum()/trades['pnl_r'].sum()*100:.0f}% equity)", flush=True)

    big = trades[trades['size'] >= 1.0].sort_values('pnl_r', ascending=False)
    print("\n20 size>=1.0:", flush=True)
    print(big[['exit_time', 'direction', 'p_entry', 'size', 'pnl_r_raw', 'pnl_r',
               'exit_reason', 'hit_target_actual']].to_string(), flush=True)

    no8 = trades[trades['exit_reason'] != 'actual']
    print(f"\nБез 8 actual:  Total R = {no8['pnl_r'].sum():+.2f}", flush=True)
    print(f"Без size>=1.0:  Total R = {trades[trades['size'] < 1.0]['pnl_r'].sum():+.2f}", flush=True)
    print(f"Без size>=1.2:  Total R = {trades[trades['size'] < 1.2]['pnl_r'].sum():+.2f}", flush=True)

    # ============================================================
    print("\n" + "=" * 70, flush=True)
    print("STUDY 5c: перенос на EURUSD H4 (те же гиперпараметры)", flush=True)
    df4 = rs.load_data("EURUSD_H4_202412270000_202609092000.csv")
    print(f"H4 баров: {len(df4)}", flush=True)
    df4 = rs.add_indicators(df4)
    ev4 = rs.simulate_positions(df4, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    st4 = rs.build_states(df4, ev4, horizon=12)
    print(f"H4 позиций: {len(ev4)}  состояний: {len(st4)}  P(target)={st4['target'].mean():.3f}", flush=True)
    t4 = run_pipeline(df4, ev4)
    m4 = t4['pnl_r']
    eq = m4.cumsum().values
    dd = (eq - np.maximum.accumulate(eq)).min()
    print(f"H4: n={len(t4)} Total R={m4.sum():+.2f} Sharpe="
          f"{m4.mean()/(m4.std()+1e-9)*np.sqrt(len(t4)):.2f} DD={dd:+.2f} "
          f"win={(m4>0).mean()*100:.1f}% avg_size={t4['size'].mean():.2f} "
          f"bars={t4['bars_held'].mean():.1f}", flush=True)
    print(t4['exit_reason'].value_counts().to_string(), flush=True)
    print("по size-бакетам:", flush=True)
    tg = t4.groupby(pd.cut(t4['size'], [0.4, 0.6, 0.8, 1.0, 1.2, 1.6],
                           labels=['0.5-0.6', '0.6-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5']),
                    observed=True)['pnl_r'].agg(['size', 'sum', 'mean'])
    print(tg.to_string(), flush=True)
    t4.to_csv("study_h4_trades.csv", index=False)

    # ============================================================
    print("\n" + "=" * 70, flush=True)
    print("STUDY 5b: перестановочный тест (flip направлений, 15 прогонов)",
          flush=True)
    rng = np.random.default_rng(123)
    totals = []
    ev = events.copy()
    for it in range(15):
        ee = ev.copy()
        flips = rng.integers(0, 2, size=len(ee), dtype=np.int8).astype(float) * 2 - 1
        ee['direction'] = ee['direction'] * flips
        t = run_pipeline(df, ee)
        totals.append(t['pnl_r'].sum())
        print(f"  flip-run {it+1}/15: Total={totals[-1]:+.2f}", flush=True)
    totals = np.array(totals)
    print(f"\nP(Total > +20.12) при случайных направлениях = {(totals > 20.12).mean():.3f}",
          flush=True)
    print(f"шuffle: 5%={np.percentile(totals,5):+.2f} med={np.median(totals):+.2f} "
          f"95%={np.percentile(totals,95):+.2f} max={totals.max():+.2f}", flush=True)
    np.save("study_shuffle_totals.npy", totals)

if __name__ == "__main__":
    main()