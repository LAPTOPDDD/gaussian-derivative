#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v17 basket: 25 инструментов (данные data/) x 2 независимых окна истории.
# Одна конфигурация EV_STEEP, честный спред по размеру пункта символа.
import os
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

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD",
        "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "XAUUSD", "XAGUSD"]

WINDOWS = {
    "W_recent": ("2024-12-27", "2026-09-09"),
    "W_early": ("2020-01-01", "2022-06-30"),
}
WARM = 1500


def get_point(path):
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        line = f.readline()
    price = line.split("\t")[3].strip()
    if "." in price:
        dec = len(price.split(".")[1])
    else:
        dec = 0
    return 10 ** -dec


def load_window(path, t0, t1):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["open", "high", "low", "close"])
    point = get_point(path)
    df["spread_price"] = df["spread"] * point
    idx = df.index
    w0 = idx[np.searchsorted(idx.values, np.datetime64(t0))]
    w1 = idx[np.searchsorted(idx.values, np.datetime64(t1), side="right") - 1]
    lo = w0 - pd.Timedelta(hours=WARM)
    return df.loc[lo:w1].copy(), t0


def run_basket():
    out = []
    trade_rows = []
    for win_name, (t0, t1) in WINDOWS.items():
        print("=" * 72, flush=True)
        print(f"{win_name}: {t0} .. {t1}  (спред по пункту каждого символа)", flush=True)
        legs = []
        for s in SYMS:
            path = os.path.join("data", f"{s}_H1.csv")
            df, tstart = load_window(path, t0, t1)
            df = v17.rs.add_indicators(df)
            ev = v17.rs.simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
            ev = ev[ev["entry_time"] >= pd.Timestamp(tstart)]
            st = v17.rs.build_states(df, ev, horizon=12)
            tr, base, rel, aucs = v17.pipeline(df, ev, exit_rule="ev", size_mode="steep")
            m = v17.metrics(tr)
            mb = v17.metrics(base)

            for kind, tt in [("model", tr), ("base", base)]:
                tmp = pd.DataFrame({"win": win_name, "sym": s, "kind": kind,
                                    "exit_time": tt["exit_time"], "pnl_r": tt["pnl_r"]})
                trade_rows.append(tmp)

            leg = {"win": win_name, "sym": s, "bars": len(df), "pos": len(ev),
                   "n": m["n"], "total": m["total_r"], "sharpe": m["sharpe"],
                   "dd": m["max_dd"], "pf": m["pf"], "bars_held": tr["bars_held"].mean(),
                   "base_total": mb["total_r"]}
            legs.append(leg)
            print(f"  {s:7s} n={m['n']:4d} total={m['total_r']:+7.2f} "
                  f"sh={m['sharpe']:+5.2f} dd={m['max_dd']:+6.2f} "
                  f"pf={m['pf']:5.2f} | base={mb['total_r']:+7.2f}", flush=True)
            pd.DataFrame(legs).to_csv(f"v17_basket_{win_name}.csv", index=False)
            pd.concat(trade_rows, ignore_index=True).to_csv("v17_basket_trades.csv", index=False)
        print(f"  [итог {win_name}] пары>0: {sum(l['total']>0 for l in legs)}/{len(legs)}  "
              f"сумма={sum(l['total'] for l in legs):+.2f}  "
              f"медиана(+пары)={np.median([l['total'] for l in legs if l['total']>0]):+.2f}",
              flush=True)
        out.extend(legs)

    print("\n" + "=" * 72, flush=True)
    print("СВОДКА ПО ДВУМ ОКНАМ", flush=True)
    tab = pd.DataFrame(out)
    for w in ["W_recent", "W_early"]:
        sub = tab[tab["win"] == w]
        p = sub[sub["total"] > 0]
        print(f"  {w}: положительных {len(p)}/{len(sub)}, сумма={sub['total'].sum():+.2f}, "
              f"медиана по положительным={p['total'].median():+.2f}", flush=True)
        print("  суммарный baseline(hold)=", f"{sub['base_total'].sum():+.2f}", flush=True)
        loo = sub["total"].sum() - sub["total"]
        print("  LOO-инструмент: худшая сумма без одной пары=", f"{loo.min():+.2f}", flush=True)
    tab.to_csv("v17_basket_all.csv", index=False)


if __name__ == "__main__":
    run_basket()