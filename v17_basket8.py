#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# v17_basket8 — исправленная версия v17_basket.py.
# Фиксы: get_point по CLOSE + мода по 100 строкам + sanity-fallback,
#        try/except на каждый символ, убран мёртвый build_states,
#        все выходы на диск с суффиксом _8 (старые файлы не трогаем).
# Внутри: прогон 25 инструментов x 2 окна + сохранение трейдов + графики эквити.
import os
import sys
import numpy as np
import pandas as pd
import importlib.util
from collections import Counter

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
OUT = "v17_basket8"


def get_point(path, t0=None, t1=None, sample=100):
    """Размер пункта через число знаков после запятой в CLOSE.
    Мода по строкам внутри окна [t0..t1]; если окно пусто — по первым
    `sample` строкам файла; если и это пусто — фоллбэк по величине цены.
    Точность котировки может меняться по эпохам (XAGUSD: 2 знака в 2009,
    3 знака с ~2015) — потому мода берётся из окна, а не из начала файла."""
    def mode_decs(rows):
        if not rows:
            return None, None
        dec = Counter(rows).most_common(1)[0][0]
        return dec, rows

    t0s = t0.replace("-", ".") if t0 else None
    t1s = t1.replace("-", ".") if t1 else None
    decs_win, last_price = [], None
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            d = parts[0].strip()
            if t0s is not None and (d < t0s or d > t1s):
                continue
            price = parts[5]
            last_price = price
            decs_win.append(len(price.split(".")[1].strip()) if "." in price else 0)

    dec, rows = mode_decs(decs_win)
    if dec is None:
        decs_head = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for i, line in enumerate(f):
                if i >= sample:
                    break
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 6:
                    continue
                price = parts[5]
                last_price = price
                decs_head.append(len(price.split(".")[1].strip()) if "." in price else 0)
        dec, rows = mode_decs(decs_head)
    if dec is None:
        raise ValueError(f"нет читаемых цен в {path}")
    if dec == 0 and last_price is not None:
        probe = float(last_price)
        dec = 5 if probe < 100 else (3 if probe < 10000 else 2)
    return 10 ** -dec


def load_window(path, t0, t1):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["open", "high", "low", "close"])
    df["spread_price"] = df["spread"] * get_point(path, t0, t1)
    idx = df.index
    w0 = idx[np.searchsorted(idx.values, np.datetime64(t0))]
    w1 = idx[np.searchsorted(idx.values, np.datetime64(t1), side="right") - 1]
    lo = w0 - pd.Timedelta(hours=WARM)
    return df.loc[lo:w1].copy(), t0


def run_basket():
    out, trade_rows, errors = [], [], []
    for win_name, (t0, t1) in WINDOWS.items():
        print("=" * 72, flush=True)
        print(f"{win_name}: {t0} .. {t1}", flush=True)
        legs = []
        for s in SYMS:
            try:
                df, tstart = load_window(os.path.join("data", f"{s}_H1.csv"), t0, t1)
                df = v17.rs.add_indicators(df)
                ev = v17.rs.simulate_positions(df, lookback=20, stop_mult=1.5,
                                               rr=2.0, max_bars=48)
                ev = ev[ev["entry_time"] >= pd.Timestamp(tstart)]
                tr, base, _, _ = v17.pipeline(df, ev, exit_rule="ev", size_mode="steep")
                m, mb = v17.metrics(tr), v17.metrics(base)
                for kind, tt in [("model", tr), ("base", base)]:
                    trade_rows.append(pd.DataFrame(
                        {"win": win_name, "sym": s, "kind": kind,
                         "exit_time": tt["exit_time"], "pnl_r": tt["pnl_r"]}))
                legs.append({"win": win_name, "sym": s, "bars": len(df), "pos": len(ev),
                             "n": m["n"], "total": m["total_r"], "sharpe": m["sharpe"],
                             "dd": m["max_dd"], "pf": m["pf"],
                             "bars_held": tr["bars_held"].mean(),
                             "base_total": mb["total_r"]})
                print(f"  {s:7s} n={m['n']:4d} total={m['total_r']:+7.2f} "
                      f"sh={m['sharpe']:+5.2f} dd={m['max_dd']:+6.2f} "
                      f"pf={m['pf']:5.2f} | base={mb['total_r']:+7.2f}", flush=True)
                pd.DataFrame(legs).to_csv(f"{OUT}_{win_name}.csv", index=False)
                pd.concat(trade_rows, ignore_index=True).to_csv(f"{OUT}_trades.csv",
                                                                index=False)
            except Exception as e:
                errors.append(f"{win_name}/{s}: {type(e).__name__}: {e}")
                print(f"  {s:7s} ОШИБКА: {e}", flush=True)
        pos = [l for l in legs if l["total"] > 0]
        print(f"  [итог {win_name}] пары>0: {len(pos)}/{len(legs)}  "
              f"сумма={sum(l['total'] for l in legs):+.2f}", flush=True)
        out.extend(legs)

    tab = pd.DataFrame(out)
    tab.to_csv(f"{OUT}_all.csv", index=False)
    print("\n" + "=" * 72, flush=True)
    print("СВОДКА", flush=True)
    for w in ["W_recent", "W_early"]:
        sub = tab[tab["win"] == w]
        if sub.empty:
            continue
        p = sub[sub["total"] > 0]
        print(f"  {w}: +{len(p)}/{len(sub)}  сумма={sub['total'].sum():+.2f}  "
              f"baseline={sub['base_total'].sum():+.2f}  "
              f"LOO-худшая={sub['total'].sum() - sub['total'].max():+.2f}", flush=True)
    if errors:
        print(f"  ошибок: {len(errors)}", flush=True)
        for e in errors:
            print(f"    {e}", flush=True)
    return tab, pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()


def make_plots(trades):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if trades.empty:
        print("нет трейдов — графики не строим", flush=True)
        return
    for win in ["W_recent", "W_early"]:
        sub = trades[trades["win"] == win]
        if sub.empty:
            continue
        fig, axes = plt.subplots(5, 5, figsize=(22, 18))
        for ax, s in zip(axes.ravel(), SYMS):
            m = sub[(sub["sym"] == s) & (sub["kind"] == "model")].sort_values("exit_time")
            b = sub[(sub["sym"] == s) & (sub["kind"] == "base")].sort_values("exit_time")
            if m.empty:
                ax.set_axis_off()
                continue
            tot = m["pnl_r"].sum()
            color = "steelblue" if tot >= 0 else "firebrick"
            if not b.empty:
                ax.plot(b["exit_time"], b["pnl_r"].cumsum(), color="gray",
                        alpha=0.6, linewidth=1.0, label="base")
            ax.plot(m["exit_time"], m["pnl_r"].cumsum(), color=color,
                    linewidth=1.6, label="v17")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(f"{s}  {tot:+.1f}R", fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)
        axes[0, 0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"v17_basket8 — Equity по инструментам — {win}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{OUT}_equity_{win}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"сохранено {OUT}_equity_{win}.png", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, win in zip(axes, ["W_recent", "W_early"]):
        sub = trades[(trades["win"] == win) & (trades["kind"] == "model")]
        if sub.empty:
            continue
        best = sub.groupby("sym")["pnl_r"].sum().sort_values(ascending=False)
        cols = ["steelblue" if v >= 0 else "firebrick" for v in best.values]
        ax.bar(best.index, best.values, color=cols, alpha=0.85)
        for i, v in enumerate(best.values):
            ax.text(i, v, f"{v:+.1f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.tick_params(axis="x", rotation=80, labelsize=7)
        ax.set_title(f"{win}: total по парам", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("v17_basket8 — прибыль (R) по 25 инструментам", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}_bars.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено {OUT}_bars.png", flush=True)


if __name__ == "__main__":
    tab, trades = run_basket()
    print("\nстроим графики...", flush=True)
    make_plots(trades)
    print("ГОТОВО", flush=True)