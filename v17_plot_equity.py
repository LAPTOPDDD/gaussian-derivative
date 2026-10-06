# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD",
        "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "XAUUSD", "XAGUSD"]

trades = pd.read_csv("v17_basket_trades.csv", parse_dates=["exit_time"])

for win in ["W_recent", "W_early"]:
    sub = trades[trades["win"] == win]
    fig, axes = plt.subplots(5, 5, figsize=(22, 18))
    for ax, s in zip(axes.ravel(), SYMS):
        m = sub[(sub["sym"] == s) & (sub["kind"] == "model")].sort_values("exit_time")
        b = sub[(sub["sym"] == s) & (sub["kind"] == "base")].sort_values("exit_time")
        tot = m["pnl_r"].sum()
        color = "steelblue" if tot >= 0 else "firebrick"
        ax.plot(b["exit_time"], b["pnl_r"].cumsum(), color="gray", alpha=0.6,
                linewidth=1.0, label="base")
        ax.plot(m["exit_time"], m["pnl_r"].cumsum(), color=color, linewidth=1.6, label="v17")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{s}  {tot:+.1f}R", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Equity по инструментам — {win} (v17 EV_STEEP, одна конфигурация)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"v17_basket_equity_{win}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено v17_basket_equity_{win}.png")

top = 9
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
for ax, win in zip(axes, ["W_recent", "W_early"]):
    sub = trades[trades["win"] == win]
    best = (sub[sub["kind"] == "model"].groupby("sym")["pnl_r"].sum()
            .sort_values(ascending=False))
    cols = ["steelblue" if v >= 0 else "firebrick" for v in best.values]
    ax.bar(best.index, best.values, color=cols, alpha=0.85)
    for i, v in enumerate(best.values):
        ax.text(i, v, f"{v:+.1f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.tick_params(axis="x", rotation=80, labelsize=7)
    ax.set_title(f"{win}: total по парам (слева-направо по убыванию)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("Итоговая прибыль (R) по 25 инструментам", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("v17_basket_bars.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("сохранено v17_basket_bars.png")