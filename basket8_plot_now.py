# -*- coding: utf-8 -*-
# Графики по уже готовым данным W_recent из v17_basket8_trades.csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "v17_basket8"
SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD",
        "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "XAUUSD", "XAGUSD"]

t = pd.read_csv(f"{OUT}_trades.csv")
win = "W_recent"
sub = t[t["win"] == win]
missing = [s for s in SYMS if s not in sub.sym.unique()]
if missing:
    raise SystemExit(f"нет данных для: {missing}")

fig, axes = plt.subplots(5, 5, figsize=(22, 18))
for ax, s in zip(axes.ravel(), SYMS):
    m = sub[(sub.sym == s) & (sub.kind == "model")].sort_values("exit_time")
    b = sub[(sub.sym == s) & (sub.kind == "base")].sort_values("exit_time")
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
fig.suptitle(f"v17_basket8 — Equity по инструментам — {win} (wifi-прерван, неполный W_early)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}_equity_{win}.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"сохранено {OUT}_equity_{win}.png")

fig2, ax2 = plt.subplots(figsize=(16, 8))
best = sub[sub.kind == "model"].groupby("sym")["pnl_r"].sum().sort_values(ascending=False)
cols = ["steelblue" if v >= 0 else "firebrick" for v in best.values]
ax2.bar(best.index, best.values, color=cols, alpha=0.85)
for i, v in enumerate(best.values):
    ax2.text(i, v, f"{v:+.1f}", ha="center", va="bottom", fontsize=7)
ax2.axhline(0, color="black", linewidth=0.7)
ax2.tick_params(axis="x", rotation=80, labelsize=7)
ax2.set_title(f"{win}: total по парам (сумма={best.sum():+.1f}R, "
              f"в плюсе {int((best > 0).sum())}/{len(best)})", fontsize=12)
ax2.grid(True, alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig(f"{OUT}_bars_{win}.png", dpi=120, bbox_inches="tight")
plt.close(fig2)
print(f"сохранено {OUT}_bars_{win}.png")
print("ГОТОВО")