#!/usr/bin/env python3
# ============================================================
# EURUSD H1 — ФИНАЛЬНАЯ КОНФИГУРАЦИЯ
# rr=2.0, step=125, prob exit@0.5, sizing по P(entry)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
RNG = np.random.default_rng(42)


# ============================================================
# ДАННЫЕ
# ============================================================
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(subset=["open","high","low","close"])
    df["spread_price"] = df["spread"] * 1e-5
    return df


def add_indicators(df):
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr50"] = tr.rolling(50).mean()
    df["regime"] = df["atr14"] / df["atr50"]

    for n in (1, 2, 3, 5, 10):
        df[f"ret_{n}"] = df["close"].pct_change(n)

    for n in (20, 50, 100):
        hh = df["high"].rolling(n).max()
        ll = df["low"].rolling(n).min()
        rng = (hh - ll).replace(0, np.nan)
        df[f"pos_in_range_{n}"] = (df["close"] - ll) / rng

    direction = np.sign(df["close"].diff())
    df["streak"] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    df["streak"] *= direction

    df["tickvol_ma50"] = df["tickvol"].rolling(50).mean()
    df["tickvol_rel"] = df["tickvol"] / df["tickvol_ma50"]
    df["spread_ma50"] = df["spread"].rolling(50).mean()
    df["spread_rel"] = df["spread"] / df["spread_ma50"]

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    return df


# ============================================================
# СИМУЛЯЦИЯ
# ============================================================
def simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48):
    events = []
    highs, lows = df["high"].values, df["low"].values
    closes, opens = df["close"].values, df["open"].values
    atr = df["atr14"].values

    i = lookback
    while i < len(df) - 1:
        long_sig = closes[i] > highs[i-lookback:i].max()
        short_sig = closes[i] < lows[i-lookback:i].min()
        if not (long_sig or short_sig):
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df) - 1:
            break

        entry_price = opens[entry_idx]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            i += 1
            continue

        R = stop_mult * a
        direction = 1 if long_sig else -1
        stop_price = entry_price - direction * R
        target_price = entry_price + direction * R * rr

        exit_idx, exit_reason = None, None
        for j in range(entry_idx, min(entry_idx + max_bars, len(df))):
            h, l = highs[j], lows[j]
            if direction == 1:
                hit_t, hit_s = h >= target_price, l <= stop_price
            else:
                hit_t, hit_s = l <= target_price, h >= stop_price
            if hit_s and hit_t:
                exit_reason, exit_idx = "both", j
                break
            if hit_s:
                exit_reason, exit_idx = "stop", j
                break
            if hit_t:
                exit_reason, exit_idx = "target", j
                break

        if exit_idx is None:
            exit_reason = "time"
            exit_idx = min(entry_idx + max_bars, len(df) - 1)

        events.append({
            "entry_idx": entry_idx, "exit_idx": exit_idx,
            "direction": direction, "entry_price": entry_price,
            "R": R, "stop_price": stop_price, "target_price": target_price,
            "exit_reason_actual": exit_reason,
            "entry_time": df.index[entry_idx], "exit_time": df.index[exit_idx],
        })
        i = exit_idx + 1
    return pd.DataFrame(events)


def build_states(df, events, horizon=12):
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    states = []
    for _, ev in events.iterrows():
        e_idx, x_idx = ev["entry_idx"], ev["exit_idx"]
        direction = ev["direction"]
        entry_price, R = ev["entry_price"], ev["R"]
        stop_price, target_price = ev["stop_price"], ev["target_price"]

        mfe, mae = 0.0, 0.0
        max_fav_idx = e_idx
        last_stop_touch = -1

        for j in range(e_idx, x_idx + 1):
            if direction == 1:
                new_mfe = (highs[j] - entry_price) / R
                mae = min(mae, (lows[j] - entry_price) / R)
                if lows[j] <= stop_price:
                    last_stop_touch = j
            else:
                new_mfe = (entry_price - lows[j]) / R
                mae = min(mae, (entry_price - highs[j]) / R)
                if highs[j] >= stop_price:
                    last_stop_touch = j
            if new_mfe > mfe:
                mfe, max_fav_idx = new_mfe, j

            bars_in_pos = j - e_idx + 1
            target = 0
            for k in range(j + 1, min(j + 1 + horizon, len(df))):
                if direction == 1:
                    hit_t, hit_s = highs[k] >= target_price, lows[k] <= stop_price
                else:
                    hit_t, hit_s = lows[k] <= target_price, highs[k] >= stop_price
                if hit_s:
                    break
                if hit_t:
                    target = 1
                    break

            row = df.iloc[j]
            pnl_now = direction * (closes[j] - entry_price) / R

            states.append({
                "bar_idx": j, "bar_time": df.index[j], "entry_idx": e_idx,
                "direction": direction, "entry_price": entry_price, "R": R,
                "bars_in_pos": bars_in_pos, "pnl_r": pnl_now,
                "mfe_r": mfe, "mae_r": mae,
                "bars_since_mfe": j - max_fav_idx,
                "bars_since_stop_touch": (j - last_stop_touch) if last_stop_touch >= 0 else 999,
                "ret_1": row["ret_1"], "ret_2": row["ret_2"],
                "ret_3": row["ret_3"], "ret_5": row["ret_5"],
                "streak": row["streak"],
                "atr_ratio": row["atr14"] / R * 1.5 if R > 0 else np.nan,
                "regime": row["regime"],
                "pos_in_range_20": row["pos_in_range_20"],
                "pos_in_range_50": row["pos_in_range_50"],
                "pos_in_range_100": row["pos_in_range_100"],
                "tickvol_rel": row["tickvol_rel"],
                "spread_rel": row["spread_rel"],
                "spread": row["spread"],
                "hour": row["hour"], "dow": row["dow"],
                "target": target,
                "actual_exit_idx": x_idx,
                "actual_exit_reason": ev["exit_reason_actual"],
            })
    return pd.DataFrame(states)


FEATURES = [
    "bars_in_pos", "pnl_r", "mfe_r", "mae_r",
    "bars_since_mfe", "bars_since_stop_touch",
    "ret_1", "ret_2", "ret_3", "ret_5", "streak",
    "atr_ratio", "regime",
    "pos_in_range_20", "pos_in_range_50", "pos_in_range_100",
    "tickvol_rel", "spread_rel", "spread",
    "hour", "dow",
]


# ============================================================
# WALK-FORWARD + СБОР ВСЕХ ДАННЫХ ДЛЯ ОТЧЁТА
# ============================================================
def walk_forward(df, states, features,
                 rr=2.0, train_frac=0.4, step=125,
                 max_bars=36, threshold=0.5):
    states = states.dropna(subset=features + ["target"]).copy()
    states = states.sort_values("bar_time").reset_index(drop=True)

    n = len(states)
    start = int(n * 0.2)
    split = int(n * train_frac)
    fold = 0

    model_trades = []
    base_trades = []
    reliability_p = []
    reliability_y = []
    importances = []
    fold_aucs = []

    total_folds = len(range(split, n, step))

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
        importances.append(model.feature_importances_)

        if test["target"].nunique() > 1:
            p_test = model.predict_proba(test[features].values)[:, 1]
            auc = roc_auc_score(test["target"], p_test)
            fold_aucs.append(auc)
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

            # --- BASELINE ---
            exit_close = df["close"].values[actual_exit_idx]
            exit_spread = df["spread_price"].values[actual_exit_idx]
            base_pnl = direction * (exit_close - entry_price) / R
            base_pnl -= (entry_spread + exit_spread) / R
            base_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[actual_exit_idx],
                "pnl_r": base_pnl,
                "bars_held": actual_exit_idx - int(e_idx) + 1,
            })

            # --- MODEL ---
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
            pnl_raw = direction * (exit_close - entry_price) / R
            pnl_raw -= (entry_spread + exit_spread) / R

            p_entry = float(P[0])
            size = float(np.clip(0.5 + (p_entry - 0.2) / 0.6, 0.5, 1.5))

            model_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[exit_idx],
                "pnl_r": pnl_raw * size,
                "pnl_r_raw": pnl_raw,
                "size": size,
                "p_entry": p_entry,
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "direction": direction,
                "regime_at_entry": float(pos["regime"].iloc[0]),
                "hit_target_actual": int(actual_exit_reason == "target"),
                "fold": fold,
            })

        fold += 1
        split = test_end
        print(f"  fold {fold}/{total_folds}  trades={len(model_trades)}", end="\r")

    print()
    model_trades = pd.DataFrame(model_trades)
    base_trades = pd.DataFrame(base_trades)
    reliability = pd.DataFrame({"p": reliability_p, "y": reliability_y})
    mean_imp = np.mean(importances, axis=0) if importances else np.zeros(len(features))

    return model_trades, base_trades, reliability, mean_imp, fold_aucs


# ============================================================
# МЕТРИКИ
# ============================================================
def metrics(trades):
    if len(trades) == 0:
        return {"n": 0, "total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "pf": 0.0}
    eq = trades["pnl_r"].cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    gains = trades.loc[trades["pnl_r"] > 0, "pnl_r"].sum()
    losses = -trades.loc[trades["pnl_r"] < 0, "pnl_r"].sum()
    return {
        "n": int(len(trades)),
        "total_r": float(trades["pnl_r"].sum()),
        "avg_r": float(trades["pnl_r"].mean()),
        "win_rate": float((trades["pnl_r"] > 0).mean()),
        "sharpe": float(trades["pnl_r"].mean() / (trades["pnl_r"].std() + 1e-9) * np.sqrt(len(trades))),
        "max_dd": float(dd.min()),
        "pf": float(gains / losses) if losses > 0 else float("inf"),
    }


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
def plot_main_report(trades, base_trades):
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    base_trades = base_trades.sort_values("exit_time").reset_index(drop=True)

    eq_model = trades["pnl_r"].cumsum().values
    eq_base = base_trades["pnl_r"].cumsum().values
    peak_m = np.maximum.accumulate(eq_model)
    peak_b = np.maximum.accumulate(eq_base)
    dd_model = eq_model - peak_m
    dd_base = eq_base - peak_b

    # Rolling Sharpe (window 30 trades)
    roll_sharpe = trades["pnl_r"].rolling(30).apply(
        lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(len(x)), raw=True)

    # Monthly PnL
    trades["month"] = pd.to_datetime(trades["exit_time"]).dt.to_period("M")
    monthly = trades.groupby("month")["pnl_r"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Equity
    ax = axes[0, 0]
    ax.plot(base_trades["exit_time"], eq_base, label="Baseline (hold always)",
            color="gray", alpha=0.7, linewidth=1.5)
    ax.plot(trades["exit_time"], eq_model, label="Model (final config)",
            color="steelblue", linewidth=2.2)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Equity curve (R)", fontsize=12, fontweight="bold")
    ax.set_ylabel("R")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[0, 1]
    ax.plot(trades["exit_time"], dd_model, color="firebrick",
            linewidth=1.8, label="Model DD")
    ax.plot(base_trades["exit_time"], dd_base, color="orange",
            alpha=0.5, linewidth=1.5, label="Baseline DD")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Drawdown (R)", fontsize=12, fontweight="bold")
    ax.set_ylabel("R")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Monthly PnL
    ax = axes[1, 0]
    colors = ["steelblue" if v > 0 else "firebrick" for v in monthly.values]
    ax.bar(monthly.index.astype(str), monthly.values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("PnL по месяцам", fontsize=12, fontweight="bold")
    ax.set_ylabel("R")
    ax.tick_params(axis="x", rotation=60)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(8)
    ax.grid(True, alpha=0.3)

    # Rolling Sharpe
    ax = axes[1, 1]
    ax.plot(trades["exit_time"], roll_sharpe, color="seagreen", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color="orange", linestyle="--", linewidth=1, label="Sharpe = 1")
    ax.axhline(2, color="green", linestyle="--", linewidth=1, label="Sharpe = 2")
    ax.set_title("Rolling Sharpe (окно 30 сделок)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("final_report.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_diagnostics(trades, reliability, feature_importance):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Reliability diagram
    ax = axes[0, 0]
    bins = pd.qcut(reliability["p"], 10, duplicates="drop")
    cal = reliability.groupby(bins, observed=True).agg(
        p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size"))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6,
            label="Perfect calibration")
    ax.plot(cal["p_mean"], cal["y_mean"], marker="o", color="steelblue",
            linewidth=2, label="Model")
    ax.fill_between(cal["p_mean"], cal["p_mean"], cal["y_mean"],
                    color="steelblue", alpha=0.15)
    ax.set_xlabel("Predicted P(target)")
    ax.set_ylabel("Actual fraction")
    ax.set_title("Калибровка модели", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[0, 1]
    fi = pd.DataFrame({"feature": FEATURES, "importance": feature_importance})
    fi = fi.sort_values("importance", ascending=True).tail(15)
    ax.barh(fi["feature"], fi["importance"], color="steelblue", alpha=0.85)
    ax.set_title("Важность признаков (среднее по фолдам)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Regime breakdown
    ax = axes[1, 0]
    t = trades.dropna(subset=["regime_at_entry"]).copy()
    t["regime_bin"] = pd.qcut(t["regime_at_entry"], 3,
                              labels=["low vol", "mid vol", "high vol"])
    g = t.groupby("regime_bin", observed=True).agg(
        total_r=("pnl_r", "sum"), n=("pnl_r", "size"))
    colors = ["steelblue" if v > 0 else "firebrick" for v in g["total_r"].values]
    bars = ax.bar(g.index.astype(str), g["total_r"].values, color=colors, alpha=0.85)
    for bar, n in zip(bars, g["n"].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("PnL по режиму волатильности", fontsize=12, fontweight="bold")
    ax.set_ylabel("R")
    ax.grid(True, alpha=0.3, axis="y")

    # Distribution of bars held + P(entry)
    ax = axes[1, 1]
    ax.hist(trades["bars_held"], bins=20, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(trades["bars_held"].mean(), color="red", linestyle="--",
               label=f"mean = {trades['bars_held'].mean():.1f}")
    ax.axvline(trades["bars_held"].median(), color="orange", linestyle="--",
               label=f"median = {trades['bars_held'].median():.0f}")
    ax.set_xlabel("bars held")
    ax.set_ylabel("число сделок")
    ax.set_title("Распределение времени в позиции", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("final_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("GBPUSD H1 — Финальная конфигурация")
    print("=" * 80)

    df = load_data("GBPUSD_H1.csv")
    print(f"Баров: {len(df)}  Период: {df.index.min()} — {df.index.max()}")

    df = add_indicators(df)

    events = simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.0, max_bars=48)
    print(f"Позиций: {len(events)}")

    states = build_states(df, events, horizon=12)
    print(f"Состояний: {len(states)}")
    print(f"Баланс target: {states['target'].mean():.3f}")
    print()

    trades, base, reliability, feat_imp, aucs = walk_forward(
        df, states, FEATURES,
        rr=2.0, train_frac=0.4, step=125,
        max_bars=36, threshold=0.5,
    )

    m = metrics(trades)
    b = metrics(base)

    # ------- Bootstrap -------
    pnls = trades["pnl_r"].values
    n_iter = 2000
    totals = np.empty(n_iter)
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        s = RNG.choice(pnls, size=len(pnls), replace=True)
        totals[i] = s.sum()
        sharpes[i] = s.mean() / (s.std() + 1e-9) * np.sqrt(len(s))
    p_pos = (totals > 0).mean()
    p_sh1 = (sharpes > 1).mean()

    # ------- Leave-one-quarter-out -------
    trades_q = trades.copy()
    trades_q["exit_time"] = pd.to_datetime(trades_q["exit_time"])
    trades_q["quarter"] = trades_q["exit_time"].dt.to_period("Q")
    loo_rows = []
    for q in sorted(trades_q["quarter"].unique()):
        sub = trades_q[trades_q["quarter"] != q]
        mm = metrics(sub)
        loo_rows.append({"excluded": str(q), "n": mm["n"],
                         "total_r": mm["total_r"], "sharpe": mm["sharpe"]})
    loo = pd.DataFrame(loo_rows)
    worst_loo = loo["total_r"].min()

    # ------- Режимный анализ -------
    t_reg = trades.dropna(subset=["regime_at_entry"]).copy()
    t_reg["regime_bin"] = pd.qcut(t_reg["regime_at_entry"], 3,
                                  labels=["low_vol", "mid_vol", "high_vol"])
    reg_summary = t_reg.groupby("regime_bin", observed=True).agg(
        n=("pnl_r", "size"),
        total_r=("pnl_r", "sum"),
        avg_r=("pnl_r", "mean"),
        win_rate=("pnl_r", lambda x: (x > 0).mean()),
    )

    # ============================================================
    # ОТЧЁТ
    # ============================================================
    print()
    print("=" * 80)
    print("  ФИНАЛЬНЫЙ ОТЧЁТ")
    print("=" * 80)
    print()
    print(f"  Конфигурация: rr=2.0, step=125, prob exit @ 0.5, sizing on")
    print(f"  Инструмент:   EURUSD H1")
    print(f"  Период:       {df.index.min().date()} — {df.index.max().date()}")
    print()
    print("  ── СДЕЛКИ ─────────────────────────────────────────")
    print(f"    Всего сделок:            {m['n']}")
    print(f"    Win rate:                {m['win_rate']*100:.1f}%")
    print(f"    Средний PnL:             {m['avg_r']:.4f} R")
    print(f"    Средний размер позиции:  {trades['size'].mean():.3f} R")
    print(f"    Среднее время в позиции: {trades['bars_held'].mean():.1f} баров")
    print()
    print("  ── ДОХОДНОСТЬ ─────────────────────────────────────")
    print(f"    Total R:                 {m['total_r']:+.2f} R")
    print(f"    Sharpe:                  {m['sharpe']:.2f}")
    print(f"    Profit factor:           {m['pf']:.2f}")
    print(f"    Max DD:                  {m['max_dd']:.2f} R")
    print()
    print("  ── BASELINE (без управления) ──────────────────────")
    print(f"    Total R:                 {b['total_r']:+.2f} R")
    print(f"    Sharpe:                  {b['sharpe']:.2f}")
    print(f"    Max DD:                  {b['max_dd']:.2f} R")
    print()
    print("  ── УСТОЙЧИВОСТЬ ───────────────────────────────────")
    print(f"    Leave-one-quarter-out:   {'✅' if worst_loo > 0 else '❌'}"
          f" без худшего квартала {worst_loo:+.2f} R")
    print(f"    Bootstrap P(profit>0):   {p_pos:.3f}  {'✅' if p_pos > 0.95 else '🟡'}")
    print(f"    Bootstrap P(Sharpe>1):   {p_sh1:.3f}  {'✅' if p_sh1 > 0.9 else '🟡'}")
    print(f"    Режимная устойчивость:   "
          f"{'✅ все 3 режима' if (reg_summary['total_r'] > 0).all() else '🟡 не все'}")
    print()
    print("  ── МОДЕЛЬ ─────────────────────────────────────────")
    print(f"    AUC по фолдам:           {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"    Число фолдов:            {len(aucs)}")
    print()
    print("  ── РЕЖИМЫ ВОЛАТИЛЬНОСТИ ───────────────────────────")
    for name, row in reg_summary.iterrows():
        print(f"    {str(name):10s}  n={int(row['n']):3d}  "
              f"total_r={row['total_r']:+.2f}  avg_r={row['avg_r']:+.4f}  "
              f"win={row['win_rate']*100:.0f}%")
    print()
    print("  ── LEAVE-ONE-QUARTER-OUT ──────────────────────────")
    print(loo.to_string(index=False))
    print()
    print("=" * 80)

    # Сохранение
    trades.to_csv("final_trades.csv", index=False)
    base.to_csv("final_baseline.csv", index=False)
    loo.to_csv("final_loo.csv", index=False)
    reg_summary.to_csv("final_regime.csv")
    print("\nСохранено: final_trades.csv, final_baseline.csv, final_loo.csv, final_regime.csv")
    print("Графики:   final_report.png, final_diagnostics.png")

    # Построение графиков
    print("\nСтроим графики...")
    plot_main_report(trades, base)
    plot_diagnostics(trades, reliability, feat_imp)
    print("Готово.")