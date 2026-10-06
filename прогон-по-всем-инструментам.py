#!/usr/bin/env python3
# ============================================================
# Batch runner: прогон конфигурации F по всем инструментам
# ============================================================
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
RNG = np.random.default_rng(42)

# ============================================================
# КОНФИГУРАЦИЯ (не менять — это наша F)
# ============================================================
DATA_DIR = "data"                # папка с CSV-файлами
RESULTS_DIR = "batch_results"    # куда писать результаты
RR = 2.0
STEP = 125
THRESHOLD = 0.5
TRAIN_FRAC = 0.4
MAX_BARS = 36
HORIZON = 12
LOOKBACK = 20
STOP_MULT = 1.5
N_BOOTSTRAP = 2000
MIN_BARS = 5000                  # минимум баров для запуска

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# ЗАГРУЗКА И ИНДИКАТОРЫ
# ============================================================
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["date", "time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")].dropna(
        subset=["open", "high", "low", "close"])
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
    df["streak"] = direction.groupby(
        (direction != direction.shift()).cumsum()).cumcount() + 1
    df["streak"] *= direction

    df["tickvol_ma50"] = df["tickvol"].rolling(50).mean()
    df["tickvol_rel"] = df["tickvol"] / df["tickvol_ma50"]
    df["spread_ma50"] = df["spread"].rolling(50).mean()
    df["spread_rel"] = df["spread"] / df["spread_ma50"]

    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    return df


def simulate_positions(df, lookback=LOOKBACK, stop_mult=STOP_MULT,
                       rr=RR, max_bars=48):
    events = []
    highs, lows = df["high"].values, df["low"].values
    closes, opens = df["close"].values, df["open"].values
    atr = df["atr14"].values

    i = lookback
    while i < len(df) - 1:
        long_sig = closes[i] > highs[i - lookback:i].max()
        short_sig = closes[i] < lows[i - lookback:i].min()
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


def build_states(df, events, horizon=HORIZON):
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
# WALK-FORWARD
# ============================================================
def walk_forward(df, states, features,
                 rr=RR, train_frac=TRAIN_FRAC, step=STEP,
                 max_bars=MAX_BARS, threshold=THRESHOLD):
    states = states.dropna(subset=features + ["target"]).copy()
    states = states.sort_values("bar_time").reset_index(drop=True)

    n = len(states)
    start = int(n * 0.2)
    split = int(n * train_frac)
    fold = 0

    model_trades = []
    base_trades = []
    aucs = []

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

        if test["target"].nunique() > 1:
            p_test = model.predict_proba(test[features].values)[:, 1]
            aucs.append(roc_auc_score(test["target"], p_test))

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

            # baseline
            exit_close = df["close"].values[actual_exit_idx]
            exit_spread = df["spread_price"].values[actual_exit_idx]
            base_pnl = direction * (exit_close - entry_price) / R
            base_pnl -= (entry_spread + exit_spread) / R
            base_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[actual_exit_idx],
                "pnl_r": base_pnl, "fold": fold,
            })

            # model
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
                exit_idx, exit_reason = actual_exit_idx, "actual"
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
                "pnl_r": pnl_raw * size, "pnl_r_raw": pnl_raw,
                "size": size, "p_entry": p_entry,
                "exit_reason": exit_reason, "bars_held": bars_held,
                "direction": direction,
                "regime_at_entry": float(pos["regime"].iloc[0]),
                "fold": fold,
            })

        fold += 1
        split = test_end

    model_trades = pd.DataFrame(model_trades)
    base_trades = pd.DataFrame(base_trades)
    return model_trades, base_trades, np.mean(aucs) if aucs else np.nan


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
        "sharpe": float(trades["pnl_r"].mean() /
                        (trades["pnl_r"].std() + 1e-9) * np.sqrt(len(trades))),
        "max_dd": float(dd.min()),
        "pf": float(gains / losses) if losses > 0 else float("inf"),
    }


def bootstrap_stats(trades, n_iter=N_BOOTSTRAP):
    if len(trades) < 30:
        return {"p_pos": 0.0, "p_sh1": 0.0, "median_total": 0.0}
    pnls = trades["pnl_r"].values
    n = len(pnls)
    totals = np.empty(n_iter)
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        s = RNG.choice(pnls, size=n, replace=True)
        totals[i] = s.sum()
        sharpes[i] = s.mean() / (s.std() + 1e-9) * np.sqrt(n)
    return {
        "p_pos": float((totals > 0).mean()),
        "p_sh1": float((sharpes > 1).mean()),
        "median_total": float(np.median(totals)),
    }


def leave_one_quarter_out(trades):
    if len(trades) < 30:
        return np.nan
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["quarter"] = t["exit_time"].dt.to_period("Q")
    totals = []
    for q in t["quarter"].unique():
        sub = t[t["quarter"] != q]
        totals.append(sub["pnl_r"].sum())
    return float(min(totals)) if totals else np.nan


# ============================================================
# ОДИН ИНСТРУМЕНТ
# ============================================================
def run_instrument(path):
    name = os.path.splitext(os.path.basename(path))[0]
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"[{name}] запуск")
    print(f"{'='*70}")

    try:
        df = load_data(path)
    except Exception as e:
        return {"instrument": name, "status": f"load_error: {e}"}

    if len(df) < MIN_BARS:
        return {"instrument": name, "status": f"too_few_bars: {len(df)}"}

    df = add_indicators(df)
    events = simulate_positions(df)
    if len(events) < 50:
        return {"instrument": name, "status": f"too_few_events: {len(events)}"}

    states = build_states(df, events)
    if len(states) < 500:
        return {"instrument": name, "status": f"too_few_states: {len(states)}"}

    trades, base, auc = walk_forward(df, states, FEATURES)

    if len(trades) < 30:
        return {"instrument": name, "status": f"too_few_trades: {len(trades)}"}

    m = metrics(trades)
    b = metrics(base)
    bs = bootstrap_stats(trades)
    loo_min = leave_one_quarter_out(trades)

    # Режимная устойчивость
    t_reg = trades.dropna(subset=["regime_at_entry"]).copy()
    if len(t_reg) >= 30:
        t_reg["regime_bin"] = pd.qcut(t_reg["regime_at_entry"], 3,
                                      labels=["low", "mid", "high"],
                                      duplicates="drop")
        regime_pnl = t_reg.groupby("regime_bin", observed=True)["pnl_r"].sum()
        regime_ok = bool((regime_pnl > 0).all())
    else:
        regime_ok = False

    elapsed = time.time() - t0
    print(f"[{name}] готово за {elapsed:.1f}s  "
          f"n={m['n']}  total_r={m['total_r']:+.2f}  sharpe={m['sharpe']:.2f}")

    # Сохраняем сделки
    trades.to_csv(os.path.join(RESULTS_DIR, f"{name}_trades.csv"), index=False)

    return {
        "instrument": name,
        "status": "ok",
        "bars": len(df),
        "events": len(events),
        "states": len(states),
        "n_trades": m["n"],
        "total_r": m["total_r"],
        "avg_r": m["avg_r"],
        "win_rate": m["win_rate"],
        "sharpe": m["sharpe"],
        "max_dd": m["max_dd"],
        "pf": m["pf"],
        "auc": auc,
        "avg_bars_held": trades["bars_held"].mean(),
        "avg_size": trades["size"].mean(),
        "base_total_r": b["total_r"],
        "base_sharpe": b["sharpe"],
        "base_max_dd": b["max_dd"],
        "delta_vs_base": m["total_r"] - b["total_r"],
        "boot_p_pos": bs["p_pos"],
        "boot_p_sh1": bs["p_sh1"],
        "boot_median": bs["median_total"],
        "loo_min": loo_min,
        "regime_ok": regime_ok,
        "elapsed_sec": elapsed,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Ищем CSV в DATA_DIR
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        # fallback: текущая папка
        files = sorted(glob.glob("*.csv"))

    if not files:
        print(f"Не найдено CSV-файлов в '{DATA_DIR}' или текущей папке.")
        raise SystemExit(1)

    print(f"Найдено файлов: {len(files)}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    rows = []
    t_start = time.time()
    for i, path in enumerate(files, 1):
        print(f"\n>>> [{i}/{len(files)}] {os.path.basename(path)}")
        try:
            r = run_instrument(path)
        except Exception as e:
            r = {"instrument": os.path.basename(path),
                 "status": f"unhandled_error: {e}"}
        rows.append(r)

    df_res = pd.DataFrame(rows)
    total_time = time.time() - t_start

    # ============================================================
    # СВОДНАЯ ТАБЛИЦА
    # ============================================================
    ok = df_res[df_res["status"] == "ok"].copy()
    failed = df_res[df_res["status"] != "ok"].copy()

    if len(ok) == 0:
        print("\nНи один инструмент не отработал успешно.")
        print(failed.to_string(index=False))
        raise SystemExit(1)

    # Ранжирование по устойчивости
    ok["robust_score"] = (
        ok["sharpe"].clip(lower=0) *
        ok["boot_p_pos"] *
        ok["boot_p_sh1"] *
        (ok["loo_min"] > 0).astype(int)
    )

    ok = ok.sort_values("robust_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 140)
    print("СВОДНАЯ ТАБЛИЦА ПО ВСЕМ ИНСТРУМЕНТАМ")
    print("=" * 140)
    cols_show = [
        "instrument", "n_trades", "total_r", "sharpe", "max_dd",
        "win_rate", "auc", "avg_bars_held", "avg_size",
        "base_total_r", "delta_vs_base",
        "boot_p_pos", "boot_p_sh1", "loo_min", "regime_ok", "robust_score",
    ]
    print(ok[cols_show].to_string(index=False))

    if len(failed) > 0:
        print("\n" + "=" * 140)
        print("НЕ ОТРАБОТАЛИ")
        print("=" * 140)
        print(failed[["instrument", "status"]].to_string(index=False))

    # Сохранение
    ok.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    failed.to_csv(os.path.join(RESULTS_DIR, "summary_failed.csv"), index=False)
    df_res.to_csv(os.path.join(RESULTS_DIR, "summary_full.csv"), index=False)

    print(f"\nОбщее время: {total_time/60:.1f} мин")
    print(f"Успешно:     {len(ok)}")
    print(f"Провалено:   {len(failed)}")

    # ============================================================
    # ГРАФИКИ
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    # Sharpe бар-chart
    ax = axes[0, 0]
    colors = ["steelblue" if s > 1 else ("orange" if s > 0 else "firebrick")
              for s in ok["sharpe"]]
    ax.barh(ok["instrument"], ok["sharpe"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axvline(1, color="green", linestyle="--", linewidth=1, label="Sharpe=1")
    ax.axvline(2, color="darkgreen", linestyle="--", linewidth=1, label="Sharpe=2")
    ax.set_title("Sharpe по инструментам", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # Total R vs baseline
    ax = axes[0, 1]
    y = np.arange(len(ok))
    h = 0.38
    ax.barh(y - h/2, ok["total_r"], h, label="Model", color="steelblue")
    ax.barh(y + h/2, ok["base_total_r"], h, label="Baseline",
            color="gray", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ok["instrument"])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title("Total R: Model vs Baseline", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Bootstrap P(profit > 0)
    ax = axes[1, 0]
    colors = ["steelblue" if p > 0.95 else ("orange" if p > 0.8 else "firebrick")
              for p in ok["boot_p_pos"]]
    ax.barh(ok["instrument"], ok["boot_p_pos"], color=colors, alpha=0.85)
    ax.axvline(0.95, color="green", linestyle="--", linewidth=1, label="0.95")
    ax.axvline(0.8, color="orange", linestyle="--", linewidth=1, label="0.80")
    ax.set_xlim(0, 1.02)
    ax.set_title("Bootstrap P(profit > 0)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Leave-one-quarter-out (min)
    ax = axes[1, 1]
    colors = ["steelblue" if v > 0 else "firebrick" for v in ok["loo_min"]]
    ax.barh(ok["instrument"], ok["loo_min"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title("Leave-one-quarter-out (min Total R)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "batch_report.png"),
                dpi=130, bbox_inches="tight")
    plt.show()

    # ============================================================
    # ВЕРДИКТ
    # ============================================================
    robust = ok[(ok["sharpe"] > 1) &
                (ok["boot_p_pos"] > 0.95) &
                (ok["loo_min"] > 0)].copy()

    print("\n" + "=" * 140)
    print("УСТОЙЧИВЫЕ ИНСТРУМЕНТЫ (Sharpe>1 + Bootstrap>0.95 + Leave-one-out>0)")
    print("=" * 140)
    if len(robust) == 0:
        print("Нет инструментов, прошедших все три критерия.")
    else:
        print(robust[cols_show].to_string(index=False))
        print(f"\nВсего устойчивых: {len(robust)} из {len(ok)}")

        # Сохранить
        robust.to_csv(os.path.join(RESULTS_DIR, "robust_instruments.csv"),
                      index=False)

    print("\nСохранено в папку:", RESULTS_DIR)