# ============================================================
# EURUSD H1 — Final verification
# 1. Best combo (перебор топ-конфигов)
# 2. Leave-one-quarter-out
# 3. Bootstrap по сделкам
# 4. Режимный анализ (тренд/флэт)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")

RNG = np.random.default_rng(42)


# ============================================================
# 1. БАЗОВЫЕ ФУНКЦИИ
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


def simulate_positions(df, lookback=20, stop_mult=1.5, rr=2.5, max_bars=48):
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
                "mfe_over_pnl": mfe / pnl_now if pnl_now > 0.05 else np.nan,
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


BASE_FEATURES = [
    "bars_in_pos", "pnl_r", "mfe_r", "mae_r",
    "bars_since_mfe", "bars_since_stop_touch",
    "ret_1", "ret_2", "ret_3", "ret_5", "streak",
    "atr_ratio", "regime",
    "pos_in_range_20", "pos_in_range_50", "pos_in_range_100",
    "tickvol_rel", "spread_rel", "spread",
    "hour", "dow",
]


_STATE_CACHE = {}

def get_states(df, rr):
    key = f"rr_{rr}"
    if key not in _STATE_CACHE:
        print(f"  [cache] строим states для rr={rr}")
        events = simulate_positions(df, lookback=20, stop_mult=1.5, rr=rr, max_bars=48)
        states = build_states(df, events, horizon=12)
        _STATE_CACHE[key] = (events, states)
    return _STATE_CACHE[key]


# ============================================================
# 2. WALK-FORWARD
# ============================================================
def walk_forward(df, states, features,
                 rr=2.5, train_frac=0.4, step=500,
                 max_bars=36, exit_mode="ev", prob_threshold=0.5,
                 use_sizing=True, seed=42):
    states = states.dropna(subset=features + ["target"]).copy()
    states = states.sort_values("bar_time").reset_index(drop=True)

    n = len(states)
    start = int(n * 0.2)
    split = int(n * train_frac)
    fold = 0

    model_trades, base_trades = [], []
    aucs = []

    while split < n:
        test_end = min(split + step, n)
        train, test = states.iloc[start:split], states.iloc[split:test_end]
        if len(train) < 200 or len(test) < 50:
            split = test_end
            continue

        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=seed,
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
            for k, (_, row) in enumerate(pos.iterrows()):
                p = P[k]
                if row["bars_in_pos"] >= max_bars:
                    exit_idx, exit_reason = int(row["bar_idx"]), "timeout"
                    break
                if exit_mode == "ev":
                    ev_hold = p * rr - (1 - p) * 1.0
                    if ev_hold < row["pnl_r"]:
                        exit_idx, exit_reason = int(row["bar_idx"]), "ev_exit"
                        break
                else:
                    if p < prob_threshold:
                        exit_idx, exit_reason = int(row["bar_idx"]), "prob_exit"
                        break

            if exit_idx is None:
                exit_idx, exit_reason = actual_exit_idx, "actual"

            exit_close = df["close"].values[exit_idx]
            exit_spread = df["spread_price"].values[exit_idx]
            pnl_raw = direction * (exit_close - entry_price) / R
            pnl_raw -= (entry_spread + exit_spread) / R

            if use_sizing:
                p_entry = P[0]
                size = float(np.clip(0.5 + (p_entry - 0.2) / 0.6, 0.5, 1.5))
            else:
                size = 1.0

            model_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[exit_idx],
                "pnl_r": pnl_raw * size, "pnl_r_raw": pnl_raw,
                "size": size, "p_entry": P[0],
                "exit_reason": exit_reason, "fold": fold,
                "direction": direction,
                "regime_at_entry": states.loc[
                    states["entry_idx"] == e_idx, "regime"].iloc[0]
                if len(states[states["entry_idx"] == e_idx]) > 0 else np.nan,
            })

        fold += 1
        split = test_end

    model_trades = pd.DataFrame(model_trades)
    base_trades = pd.DataFrame(base_trades)
    if len(model_trades) > 0:
        model_trades.attrs["auc_mean"] = float(np.mean(aucs)) if aucs else np.nan
    return model_trades, base_trades


def metrics(trades):
    if len(trades) == 0:
        return {"n": 0, "total_r": np.nan, "avg_r": np.nan, "win_rate": np.nan,
                "sharpe": np.nan, "max_dd": np.nan, "pf": np.nan}
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
        "pf": float(gains / losses) if losses > 0 else np.inf,
    }


# ============================================================
# 3. TEST 1 — BEST COMBO
# ============================================================
def test_best_combo(df):
    print("\n" + "=" * 100)
    print("TEST 1: перебор лучших комбинаций")
    print("=" * 100)

    configs = [
        # name,                     rr,  step, exit_mode, sizing
        ("A rr2.0 step250 prob",     2.0, 250, "prob", True),
        ("B rr2.0 step250 EV",       2.0, 250, "ev",   True),
        ("C rr2.0 step500 prob",     2.0, 500, "prob", True),
        ("D rr2.0 step250 prob nosz",2.0, 250, "prob", False),
        ("E rr2.5 step250 prob",     2.5, 250, "prob", True),
        ("F rr2.0 step125 prob",     2.0, 125, "prob", True),
    ]

    rows = []
    trades_store = {}
    for name, rr, step, em, sz in configs:
        _, states = get_states(df, rr)
        m, b = walk_forward(df, states, BASE_FEATURES,
                            rr=rr, step=step, exit_mode=em, use_sizing=sz)
        mm = metrics(m)
        bb = metrics(b)
        rows.append({
            "config": name, "n": mm["n"], "total_r": mm["total_r"],
            "avg_r": mm["avg_r"], "win_rate": mm["win_rate"],
            "sharpe": mm["sharpe"], "max_dd": mm["max_dd"], "pf": mm["pf"],
            "base_total_r": bb["total_r"], "auc": m.attrs.get("auc_mean", np.nan),
        })
        trades_store[name] = m
        print(f"  {name}: n={mm['n']}, total_r={mm['total_r']:.2f}, "
              f"sharpe={mm['sharpe']:.2f}, dd={mm['max_dd']:.2f}")

    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\nТоп конфигураций:")
    print(res.to_string(index=False))

    best_name = res.iloc[0]["config"]
    print(f"\n>>> ЛУЧШАЯ КОНФИГУРАЦИЯ: {best_name}")
    return best_name, trades_store[best_name], res


# ============================================================
# 4. TEST 2 — LEAVE-ONE-QUARTER-OUT
# ============================================================
def test_loo_quarter(trades):
    print("\n" + "=" * 100)
    print("TEST 2: leave-one-quarter-out")
    print("=" * 100)

    trades = trades.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["quarter"] = trades["exit_time"].dt.to_period("Q")

    quarters = sorted(trades["quarter"].unique())
    full_total = trades["pnl_r"].sum()
    full_sharpe = metrics(trades)["sharpe"]

    print(f"\nПолная выборка: n={len(trades)}, total_r={full_total:.2f}, "
          f"sharpe={full_sharpe:.2f}\n")

    rows = []
    for q in quarters:
        sub = trades[trades["quarter"] != q]
        m = metrics(sub)
        delta = m["total_r"] - full_total
        rows.append({
            "excluded": str(q),
            "n": m["n"], "total_r": m["total_r"],
            "avg_r": m["avg_r"], "sharpe": m["sharpe"],
            "max_dd": m["max_dd"], "delta_r": delta,
        })

    loo = pd.DataFrame(rows)
    print(loo.to_string(index=False))

    # Критерий: если удаление любого одного квартала не превращает total_r
    # в глубоко отрицательное значение — модель устойчива
    worst = loo["total_r"].min()
    if worst > 0:
        verdict = "✅ УСТОЙЧИВА: даже без худшего квартала прибыль положительная"
    elif worst > -2:
        verdict = "🟡 ПОГРАНИЧНАЯ: без худшего квартала результат около нуля"
    else:
        verdict = "❌ ХРУПКАЯ: один квартал даёт весь результат"
    print(f"\n>>> {verdict}")
    return loo, verdict


# ============================================================
# 5. TEST 3 — BOOTSTRAP
# ============================================================
def test_bootstrap(trades, n_iter=2000):
    print("\n" + "=" * 100)
    print("TEST 3: bootstrap по сделкам")
    print("=" * 100)

    pnls = trades["pnl_r"].values
    n = len(pnls)

    totals = np.empty(n_iter)
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        sample = RNG.choice(pnls, size=n, replace=True)
        totals[i] = sample.sum()
        sharpes[i] = sample.mean() / (sample.std() + 1e-9) * np.sqrt(n)

    p_positive = (totals > 0).mean()
    p_sharpe_gt1 = (sharpes > 1.0).mean()

    print(f"\nTotal R: median={np.median(totals):.2f}, "
          f"5%={np.percentile(totals, 5):.2f}, "
          f"95%={np.percentile(totals, 95):.2f}")
    print(f"Sharpe:  median={np.median(sharpes):.2f}, "
          f"5%={np.percentile(sharpes, 5):.2f}, "
          f"95%={np.percentile(sharpes, 95):.2f}")
    print(f"\nP(Total R > 0) = {p_positive:.3f}")
    print(f"P(Sharpe > 1)  = {p_sharpe_gt1:.3f}")

    if p_positive > 0.95:
        verdict = "✅ УСТОЙЧИВА: >95% бутстрэп-выборок прибыльны"
    elif p_positive > 0.80:
        verdict = "🟡 УМЕРЕННАЯ: 80–95% выборок прибыльны"
    else:
        verdict = "❌ НЕУСТОЙЧИВА: менее 80% выборок прибыльны"
    print(f"\n>>> {verdict}")

    # График распределения
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(totals, bins=60, color="steelblue", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].axvline(np.median(totals), color="red", linestyle="--", label="median")
    axes[0].set_title("Bootstrap: Total R")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(sharpes, bins=60, color="seagreen", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].axvline(1, color="orange", linestyle="--", label="Sharpe=1")
    axes[1].axvline(np.median(sharpes), color="red", linestyle="--", label="median")
    axes[1].set_title("Bootstrap: Sharpe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("bootstrap.png", dpi=120)
    plt.show()

    return {"p_positive": p_positive, "p_sharpe_gt1": p_sharpe_gt1,
            "median_total": np.median(totals), "verdict": verdict}


# ============================================================
# 6. TEST 4 — РЕЖИМНЫЙ АНАЛИЗ
# ============================================================
def test_regime(trades):
    print("\n" + "=" * 100)
    print("TEST 4: разбивка по режиму волатильности")
    print("=" * 100)

    if "regime_at_entry" not in trades.columns:
        print("Нет regime_at_entry")
        return None

    t = trades.dropna(subset=["regime_at_entry"]).copy()
    t["regime_bin"] = pd.qcut(t["regime_at_entry"], 3,
                              labels=["low_vol", "mid_vol", "high_vol"])

    g = t.groupby("regime_bin", observed=True).agg(
        n=("pnl_r", "size"),
        total_r=("pnl_r", "sum"),
        avg_r=("pnl_r", "mean"),
        win_rate=("pnl_r", lambda x: (x > 0).mean()),
    )
    print("\n", g.to_string())

    if (g["total_r"] > 0).all():
        verdict = "✅ Прибыль есть во всех режимах волатильности"
    elif (g["total_r"] > 0).sum() >= 2:
        verdict = "🟡 Прибыль в 2 из 3 режимов — есть концентрация"
    else:
        verdict = "❌ Прибыль только в одном режиме — модель не универсальна"
    print(f"\n>>> {verdict}")
    return g, verdict


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_data("EURUSD_H1.csv")
    df = add_indicators(df)
    print(f"Баров: {len(df)}, период {df.index.min()} — {df.index.max()}\n")

    best_name, best_trades, combo_res = test_best_combo(df)
    loo, loo_verdict = test_loo_quarter(best_trades)
    boot = test_bootstrap(best_trades, n_iter=2000)
    regime = test_regime(best_trades)

    print("\n" + "=" * 100)
    print("ИТОГОВЫЙ ВЕРДИКТ")
    print("=" * 100)
    print(f"  Лучшая конфигурация:  {best_name}")
    print(f"  Total R:              {best_trades['pnl_r'].sum():.2f}")
    print(f"  Sharpe:               {metrics(best_trades)['sharpe']:.2f}")
    print(f"  Leave-one-out:        {loo_verdict}")
    print(f"  Bootstrap:            {boot['verdict']}")
    if regime is not None:
        print(f"  Regime:               {regime[1]}")

    # Сохранение
    best_trades.to_csv("final_trades.csv", index=False)
    combo_res.to_csv("final_combo.csv", index=False)
    loo.to_csv("final_loo.csv", index=False)
    print("\nСохранено: final_trades.csv, final_combo.csv, final_loo.csv")