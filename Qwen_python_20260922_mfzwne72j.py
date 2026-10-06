# ============================================================
# EURUSD H1 — Robustness check
# 5 тестов: годы/кварталы, шаг WF, k-NN, RR, режим выхода
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


# ============================================================
# 1. БАЗОВЫЕ ФУНКЦИИ (те же, что в v2)
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


def compute_knn_feature(states, features, k=30, block=200):
    data = states[features].values
    targets = states["target"].values
    n = len(data)
    knn = np.full(n, np.nan)
    mu = np.nanmean(data, axis=0)
    sigma = np.nanstd(data, axis=0) + 1e-9
    X = np.nan_to_num((data - mu) / sigma)

    for start in range(k, n, block):
        end = min(start + block, n)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        nn.fit(X[:start])
        _, idxs = nn.kneighbors(X[start:end])
        for ii, i in enumerate(range(start, end)):
            knn[i] = targets[idxs[ii]].mean()
    return knn


# ============================================================
# 2. КЭШ СОСТОЯНИЙ ПО RR
# ============================================================
_STATE_CACHE = {}

def get_states(df, rr):
    key = f"rr_{rr}"
    if key not in _STATE_CACHE:
        print(f"  [cache] строим states для rr={rr}")
        events = simulate_positions(df, lookback=20, stop_mult=1.5, rr=rr, max_bars=48)
        states = build_states(df, events, horizon=12)

        BASE_FEATURES = [
            "bars_in_pos", "pnl_r", "mfe_r", "mae_r",
            "bars_since_mfe", "bars_since_stop_touch",
            "ret_1", "ret_2", "ret_3", "ret_5", "streak",
            "atr_ratio", "regime",
            "pos_in_range_20", "pos_in_range_50", "pos_in_range_100",
            "tickvol_rel", "spread_rel", "spread",
            "hour", "dow",
        ]
        print(f"  [cache] считаем k-NN feature для rr={rr}...")
        states["knn_score"] = compute_knn_feature(states, BASE_FEATURES, k=30, block=200)
        _STATE_CACHE[key] = (events, states, BASE_FEATURES)
    return _STATE_CACHE[key]


# ============================================================
# 3. WALK-FORWARD С ГИБКИМИ ПАРАМЕТРАМИ
# ============================================================
def walk_forward(df, states, features,
                 rr=2.5, train_frac=0.4, step=500,
                 max_bars=36, exit_mode="ev", prob_threshold=0.5):
    """
    exit_mode:
      'ev'   — выход когда EV_hold < текущий PnL
      'prob' — выход когда P < prob_threshold
    """
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

            # --- baseline
            exit_close = df["close"].values[actual_exit_idx]
            exit_spread = df["spread_price"].values[actual_exit_idx]
            base_pnl = direction * (exit_close - entry_price) / R
            base_pnl -= (entry_spread + exit_spread) / R
            base_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[actual_exit_idx],
                "pnl_r": base_pnl, "fold": fold,
            })

            # --- model
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
                else:  # prob
                    if p < prob_threshold:
                        exit_idx, exit_reason = int(row["bar_idx"]), "prob_exit"
                        break

            if exit_idx is None:
                exit_idx, exit_reason = actual_exit_idx, "actual"

            exit_close = df["close"].values[exit_idx]
            exit_spread = df["spread_price"].values[exit_idx]
            pnl_raw = direction * (exit_close - entry_price) / R
            pnl_raw -= (entry_spread + exit_spread) / R

            p_entry = P[0]
            size = float(np.clip(0.5 + (p_entry - 0.2) / 0.6, 0.5, 1.5))

            model_trades.append({
                "entry_time": pos["bar_time"].iloc[0],
                "exit_time": df.index[exit_idx],
                "pnl_r": pnl_raw * size, "pnl_r_raw": pnl_raw,
                "size": size, "p_entry": p_entry,
                "exit_reason": exit_reason, "fold": fold,
            })

        fold += 1
        split = test_end

    model_trades = pd.DataFrame(model_trades)
    base_trades = pd.DataFrame(base_trades)
    if len(model_trades) > 0:
        model_trades.attrs["auc_mean"] = float(np.mean(aucs)) if aucs else np.nan
        model_trades.attrs["auc_std"] = float(np.std(aucs)) if aucs else np.nan
    return model_trades, base_trades


# ============================================================
# 4. МЕТРИКИ
# ============================================================
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
# 5. ЭКСПЕРИМЕНТ ПО ОДНОМУ КОНФИГУ
# ============================================================
def run_experiment(df, rr, use_knn, exit_mode, step, max_bars=36):
    events, states, base_features = get_states(df, rr)
    features = base_features + (["knn_score"] if use_knn else [])

    m_trades, b_trades = walk_forward(
        df, states, features,
        rr=rr, train_frac=0.4, step=step,
        max_bars=max_bars, exit_mode=exit_mode,
    )
    m = metrics(m_trades)
    b = metrics(b_trades)
    m["auc_mean"] = m_trades.attrs.get("auc_mean", np.nan) if len(m_trades) > 0 else np.nan
    m["auc_std"] = m_trades.attrs.get("auc_std", np.nan) if len(m_trades) > 0 else np.nan
    return m, b, m_trades, b_trades


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    df = load_data("EURUSD_H1.csv")
    df = add_indicators(df)
    print(f"Баров: {len(df)}  Период: {df.index.min()} — {df.index.max()}\n")

    # ---- Набор конфигов ----
    configs = [
        # name,                 rr,   knn,   exit_mode, step
        ("base (rr2.5,knn+,EV)",   2.5, True,  "ev",   500),
        ("no knn",                 2.5, False, "ev",   500),
        ("prob exit",              2.5, True,  "prob", 500),
        ("no knn + prob exit",     2.5, False, "prob", 500),
        ("step 250",               2.5, True,  "ev",   250),
        ("step 1000",              2.5, True,  "ev",   1000),
        ("rr 2.0",                 2.0, True,  "ev",   500),
        ("rr 3.0",                 3.0, True,  "ev",   500),
    ]

    rows = []
    detail = {}

    for name, rr, knn, em, step in configs:
        print(f"\n>>> Эксперимент: {name}")
        m, b, m_trades, b_trades = run_experiment(df, rr, knn, em, step)
        rows.append({
            "config": name, "rr": rr, "knn": knn, "exit": em, "step": step,
            "n": m["n"], "total_r": m["total_r"], "avg_r": m["avg_r"],
            "win_rate": m["win_rate"], "sharpe": m["sharpe"],
            "max_dd": m["max_dd"], "pf": m["pf"],
            "auc": m["auc_mean"], "auc_std": m["auc_std"],
            "base_total_r": b["total_r"], "base_max_dd": b["max_dd"],
        })
        detail[name] = {"model": m_trades, "base": b_trades}

    results = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 120)
    print(results[[
        "config", "n", "total_r", "avg_r", "win_rate",
        "sharpe", "max_dd", "pf", "auc", "auc_std",
        "base_total_r", "base_max_dd"
    ]].to_string(index=False))

    results.to_csv("robustness_results.csv", index=False)

    # ============================================================
    # 7. РАЗБИВКА BASE-МОДЕЛИ ПО КВАРТАЛАМ
    # ============================================================
    base_model = detail["base (rr2.5,knn+,EV)"]["model"]
    if len(base_model) > 0:
        base_model = base_model.copy()
        base_model["exit_time"] = pd.to_datetime(base_model["exit_time"])
        base_model["quarter"] = base_model["exit_time"].dt.to_period("Q")

        q = base_model.groupby("quarter").agg(
            n=("pnl_r", "size"),
            total_r=("pnl_r", "sum"),
            avg_r=("pnl_r", "mean"),
            win_rate=("pnl_r", lambda x: (x > 0).mean()),
        )
        print("\n" + "=" * 120)
        print("BASE-МОДЕЛЬ: разбивка по кварталам")
        print("=" * 120)
        print(q.to_string())

        # ---- График ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].bar(q.index.astype(str), q["total_r"],
                    color=["steelblue" if x > 0 else "firebrick" for x in q["total_r"]])
        axes[0].set_title("Total R по кварталам (base model)")
        axes[0].axhline(0, color="black", linewidth=0.7)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Сравнение конфигов по total_r
        cfg_labels = results["config"].values
        cfg_r = results["total_r"].values
        cfg_base = results["base_total_r"].values
        x = np.arange(len(cfg_labels))
        w = 0.38
        axes[1].bar(x - w/2, cfg_r, w, label="Model", color="steelblue")
        axes[1].bar(x + w/2, cfg_base, w, label="Baseline", color="gray", alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cfg_labels, rotation=30, ha="right")
        axes[1].set_title("Total R: конфиги vs baseline")
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("robustness.png", dpi=120)
        plt.show()

    print("\nСохранено: robustness_results.csv, robustness.png")