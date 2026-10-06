#!/usr/bin/env python3
# ============================================================
# BATCH RUNNER — параллельный прогон конфигурации F
# Весь вывод пишется в один файл: batch_results/report.txt
# ============================================================

import os
import glob
import time
import warnings
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================
# НАСТРОЙКИ
# ============================================================
DATA_DIR = "data"
RESULTS_DIR = "batch_results"
REPORT_FILE = os.path.join(RESULTS_DIR, "report.txt")

# --- Конфигурация F (не менять!) ---
RR = 2.0
THRESHOLD = 0.5
TRAIN_FRAC = 0.4
MAX_BARS = 36
HORIZON = 12
LOOKBACK = 20
STOP_MULT = 1.5
N_BOOTSTRAP = 1000
MIN_BARS = 5000

# --- Вычислительные параметры ---
STEP = 250                  # компромисс между скоростью и стабильностью
N_ESTIMATORS = 200          # было 300

# --- Отсечение по времени ---
CUTOFF_DATE = pd.Timestamp("2020-01-01")   # naive datetime

# --- Параллелизм ---
N_WORKERS = 8               # число параллельных процессов (можно 4, 6, 8)

RNG_SEED = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# БУФЕР ОТЧЁТА (только для main-процесса)
# ============================================================
_report_lines = []

def log(msg=""):
    print(msg)
    _report_lines.append(str(msg))

def flush_report():
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_report_lines))
    print(f"\n[OK] Отчёт сохранён: {REPORT_FILE}")


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

    # Отсечение по времени (naive vs naive)
    if CUTOFF_DATE is not None:
        cutoff = pd.Timestamp(CUTOFF_DATE)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if cutoff.tz is not None:
            cutoff = cutoff.tz_localize(None)
        df = df[df.index >= cutoff]

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
        })
        i = exit_idx + 1

    return pd.DataFrame(events)


def build_states(df, events, horizon=HORIZON):
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n_bars = len(df)

    ret_cols = {n: df[f"ret_{n}"].values for n in (1, 2, 3, 5)}
    streak_arr = df["streak"].values
    atr14_arr = df["atr14"].values
    regime_arr = df["regime"].values
    pir20 = df["pos_in_range_20"].values
    pir50 = df["pos_in_range_50"].values
    pir100 = df["pos_in_range_100"].values
    tvol = df["tickvol_rel"].values
    srel = df["spread_rel"].values
    sprd = df["spread"].values
    hour_arr = df["hour"].values
    dow_arr = df["dow"].values
    times = df.index

    states = []

    for ev in events.itertuples(index=False):
        e_idx = ev.entry_idx
        x_idx = ev.exit_idx
        direction = ev.direction
        entry_price = ev.entry_price
        R = ev.R
        stop_price = ev.stop_price
        target_price = ev.target_price

        mfe = 0.0
        mae = 0.0
        max_fav_idx = e_idx
        last_stop_touch = -1

        for j in range(e_idx, x_idx + 1):
            h = highs[j]
            l = lows[j]

            if direction == 1:
                new_mfe = (h - entry_price) / R
                mae = min(mae, (l - entry_price) / R)
                if l <= stop_price:
                    last_stop_touch = j
            else:
                new_mfe = (entry_price - l) / R
                mae = min(mae, (entry_price - h) / R)
                if h >= stop_price:
                    last_stop_touch = j

            if new_mfe > mfe:
                mfe = new_mfe
                max_fav_idx = j

            bars_in_pos = j - e_idx + 1

            # forward-looking target
            target = 0
            end_k = min(j + 1 + horizon, n_bars)
            for k in range(j + 1, end_k):
                if direction == 1:
                    hit_t = highs[k] >= target_price
                    hit_s = lows[k] <= stop_price
                else:
                    hit_t = lows[k] <= target_price
                    hit_s = highs[k] >= stop_price
                if hit_s:
                    break
                if hit_t:
                    target = 1
                    break

            close_j = closes[j]
            pnl_now = direction * (close_j - entry_price) / R

            states.append({
                "bar_idx": j, "bar_time": times[j], "entry_idx": e_idx,
                "direction": direction, "entry_price": entry_price, "R": R,
                "bars_in_pos": bars_in_pos, "pnl_r": pnl_now,
                "mfe_r": mfe, "mae_r": mae,
                "bars_since_mfe": j - max_fav_idx,
                "bars_since_stop_touch": (j - last_stop_touch) if last_stop_touch >= 0 else 999,
                "ret_1": ret_cols[1][j], "ret_2": ret_cols[2][j],
                "ret_3": ret_cols[3][j], "ret_5": ret_cols[5][j],
                "streak": streak_arr[j],
                "atr_ratio": atr14_arr[j] / R * 1.5 if R > 0 else np.nan,
                "regime": regime_arr[j],
                "pos_in_range_20": pir20[j],
                "pos_in_range_50": pir50[j],
                "pos_in_range_100": pir100[j],
                "tickvol_rel": tvol[j],
                "spread_rel": srel[j],
                "spread": sprd[j],
                "hour": hour_arr[j], "dow": dow_arr[j],
                "target": target,
                "actual_exit_idx": x_idx,
                "actual_exit_reason": ev.exit_reason_actual,
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

    model_trades = []
    base_trades = []
    aucs = []

    closes_arr = df["close"].values
    spreads_arr = df["spread_price"].values

    while split < n:
        test_end = min(split + step, n)
        train = states.iloc[start:split]
        test = states.iloc[split:test_end]
        if len(train) < 200 or len(test) < 50:
            split = test_end
            continue

        model = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(train[features].values, train["target"].values)

        if test["target"].nunique() > 1:
            p_test = model.predict_proba(test[features].values)[:, 1]
            aucs.append(roc_auc_score(test["target"], p_test))

        for e_idx in test["entry_idx"].unique():
            pos = states[states["entry_idx"] == e_idx]
            if len(pos) == 0:
                continue

            direction = pos["direction"].iloc[0]
            entry_price = pos["entry_price"].iloc[0]
            R = pos["R"].iloc[0]
            actual_exit_idx = int(pos["actual_exit_idx"].iloc[0])
            entry_spread = spreads_arr[int(e_idx)]

            X = pos[features].values
            P = model.predict_proba(X)[:, 1]

            # baseline
            exit_close = closes_arr[actual_exit_idx]
            exit_spread = spreads_arr[actual_exit_idx]
            base_pnl = direction * (exit_close - entry_price) / R
            base_pnl -= (entry_spread + exit_spread) / R
            base_trades.append(base_pnl)

            # model
            exit_idx = None
            exit_reason = None
            bars_held = 1
            pos_arrays = pos[["bar_idx", "bars_in_pos"]].values
            for k in range(len(pos_arrays)):
                p = P[k]
                b_idx = int(pos_arrays[k, 0])
                bip = pos_arrays[k, 1]
                if bip >= max_bars:
                    exit_idx, exit_reason, bars_held = b_idx, "timeout", int(bip)
                    break
                if p < threshold:
                    exit_idx, exit_reason, bars_held = b_idx, "prob_exit", int(bip)
                    break

            if exit_idx is None:
                exit_idx, exit_reason = actual_exit_idx, "actual"
                bars_held = exit_idx - int(e_idx) + 1

            exit_close = closes_arr[exit_idx]
            exit_spread = spreads_arr[exit_idx]
            pnl_raw = direction * (exit_close - entry_price) / R
            pnl_raw -= (entry_spread + exit_spread) / R

            p_entry = float(P[0])
            size = float(np.clip(0.5 + (p_entry - 0.2) / 0.6, 0.5, 1.5))

            model_trades.append((
                pnl_raw * size, size, p_entry, exit_reason,
                bars_held, direction, float(pos["regime"].iloc[0])
            ))

        split = test_end

    if not model_trades:
        return None, None, np.nan

    mdf = pd.DataFrame(model_trades, columns=[
        "pnl_r", "size", "p_entry", "exit_reason",
        "bars_held", "direction", "regime_at_entry"
    ])
    bdf = pd.DataFrame({"pnl_r": base_trades})
    return mdf, bdf, float(np.mean(aucs)) if aucs else np.nan


# ============================================================
# МЕТРИКИ
# ============================================================
def metrics(pnl_series):
    if len(pnl_series) == 0:
        return {"n": 0, "total_r": 0.0, "avg_r": 0.0, "win_rate": 0.0,
                "sharpe": 0.0, "max_dd": 0.0, "pf": 0.0}
    eq = pnl_series.cumsum().values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    gains = pnl_series[pnl_series > 0].sum()
    losses = -pnl_series[pnl_series < 0].sum()
    return {
        "n": int(len(pnl_series)),
        "total_r": float(pnl_series.sum()),
        "avg_r": float(pnl_series.mean()),
        "win_rate": float((pnl_series > 0).mean()),
        "sharpe": float(pnl_series.mean() /
                        (pnl_series.std() + 1e-9) * np.sqrt(len(pnl_series))),
        "max_dd": float(dd.min()),
        "pf": float(gains / losses) if losses > 0 else float("inf"),
    }


def bootstrap_stats(pnls, n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    if len(pnls) < 30:
        return {"p_pos": 0.0, "p_sh1": 0.0}
    rng = np.random.default_rng(seed)
    pnls = np.asarray(pnls)
    n = len(pnls)
    totals = np.empty(n_iter)
    sharpes = np.empty(n_iter)
    for i in range(n_iter):
        s = rng.choice(pnls, size=n, replace=True)
        totals[i] = s.sum()
        sharpes[i] = s.mean() / (s.std() + 1e-9) * np.sqrt(n)
    return {
        "p_pos": float((totals > 0).mean()),
        "p_sh1": float((sharpes > 1).mean()),
    }


# ============================================================
# ОДИН ИНСТРУМЕНТ
# ============================================================
def run_instrument(path):
    name = os.path.splitext(os.path.basename(path))[0].replace("_H1", "")

    try:
        df = load_data(path)
    except Exception as e:
        return {"instrument": name, "status": f"load_error: {e}"}

    if len(df) < MIN_BARS:
        return {"instrument": name, "status": f"too_few_bars: {len(df)}"}

    df = add_indicators(df)
    events = simulate_positions(df)
    if len(events) < 30:
        return {"instrument": name, "status": f"too_few_events: {len(events)}"}

    states = build_states(df, events)
    if len(states) < 500:
        return {"instrument": name, "status": f"too_few_states: {len(states)}"}

    mdf, bdf, auc = walk_forward(df, states, FEATURES)
    if mdf is None or len(mdf) < 30:
        return {"instrument": name, "status":
                f"too_few_trades: {len(mdf) if mdf is not None else 0}"}

    m = metrics(mdf["pnl_r"])
    b = metrics(bdf["pnl_r"])
    bs = bootstrap_stats(mdf["pnl_r"].values)

    # leave-one-block-out (4 блока)
    if len(mdf) >= 40:
        block_id = np.zeros(len(mdf), dtype=int)
        chunks = np.array_split(np.arange(len(mdf)), 4)
        for i, idx in enumerate(chunks):
            block_id[idx] = i
        mdf["block"] = block_id
        loo_totals = []
        for b_i in range(4):
            sub = mdf[mdf["block"] != b_i]
            loo_totals.append(sub["pnl_r"].sum())
        loo_min = float(min(loo_totals))
    else:
        loo_min = float("nan")

    # режимная устойчивость
    t_reg = mdf.dropna(subset=["regime_at_entry"])
    if len(t_reg) >= 30:
        try:
            t_reg = t_reg.copy()
            t_reg["regime_bin"] = pd.qcut(t_reg["regime_at_entry"], 3,
                                          labels=["low", "mid", "high"],
                                          duplicates="drop")
            regime_pnl = t_reg.groupby("regime_bin", observed=True)["pnl_r"].sum()
            regime_ok = bool((regime_pnl > 0).all())
        except Exception:
            regime_ok = False
    else:
        regime_ok = False

    return {
        "instrument": name, "status": "ok",
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
        "avg_bars_held": float(mdf["bars_held"].mean()),
        "avg_size": float(mdf["size"].mean()),
        "base_total_r": b["total_r"],
        "base_sharpe": b["sharpe"],
        "base_max_dd": b["max_dd"],
        "delta_vs_base": m["total_r"] - b["total_r"],
        "boot_p_pos": bs["p_pos"],
        "boot_p_sh1": bs["p_sh1"],
        "loo_min": loo_min,
        "regime_ok": regime_ok,
    }


# ============================================================
# WORKER ДЛЯ ProcessPoolExecutor
# ============================================================
def _worker(args):
    i, total, path = args
    name = os.path.basename(path).replace("_H1.csv", "")
    t0 = time.time()
    try:
        r = run_instrument(path)
    except Exception as e:
        r = {"instrument": name, "status": f"unhandled: {e}"}
    r["elapsed_sec"] = time.time() - t0
    r["_idx"] = i
    r["_name"] = name
    return r


# ============================================================
# MAIN — параллельный запуск
# ============================================================
def main():
    t_start = time.time()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]

    if not files:
        print(f"Не найдено CSV в '{DATA_DIR}'")
        return

    n_workers = min(N_WORKERS, len(files))

    log("=" * 100)
    log("BATCH RUN — конфигурация F (параллельно)")
    log("=" * 100)
    log(f"Запуск:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Инструментов: {len(files)}")
    log(f"Параллельно:  {n_workers} процессов")
    log(f"Конфигурация: rr={RR}, step={STEP}, threshold={THRESHOLD}")
    log(f"Отсечение:    {CUTOFF_DATE.date() if CUTOFF_DATE is not None else 'нет'}")
    log(f"Bootstrap:    {N_BOOTSTRAP} итераций")
    log(f"Trees:        {N_ESTIMATORS}")
    log("")

    tasks = [(i, len(files), path) for i, path in enumerate(files, 1)]
    rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            r = fut.result()
            rows.append(r)
            status = r.get("status", "unknown")
            if status == "ok":
                log(f"[{completed}/{len(files)}] {r['_name']:8s}  "
                    f"OK  n={r['n_trades']:4d}  "
                    f"total_r={r['total_r']:+7.1f}  "
                    f"sharpe={r['sharpe']:+6.2f}  "
                    f"({r['elapsed_sec']:.0f}s)")
            else:
                log(f"[{completed}/{len(files)}] {r['_name']:8s}  "
                    f"FAIL  {status}  ({r['elapsed_sec']:.0f}s)")

    df_res = pd.DataFrame(rows)
    total_time = time.time() - t_start

    ok = df_res[df_res["status"] == "ok"].copy()
    failed = df_res[df_res["status"] != "ok"].copy()

    if len(ok) == 0:
        log("\nНи один инструмент не отработал.")
        flush_report()
        return

    # Скоринг устойчивости
    ok["robust_score"] = (
        ok["sharpe"].clip(lower=0) *
        ok["boot_p_pos"] *
        ok["boot_p_sh1"] *
        (ok["loo_min"].fillna(0) > 0).astype(int)
    )
    ok = ok.sort_values("robust_score", ascending=False).reset_index(drop=True)

    # ============================================================
    # ОТЧЁТ
    # ============================================================
    log("")
    log("=" * 100)
    log("СВОДНАЯ ТАБЛИЦА — все инструменты, отсортировано по robust_score")
    log("=" * 100)

    cols_show = [
        "instrument", "n_trades", "total_r", "sharpe", "max_dd", "win_rate",
        "auc", "avg_bars_held", "avg_size",
        "base_total_r", "delta_vs_base",
        "boot_p_pos", "boot_p_sh1", "loo_min", "regime_ok", "robust_score",
    ]

    ok_fmt = ok[cols_show].copy()
    for c in ["total_r", "sharpe", "max_dd", "base_total_r",
              "delta_vs_base", "loo_min"]:
        ok_fmt[c] = ok_fmt[c].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
    for c in ["win_rate", "boot_p_pos", "boot_p_sh1", "robust_score", "auc"]:
        ok_fmt[c] = ok_fmt[c].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    for c in ["avg_bars_held", "avg_size"]:
        ok_fmt[c] = ok_fmt[c].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")

    log(ok_fmt.to_string(index=False))

    if len(failed) > 0:
        log("")
        log("=" * 100)
        log("НЕ ОТРАБОТАЛИ")
        log("=" * 100)
        log(failed[["instrument", "status"]].to_string(index=False))

    # Устойчивые
    robust = ok[(ok["sharpe"] > 1) &
                (ok["boot_p_pos"] > 0.95) &
                (ok["loo_min"].fillna(-999) > 0)].copy()

    log("")
    log("=" * 100)
    log("УСТОЙЧИВЫЕ (Sharpe>1 + bootstrap>0.95 + leave-one-out>0)")
    log("=" * 100)
    if len(robust) == 0:
        log("Нет инструментов, прошедших все три критерия.")
    else:
        log(f"Всего: {len(robust)} из {len(ok)}")
        log("")
        robust_fmt = robust[cols_show].copy()
        for c in ["total_r", "sharpe", "max_dd", "base_total_r",
                  "delta_vs_base", "loo_min"]:
            robust_fmt[c] = robust_fmt[c].apply(
                lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
        for c in ["win_rate", "boot_p_pos", "boot_p_sh1",
                  "robust_score", "auc"]:
            robust_fmt[c] = robust_fmt[c].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
        log(robust_fmt.to_string(index=False))

    # ТОП-5 по Sharpe
    log("")
    log("=" * 100)
    log("ТОП-5 по Sharpe")
    log("=" * 100)
    top_sharpe = ok.nlargest(5, "sharpe")[
        ["instrument", "n_trades", "total_r", "sharpe", "max_dd", "win_rate"]
    ].copy()
    for c in ["total_r", "sharpe", "max_dd"]:
        top_sharpe[c] = top_sharpe[c].apply(lambda v: f"{v:+.2f}")
    for c in ["win_rate"]:
        top_sharpe[c] = top_sharpe[c].apply(lambda v: f"{v:.3f}")
    log(top_sharpe.to_string(index=False))

    # Итог
    log("")
    log("=" * 100)
    log("ИТОГ")
    log("=" * 100)
    log(f"Всего инструментов:  {len(df_res)}")
    log(f"Успешно:             {len(ok)}")
    log(f"Провалено:           {len(failed)}")
    log(f"Устойчивых:          {len(robust)}")
    log(f"Время:               {total_time:.1f} с  ({total_time/60:.1f} мин)")

    if len(ok) > 0:
        log("")
        log("Средние метрики по успешным:")
        log(f"  Sharpe:            {ok['sharpe'].mean():.2f}")
        log(f"  Total R:           {ok['total_r'].mean():+.2f}")
        log(f"  Baseline Total R:  {ok['base_total_r'].mean():+.2f}")
        log(f"  Delta vs baseline: {ok['delta_vs_base'].mean():+.2f}")
        log(f"  Средний AUC:       {ok['auc'].mean():.3f}")

    log("")
    log("=" * 100)
    log("КОНЕЦ ОТЧЁТА")
    log("=" * 100)

    flush_report()


if __name__ == "__main__":
    mp.freeze_support()   # важно для Windows
    main()