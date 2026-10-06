#!/usr/bin/env python3
# ============================================================
# MT5 Data Downloader
# Скачивает котировки H1 из MetaTrader 5 и сохраняет в data/
# ============================================================

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import MetaTrader5 as mt5

# ============================================================
# НАСТРОЙКИ
# ============================================================
OUTPUT_DIR = "data"

# Список инструментов для скачивания.
# Имена должны точно совпадать с тем, как они называются у вашего брокера.
# Например: "EURUSD", "EURUSD.a", "EURUSDm", "EURUSD_i" и т.д.
SYMBOLS = [
    # Мажоры
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    # Кроссы
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "AUDJPY",
    "AUDNZD",
    "AUDCAD",
    "NZDJPY",
    "NZDCAD",
    "CADJPY",
    "CHFJPY",
    # Металлы и индексы (если брокер даёт)
    "XAUUSD",
    "XAGUSD",
]

# Таймфрейм и количество баров.
TIMEFRAME = mt5.TIMEFRAME_H1
BARS_COUNT = 100000  # запрашиваем с запасом

# ============================================================
# ПОДКЛЮЧЕНИЕ
# ============================================================
def connect_mt5():
    """Инициализация подключения к MetaTrader 5."""
    if not mt5.initialize():
        print(f"[ERROR] mt5.initialize() failed, error code = {mt5.last_error()}")
        return False

    info = mt5.terminal_info()
    if info is None:
        print(f"[ERROR] terminal_info() returned None, error = {mt5.last_error()}")
        mt5.shutdown()
        return False

    account = mt5.account_info()
    if account is None:
        print("[WARN] account_info() is None — возможно, вы не залогинены. "
              "Скачивание данных всё равно может работать.")
    else:
        print(f"[OK] Подключено к {info.name}")
        print(f"[OK] Аккаунт: {account.login}  ({account.server})  "
              f"balance={account.balance} {account.currency}")

    return True


# ============================================================
# СКАЧИВАНИЕ ОДНОГО ИНСТРУМЕНТА
# ============================================================
def download_symbol(symbol, bars=BARS_COUNT, timeframe=TIMEFRAME):
    """
    Скачивает бары и возвращает DataFrame в формате MT5 CSV.
    Возвращает None при ошибке.
    """
    # Проверяем, что символ доступен
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        print(f"  [SKIP] {symbol}: символ не найден в терминале")
        return None

    # Убеждаемся, что символ виден в Market Watch
    if not sym_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"  [WARN] {symbol}: не удалось добавить в Market Watch, "
                  f"продолжаем...")

    # Запрашиваем бары
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None:
        err = mt5.last_error()
        print(f"  [ERROR] {symbol}: copy_rates_from_pos вернул None, "
              f"error = {err}")
        return None

    if len(rates) == 0:
        print(f"  [WARN] {symbol}: 0 баров — возможно, история не загружена")
        return None

    df = pd.DataFrame(rates)

    # time в MT5 — это секунды от эпохи в UTC
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    # Формируем формат, как в MT5 CSV
    out = pd.DataFrame({
        "<DATE>":    df["time"].dt.strftime("%Y.%m.%d"),
        "<TIME>":    df["time"].dt.strftime("%H:%M:%S"),
        "<OPEN>":    df["open"].astype(float),
        "<HIGH>":    df["high"].astype(float),
        "<LOW>":     df["low"].astype(float),
        "<CLOSE>":   df["close"].astype(float),
        "<TICKVOL>": df["tick_volume"].astype(int),
        "<VOL>":     df.get("real_volume", pd.Series([0] * len(df))).astype(int),
        "<SPREAD>":  df["spread"].astype(int),
    })

    return out


# ============================================================
# СОХРАНЕНИЕ
# ============================================================
def save_to_csv(symbol, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{symbol}_H1.csv")

    df.to_csv(path, sep="\t", index=False)

    # Первая и последняя дата — для лога
    first = df["<DATE>"].iloc[0] + " " + df["<TIME>"].iloc[0]
    last  = df["<DATE>"].iloc[-1] + " " + df["<TIME>"].iloc[-1]
    return path, len(df), first, last


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("MT5 Data Downloader")
    print("=" * 70)

    if not connect_mt5():
        return

    try:
        total = len(SYMBOLS)
        results = []

        for i, symbol in enumerate(SYMBOLS, 1):
            print(f"\n[{i}/{total}] {symbol}")
            t0 = time.time()

            df = download_symbol(symbol)
            if df is None:
                results.append({
                    "symbol": symbol, "status": "fail",
                    "path": "", "bars": 0, "first": "", "last": "",
                })
                continue

            path, n, first, last = save_to_csv(symbol, df)
            elapsed = time.time() - t0

            print(f"  [OK] {n} баров  {first}  →  {last}  ({elapsed:.1f}s)")
            print(f"  [OK] сохранено: {path}")

            results.append({
                "symbol": symbol, "status": "ok",
                "path": path, "bars": n,
                "first": first, "last": last,
            })

        # Сводка
        print("\n" + "=" * 70)
        print("ИТОГ")
        print("=" * 70)
        summary = pd.DataFrame(results)
        print(summary.to_string(index=False))

        ok = (summary["status"] == "ok").sum()
        fail = (summary["status"] != "ok").sum()
        print(f"\nУспешно: {ok}  /  Провалено: {fail}")

        summary.to_csv(os.path.join(OUTPUT_DIR, "_download_summary.csv"),
                       index=False)

    finally:
        mt5.shutdown()
        print("\n[OK] mt5.shutdown() выполнен")


if __name__ == "__main__":
    main()