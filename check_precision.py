# -*- coding: utf-8 -*-
# Проверка: как меняется точность (знаков после запятой) по эпохам в файле,
# и объясняются ли расхождения old/new только размером пункта.
import os
import numpy as np
import pandas as pd
from collections import Counter

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD",
        "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "XAUUSD", "XAGUSD"]

def dec_series(path, col=5, n=None):
    decs = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            p = parts[col]
            decs.append(len(p.split(".")[1].strip()) if "." in p else 0)
    return np.array(decs)

print("=== точность первые 100 vs последние 1000 строк (CLOSE) ===")
print("sym       first100 mode   last1000 mode   changed?")
for s in SYMS:
    path = os.path.join("data", f"{s}_H1.csv")
    d1 = Counter(dec_series(path, n=100)).most_common(1)[0][0]
    # последние 1000: пропускаем всё кроме последних — через pandas быстрее
    df = pd.read_csv(path, sep="\t", usecols=["<CLOSE>"])
    tail = df["<CLOSE>"].tail(1000)
    dt = tail.map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
    d2 = Counter(dt).most_common(1)[0][0]
    flag = "" if d1 == d2 else "  *** CHANGED"
    print(f"{s:8s} {d1:8d}       {d2:8d}{flag}")

print()
print("=== XAGUSD: распределение знаков в первых 100 и в окнах теста ===")
path = os.path.join("data", "XAGUSD_H1.csv")
d_head = Counter(dec_series(path, n=100))
print("первые 100 (2009):", dict(sorted(d_head.items())))
df = pd.read_csv(path, sep="\t", usecols=["<DATE>", "<CLOSE>"])
for w, (a, b) in {"2020-2022": ("2020-01-01", "2022-06-30"),
                  "2024-2026": ("2024-12-27", "2026-09-09")}.items():
    seg = df[(df["<DATE>"] >= a) & (df["<DATE>"] <= b)]["<CLOSE>"]
    dd = seg.map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
    print(f"{w}: {dict(sorted(Counter(dd).most_common(5)))}")