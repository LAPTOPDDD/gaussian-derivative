# -*- coding: utf-8 -*-
# Сравнение get_point(мода по первым 100 строкам) с модой по строкам внутри окна.
import os
from collections import Counter

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "EURCAD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPCHF", "AUDJPY", "AUDNZD", "AUDCAD",
        "NZDJPY", "NZDCAD", "CADJPY", "CHFJPY", "XAUUSD", "XAGUSD"]

WINDOWS = {"W_recent": ("2024-12-27", "2026-09-09"),
           "W_early": ("2020-01-01", "2022-06-30")}

def point_window(path, t0, t1):
    decs = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            d = parts[0].strip()
            if d < t0 or d > t1:
                continue
            p = parts[5]
            decs.append(len(p.split(".")[1].strip()) if "." in p else 0)
    if not decs:
        return None, 0
    return Counter(decs).most_common(1)[0][0], len(decs)

def point_head(path, n=100):
    decs = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 6:
                continue
            p = parts[5]
            decs.append(len(p.split(".")[1].strip()) if "." in p else 0)
    return Counter(decs).most_common(1)[0][0]

for wname, (t0, t1) in WINDOWS.items():
    print(f"--- {wname} {t0}..{t1} ---")
    diffs = 0
    for s in SYMS:
        path = os.path.join("data", f"{s}_H1.csv")
        dh = point_head(path)
        dw, cnt = point_window(path, t0, t1)
        if dw is None:
            print(f"{s:8s} head={dh} window=EMPTY")
            continue
        if dw != dh:
            diffs += 1
            print(f"{s:8s} head={dh} window={dw} (n={cnt})  *** DIFF -> 1e-{dw} вместо 1e-{dh}")
    if diffs == 0:
        print("все 25 совпали")