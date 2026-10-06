#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("v17", "reactive_v17.py")
v17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v17)

_P = print
def print(*a, **k):
    k.setdefault("flush", True)
    _P(*a, **k)

syms = ["EURUSD", "GBPUSD", "USDJPY"]
dfs, evs = {}, {}
for s in syms:
    dfs[s], evs[s] = v17.load(s)

rng = np.random.default_rng(123)
start = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
null = []
for it in range(n):
    pool = 0.0
    for s in syms:
        ee = evs[s].copy()
        flips = rng.integers(0, 2, size=len(ee), dtype=np.int8).astype(float) * 2 - 1
        ee["direction"] = ee["direction"] * flips
        tr, *_ = v17.pipeline(dfs[s], ee, exit_rule="ev", size_mode="steep")
        pool += tr["pnl_r"].sum()
    null.append(pool)
    print(f"    perm-run {start+it}/{start+n-1}: Total={pool:+.2f}", flush=True)

old = np.load("v17_perm_totals.npy")
allv = np.concatenate([old, np.array(null)])
real = 69.70
print(f"\nвсего perm-прогонов: {len(allv)}", flush=True)
print(f"    реальный портфель EV_STEEP = {real:+.2f}", flush=True)
print(f"    P(perm >= real) = {(allv >= real).mean():.3f}", flush=True)
print(f"    null: 5%={np.percentile(allv,5):+.2f} med={np.median(allv):+.2f} "
      f"95%={np.percentile(allv,95):+.2f} max={allv.max():+.2f} min={allv.min():+.2f}", flush=True)
np.save("v17_perm_totals.npy", allv)