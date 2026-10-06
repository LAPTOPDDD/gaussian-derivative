# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def load(p):
    d = pd.read_csv(p, sep="\t")
    d.columns = [c.strip("<>\"' ").lower() for c in d.columns]
    d["dt"] = pd.to_datetime(d["date"] + " " + d["time"])
    return d.set_index("dt").sort_index()

old = load("EURUSD_H1.csv")
new = load("data/EURUSD_H1.csv")
new = new.loc["2024-12-27":"2026-09-09"]
common = old.index.intersection(new.index)
print("общих баров:", len(common), " старый всего:", len(old), " новый в окне:", len(new))
co = old.loc[common, "close"].values
cn = new.loc[common, "close"].values
print("совпадение close точно:", round((np.abs(co - cn) < 1e-12).mean(), 4))
print("макс |расхождение close|:", np.abs(co - cn).max())
print("средний спред старый:", round(old.loc[common, "spread"].mean(), 2),
      " новый:", round(new.loc[common, "spread"].mean(), 2))