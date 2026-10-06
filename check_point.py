# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, ".")
from v17_basket8 import get_point as new_point, SYMS

def old_point(path):
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        line = f.readline()
    price = line.split("\t")[3].strip()
    dec = len(price.split(".")[1]) if "." in price else 0
    return 10 ** -dec, price

print("sym       old(HIGH 1st row)   raw       new(mode 100)   same?")
for s in SYMS:
    p = os.path.join("data", f"{s}_H1.csv")
    o, raw = old_point(p)
    n = new_point(p)
    flag = "OK" if o == n else "*** DIFF"
    print(f"{s:8s} {o:10.0e}   {raw:>12s}   {n:10.0e}   {flag}")