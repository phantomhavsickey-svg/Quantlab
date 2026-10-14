# -*- coding: utf-8 -*-
"""新面板 vs backup_20261009：新增 3 天各有多少行，历史不等格子差多大。

    python research/probe_newdays.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

OLD = "data/cache/backup_20261009/factor_panel.parquet"
NEW = "data/cache/factor_panel.parquet"

o = pd.read_parquet(OLD)
n = pd.read_parquet(NEW)
for d in (o, n):
    d["date"] = pd.to_datetime(d["date"])
cols = [c for c in n.columns if c not in ("date", "symbol")]

print("### 1. 新面板多出来的那几天，各有多少行 / 多少只股票")
cut = o["date"].max()
tail = n[n["date"] > cut]
for dt, g in tail.groupby("date"):
    nz = g[cols].notna().sum(axis=1)
    print("  %s  行 %-6d 股票 %-6d 只 | 25 列全空的行 %d | 每行非空列数中位 %.0f"
          % (dt.date(), len(g), g["symbol"].nunique(), int((nz == 0).sum()), nz.median()))
print("  对照：旧面板最后一天 %s 有 %d 行 %d 只"
      % (cut.date(), int((o["date"] == cut).sum()), o[o["date"] == cut]["symbol"].nunique()))

print()
print("### 2. 历史那部分不等格子的差到底多大（单位：z 分数）")
j = n.merge(o, on=["date", "symbol"], suffixes=("_n", "_o"))
j = j[j["date"] <= cut]
d_all = np.concatenate([(j[c + "_n"].to_numpy() - j[c + "_o"].to_numpy())[
    np.isfinite(j[c + "_n"].to_numpy()) & np.isfinite(j[c + "_o"].to_numpy())]
    for c in cols])
a = np.abs(d_all)
print("  比较的格子 %d 个 | 完全相等 %d | 不等 %d (%.4f%%)"
      % (a.size, int((a == 0).sum()), int((a > 0).sum()), (a > 0).mean() * 100))
for t in (1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    print("    |Δ| > %-8.0e 的格子 %-9d 占 %.6f%%" % (t, int((a > t).sum()), (a > t).mean() * 100))
print("  不等格子里：最大 |Δ| = %.3e，中位 |Δ| = %.3e" % (a.max(), np.median(a[a > 0])))
print("  z 分数本身的尺度（新面板）：标准差中位 %.4f"
      % float(np.median([n[c].std() for c in cols])))

print()
print("### 3. 行情缓存：1000 只里各自的最后一根 K 线落在哪天")
from data.cache import CacheManager
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cm = CacheManager(cfg["cache"]["directory"])
syms = cm.list_cached_symbols()
last = {}
per_day = {pd.Timestamp(x): 0 for x in ("2026-09-30", "2026-10-08", "2026-10-09")}
for s in syms:
    d = cm.get_daily(s)
    if d is None or len(d) == 0:
        continue
    idx = pd.to_datetime(d["日期"]).sort_values()
    last[idx.iloc[-1]] = last.get(idx.iloc[-1], 0) + 1
    have = set(idx.iloc[-5:])
    for k in per_day:
        if k in have:
            per_day[k] += 1
print("  缓存里共 %d 只" % len(syms))
print("  最后一根 K 线的日期分布（只列最后 5 档）:")
for dt in sorted(last)[-5:]:
    print("    %s  %d 只" % (dt.date(), last[dt]))
print("  含有该日 K 线的股票数:")
for k in sorted(per_day):
    print("    %s  %d 只" % (k.date(), per_day[k]))
