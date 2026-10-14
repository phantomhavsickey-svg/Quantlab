# -*- coding: utf-8 -*-
"""旧面板 vs 新面板：同一天的因子值，是"整列缩放"还是"次序也变了"。

    python research/panel_rank_shift.py

缩放 = 只有当天的股票池/归一化基准变了（原始价格没动）。
次序变 = 原始输入被改写过（前复权按新拉取日重新定基准）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

OLD = sys.argv[1] if len(sys.argv) > 1 else "data/cache/backup_20260930/factor_panel.parquet"
NEW = "data/cache/factor_panel.parquet"

o = pd.read_parquet(OLD)
n = pd.read_parquet(NEW)
for d in (o, n):
    d["date"] = pd.to_datetime(d["date"])
cols = [c for c in n.columns if c not in ("date", "symbol")]

print("### 1. 覆盖面")
print("  旧 %d 行 %d 列 | 新 %d 行 %d 列" % (len(o), o.shape[1], len(n), n.shape[1]))
print("  旧日期 %s ~ %s | 新 %s ~ %s"
      % (o["date"].min().date(), o["date"].max().date(),
         n["date"].min().date(), n["date"].max().date()))
print("  旧名字 %d 只 | 新名字 %d 只 | 只在新里的 %d 只"
      % (o["symbol"].nunique(), n["symbol"].nunique(),
         len(set(n["symbol"]) - set(o["symbol"]))))

j = o.merge(n, on=["date", "symbol"], suffixes=("_o", "_n"))
print("  能配对的历史行 %d 条" % len(j))

print()
print("### 2. 逐日：同一只股票的绝对差 / 当天截面内的排序是否变了")
rows, RANKS = [], {}
for c in cols:
    d1 = (j[c + "_n"] - j[c + "_o"]).abs()
    rs = []
    for _, g in j.groupby("date", sort=False):
        a = g[c + "_o"].to_numpy(float)
        b = g[c + "_n"].to_numpy(float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 8:
            continue
        ra = pd.Series(a[ok]).rank().to_numpy()
        rb = pd.Series(b[ok]).rank().to_numpy()
        rs.append(np.corrcoef(ra, rb)[0, 1])
    rs = np.array([v for v in rs if not np.isnan(v)], float)
    RANKS[c] = rs
    rows.append((c, float(d1.median()), float(d1.max()), len(rs),
                 float(np.median(rs)), float(rs.min())))
tab = pd.DataFrame(rows, columns=["因子", "绝对差中位", "绝对差最大",
                                  "有数天数", "当日排序相关中位", "当日排序相关最小"])
tab = tab.sort_values("当日排序相关中位")
print("  %-22s %10s %10s %8s %14s %14s" % tuple(tab.columns))
for _, r in tab.iterrows():
    print("  %-22s %10.4f %10.4f %8d %14.6f %14.6f"
          % (r["因子"], r["绝对差中位"], r["绝对差最大"], r["有数天数"],
             r["当日排序相关中位"], r["当日排序相关最小"]))
print()
allv = np.array([v for c in cols for v in RANKS[c]], float)
allv = allv[~np.isnan(allv)]
print("  全部 %d 个因子 x 全部交易日，共 %d 个『当日排序相关』读数："
      "中位 %.6f，最小 %.6f，低于 0.999 的占比 %.2f%%"
      % (len(cols), len(allv), np.median(allv), allv.min(),
         (allv < 0.999).mean() * 100))
print("  若这一列≈1.00 就是整列缩放；明显低于 1.00 说明当天谁高谁低的次序被改过")

print()
print("### 3. 同日股票池大小")
for y in (2023, 2024, 2025):
    yo = o[o["date"].dt.year == y].groupby("date")["symbol"].size()
    yn = n[n["date"].dt.year == y].groupby("date")["symbol"].size()
    print("  %d  旧 日均 %6.1f 只（%d 天） | 新 日均 %6.1f 只（%d 天）"
          % (y, yo.mean(), len(yo), yn.mean(), len(yn)))

print()
print("### 4. 历史那部分是不是一个字都没变（逐格完全相等）")
common = o["date"].max()
j2 = j[j["date"] <= common]
eq = {}
num_eq = num_both = nan_pair = 0
for c in cols:
    a = j2[c + "_o"].to_numpy()
    b = j2[c + "_n"].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    e = int(((a == b) & both).sum())
    eq[c] = (e, int(both.sum()))
    num_eq += e
    num_both += int(both.sum())
    nan_pair += int((~np.isfinite(a) & ~np.isfinite(b)).sum())
tot_cells = len(j2) * len(cols)
print("  只看 %s 及之前、能配上的 %d 行 x %d 列 = %d 个格子"
      % (str(common.date()), len(j2), len(cols), tot_cells))
print("    两边都有数的 %d 个：完全相等 %d 个（%.6f%%），不等 %d 个"
      % (num_both, num_eq, num_eq / num_both * 100, num_both - num_eq))
print("    两边都是空（NaN，因子预热期不足）的 %d 个 | 一边空一边有数的 %d 个"
      % (nan_pair, tot_cells - num_both - nan_pair))
bad = sorted(eq, key=lambda c: eq[c][1] - eq[c][0], reverse=True)[:5]
print("  不等格子最多的 5 列: " + "  ".join("%s %d/%d" % (c, eq[c][1] - eq[c][0], eq[c][1]) for c in bad))
newdays = sorted(set(n["date"]) - set(o["date"]))
print("  新面板多出来的交易日 %d 个: %s"
      % (len(newdays), " ".join(str(pd.Timestamp(d).date()) for d in newdays)))
gone = sorted(set(o["date"]) - set(n["date"]))
print("  旧面板里有、新面板里没有的交易日 %d 个%s"
      % (len(gone), "" if not gone else ": " + " ".join(str(pd.Timestamp(d).date()) for d in gone)))
