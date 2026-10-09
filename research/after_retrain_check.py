"""重训后 predictions 的现状盘点（只读）：区间、三条线的实际分位、过线只数、末日名单。

三条线是 2026-09-27 按当时的分数分布定在 P84/P94/P97 的固定数值；重训会改变
分数分布，所以线不动的情况下实际分位会漂。这里量漂移，不改 config。

用法: python research/after_retrain_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
POL = CFG["position_policy"]
SELL, BUY, STRONG = (float(POL["sell_score"]), float(POL["buy_score"]),
                     float(POL["strong_score"]))
top_n = int(POL["max_names"])

p = pd.read_parquet("data/cache/predictions.parquet")
p["date"] = pd.to_datetime(p["date"])
dates = sorted(p["date"].unique())
print("行数 %d | 日期 %s .. %s | 共 %d 个信号日" % (
    len(p), pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(), len(dates)))
print("整体: 均值 %.5f 标准差 %.5f 最小 %.5f 最大 %.5f" % (
    p["prediction"].mean(), p["prediction"].std(),
    p["prediction"].min(), p["prediction"].max()))


def pct(x, series):
    return 100.0 * float((series < x).mean()) + 0.5 * float((series == x).mean())


for name, v in (("清仓线 sell_score", SELL), ("建仓线 buy_score", BUY),
               ("强档/补仓 strong_score", STRONG)):
    d_last = p[p["date"] == dates[-1]]["prediction"]
    print("  %-22s %.5f -> 全样本 P%.2f | 最新截面(%s) P%.2f" % (
        name, v, pct(v, p["prediction"]), pd.Timestamp(dates[-1]).date(),
        pct(v, d_last)))

per_day_std = p.groupby("date")["prediction"].std()
print("逐日截面标准差: 全样本均值 %.5f | 最近 20 日均值 %.5f" % (
    per_day_std.mean(), per_day_std.tail(20).mean()))
print("有标签率 forward_return 非空 %.2f%%" % (
    100.0 * p["forward_return"].notna().mean()))

print()
print("最近 6 个信号日的过线只数:")
for d in dates[-6:]:
    s = p[p["date"] == d]["prediction"]
    print("  %s  全市场 %4d 只 | ≥建仓线 %3d | ≥强档线 %3d | 最高 %.5f" % (
        pd.Timestamp(d).date(), len(s), int((s >= BUY).sum()),
        int((s >= STRONG).sum()), s.max()))

last = p[p["date"] == dates[-1]].sort_values("prediction", ascending=False)
print("最新截面 %s 前 %d 只:" % (pd.Timestamp(dates[-1]).date(), top_n))
for i, (sym, sc) in enumerate(
        zip(last["symbol"].head(top_n), last["prediction"].head(top_n)), start=1):
    print("  #%02d %s %.5f %s" % (i, sym, sc,
                                  "≥强档" if sc >= STRONG else ""))
