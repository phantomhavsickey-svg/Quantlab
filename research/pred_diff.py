# -*- coding: utf-8 -*-
"""把 data/cache/predictions.parquet 与 backup_20260930 里那份按年对比分数差。

    python research/pred_diff.py
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
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
pp = cfg["position_policy"]
BUY = float(pp["buy_score"])
STRONG = float(pp["strong_score"])
TOPN = int(pp["max_names"])

new = pd.read_parquet("data/cache/predictions.parquet")
OLD = sys.argv[1] if len(sys.argv) > 1 else "data/cache/backup_20260930/predictions.parquet"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "旧 备份"
old = pd.read_parquet(OLD)
for d in (new, old):
    d["date"] = pd.to_datetime(d["date"])

print("### 0. 两份文件概况")
print("  对照文件: %s" % OLD)
for label, d in (("新 当前", new), (LABEL, old)):
    print("  %-8s 行 %-7d %s ~ %s | 均值 %+.5f σ %.5f | ≥买线 %.2f%% ≥强线 %.2f%%"
          % (label, len(d), d["date"].min().date(), d["date"].max().date(),
             d["prediction"].mean(), d["prediction"].std(),
             (d["prediction"] >= BUY).mean() * 100, (d["prediction"] >= STRONG).mean() * 100))

j = new.merge(old, on=["date", "symbol"], suffixes=("_n", "_o"))
j["year"] = j["date"].dt.year
print("  同 (日期,股票) 配上的 %d 行（新 %d / 旧 %d）" % (len(j), len(new), len(old)))

print()
print("### 1. 按年：两套分数的相关、离散与逐行差")
for y, g in j.groupby("year"):
    print("  %d n=%-7d Spearman %+.4f Pearson %+.4f | σ 新 %.5f 旧 %.5f | 平均|Δ| %.5f | 平均Δ %+.5f"
          % (y, len(g),
             g["prediction_n"].corr(g["prediction_o"], method="spearman"),
             g["prediction_n"].corr(g["prediction_o"]),
             g["prediction_n"].std(), g["prediction_o"].std(),
             (g["prediction_n"] - g["prediction_o"]).abs().mean(),
             (g["prediction_n"] - g["prediction_o"]).mean()))

print()
print("### 2. 每月最后一个信号日：前 %d 只重合情况" % TOPN)
sig = j.groupby(j["date"].dt.to_period("M"))["date"].max().tolist()
rec = []
for d in sig:
    g = j[j["date"] == d]
    tn = set(g.nlargest(TOPN, "prediction_n")["symbol"])
    to = set(g.nlargest(TOPN, "prediction_o")["symbol"])
    bn = set(g.loc[g["prediction_n"] >= BUY, "symbol"])
    bo = set(g.loc[g["prediction_o"] >= BUY, "symbol"])
    rec.append(dict(date=d, year=d.year, inter=len(tn & to), n_new=len(bn), n_old=len(bo),
                    jac=(len(bn & bo) / max(1, len(bn | bo)))))
r = pd.DataFrame(rec)
for y, g in r.groupby("year"):
    print("  %d %2d 个月 | 前%d只平均重合 %.1f 只（%.0f%%）| 过买线只数 新 %.1f vs 旧 %.1f | "
          "过线名单重合度 %.2f"
          % (y, len(g), TOPN, g["inter"].mean(), g["inter"].mean() / TOPN * 100,
             g["n_new"].mean(), g["n_old"].mean(), g["jac"].mean()))

print()
print("### 3. 两套名单的实际 5 日标签（用新文件的 forward_return，信号日同 §2）")
labmap = dict(zip(zip(new["date"], new["symbol"]), new["forward_return"]))
rows = []
for d in sig:
    g = j[j["date"] == d]
    for col, tag in (("prediction_n", "新选"), ("prediction_o", "旧选")):
        vals = [labmap.get((d, s), np.nan) for s in g.nlargest(TOPN, col)["symbol"]]
        vals = [v for v in vals if v == v]
        if vals:
            rows.append(dict(date=d, year=d.year, tag=tag, mean=float(np.mean(vals)),
                             win=float(np.mean([v > 0 for v in vals]))))
q = pd.DataFrame(rows)
base = new[new["date"].isin(sig)]["forward_return"]
print("  信号日全池标签均值 %+.3f%% | 为正 %.1f%%（n=%d）"
      % (base.mean() * 100, (base > 0).mean() * 100, len(base)))
for tag, g in q.groupby("tag", sort=False):
    print("  %s 前%-3d只 月均标签 %+.3f%%  为正 %.1f%%  样本 %d 月"
          % (tag, TOPN, g["mean"].mean() * 100, g["win"].mean() * 100, len(g)))
for y, g in q[q["year"] <= 2025].groupby(["year", "tag"]):
    print("  %d %s 月均标签 %+.3f%%（%d 月）" % (y[0], y[1], g["mean"].mean() * 100, len(g)))
