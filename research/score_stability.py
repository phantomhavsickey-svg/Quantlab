# -*- coding: utf-8 -*-
"""两臂分数到底差多少：逐日截面排序相关，按折拆开看。

    python research/score_stability.py

旧臂 = data/cache/backup_20260930/predictions.parquet（09-25 那次 h5 训练，
面板取到 09-22，854,763 条）；新臂 = data/cache/predictions.parquet（09-30 训练，
面板取到 09-29，860,431 条 + 10-09 那天的 3 天增量）。

每折的 best_iteration 从两份 train log 里现读，不手抄。
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

OLD = "data/cache/backup_20260930/predictions.parquet"
NEW = "data/cache/predictions.parquet"
OLD_LOG = "logs/h5_train.log"
NEW_LOG = "logs/train_20260930.log"


def read_folds(path):
    """从 train log 里读 (训练截止日, best_iteration)。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    cuts = re.findall(r"Fold (\d+): 训练截止 (\d{4}-\d{2}-\d{2})", txt)
    iters = re.findall(r"训练完成: best_iteration=(\d+)", txt)
    out = []
    for (i, cut), it in zip(cuts, iters):
        out.append((int(i), pd.Timestamp(cut), int(it)))
    return out


folds_old = read_folds(OLD_LOG)
folds_new = read_folds(NEW_LOG)
print("### 0. 两臂的折切点与早停迭代数")
print("  折 | 训练截止 | best_iteration 旧 -> 新")
for (i, c, io), (_, _, ino) in zip(folds_old, folds_new):
    print("  %d  | %s | %4d -> %4d%s" % (i, c.date(), io, ino, "" if io != ino else "   <- 没变"))

old = pd.read_parquet(OLD)
new = pd.read_parquet(NEW)
for d in (old, new):
    d["date"] = pd.to_datetime(d["date"])
j = old.merge(new, on=["date", "symbol"], suffixes=("_o", "_n"))
print()
print("### 1. 能配上的 (日期,股票)")
print("  旧 %d 行 / 新 %d 行 -> 配上 %d 行，%d 个共同交易日"
      % (len(old), len(new), len(j), j["date"].nunique()))

# 每折窗口：[cut_i, cut_{i+1})
bounds = [c for _, c, _ in folds_new] + [pd.Timestamp.max]

rows = []
for k, (fi, cut, _) in enumerate(folds_new):
    lo, hi = bounds[k], bounds[k + 1]
    g = j[(j["date"] >= lo) & (j["date"] < hi)]
    rhos = []
    for _, day in g.groupby("date", sort=False):
        if len(day) < 30:
            continue
        r = day["prediction_o"].corr(day["prediction_n"], method="spearman")
        if np.isfinite(r):
            rhos.append(r)
    rhos = np.array(rhos)
    if len(rhos) == 0:
        continue
    rows.append(dict(fold=fi, cut=cut, days=len(rhos),
                     med=np.median(rhos), mean=rhos.mean(), p10=np.percentile(rhos, 10),
                     frac95=(rhos < 0.95).mean(), frac99=(rhos < 0.99).mean(),
                     it_old=folds_old[k][2], it_new=folds_new[k][2]))
print()
print("### 2. 逐折：同一天截面内，旧分数 vs 新分数的排序相关（Spearman）")
print("  折 | 天数 | 中位 | 均值 | 最差10% | <0.95 的天占比 | best_iteration")
for r in rows:
    print("  %d  | %4d | %.4f | %.4f | %.4f | %5.1f%% | %4d -> %4d"
          % (r["fold"], r["days"], r["med"], r["mean"], r["p10"], r["frac95"] * 100,
             r["it_old"], r["it_new"]))

same = [r for r in rows if r["it_old"] == r["it_new"]]
diff = [r for r in rows if r["it_old"] != r["it_new"]]
def agg(rs, key):
    return np.average([r[key] for r in rs], weights=[r["days"] for r in rs])
if same:
    print("  迭代数相同的折: %d 个，加权日均中位 %.4f，<0.95 的天占比 %.1f%%"
          % (len(same), agg(same, "med"), agg(same, "frac95") * 100))
if diff:
    print("  迭代数变了的折: %d 个，加权日均中位 %.4f，<0.95 的天占比 %.1f%%"
          % (len(diff), agg(diff, "med"), agg(diff, "frac95") * 100))

print()
print("### 3. 每个月末：两臂各自分数最高的 12 只，重合几只")
j["ym"] = j["date"].dt.strftime("%Y-%m")
ov = []
for ym, g in j.groupby("ym"):
    last = g["date"].max()
    day = g[g["date"] == last]
    if len(day) < 30:
        continue
    a = set(day.nlargest(12, "prediction_o")["symbol"])
    b = set(day.nlargest(12, "prediction_n")["symbol"])
    ov.append(len(a & b))
ov = np.array(ov)
print("  %d 个月末：平均重合 %.2f / 12（中位 %.1f，最少 %d，最多 %d）"
      % (len(ov), ov.mean(), np.median(ov), ov.min(), ov.max()))
print("  完全一样的月份 %d 个；重合 <=6 只的 %d 个" % (int((ov == 12).sum()), int((ov <= 6).sum())))
