# -*- coding: utf-8 -*-
"""回答"分数为什么每天变、过线只数为什么变这么多"，四个实测分段。

    python research/daily_score_variance.py
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
H = int(cfg["model"]["horizon"])

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])

g = pred.groupby("date")["prediction"]
cnt = g.apply(lambda x: int((x >= BUY).sum()))
sd = g.std()
mean = g.mean()
mx = g.max()
n = g.size()

print("### 1. 每天截面（全部股票同一天）的分数统计，%d 个交易日" % len(cnt))
print("  当日只数        中位 %5.0f  最小 %d  最大 %d" % (n.median(), n.min(), n.max()))
print("  截面均值        中位 %+.5f  最小 %+.5f  最大 %+.5f" % (mean.median(), mean.min(), mean.max()))
print("  截面标准差      中位 %.5f  最小 %.5f  最大 %.5f" % (sd.median(), sd.min(), sd.max()))
print("  ≥ 建仓线 %.5f 的只数  中位 %d  平均 %.1f  最小 %d  最大 %d  日间变化标准差 %.1f"
      % (BUY, cnt.median(), cnt.mean(), cnt.min(), cnt.max(), cnt.diff().std()))
print("  过线只数与当日截面标准差的等级相关 %+.4f；与当日只数的等级相关 %+.4f"
      % (cnt.corr(sd, method="spearman"), cnt.corr(n, method="spearman")))

print()
print("### 2. 最后 15 个交易日的过线只数序列")
tail = cnt.tail(15)
print("  " + "  ".join("%s:%d" % (d.strftime("%m-%d"), c) for d, c in tail.items()))
print("  同窗口截面标准差 " + "  ".join("%.4f" % sd[d] for d in tail.index))

print()
print("### 3. 相邻两个交易日：同一只股票的分数动了多少")
dts = sorted(pred["date"].unique())
piv = pred.pivot_table(index="date", columns="symbol", values="prediction")
diffs, jaccs = [], []
for a, b in zip(dts[:-1], dts[1:]):
    row = (piv.loc[b] - piv.loc[a]).dropna()
    if len(row) > 50:
        diffs.append((b, float(row.abs().median()), float(row.abs().mean())))
    sa = set(piv.loc[a].dropna()[lambda x: x >= BUY].index)
    sb = set(piv.loc[b].dropna()[lambda x: x >= BUY].index)
    if sa or sb:
        jaccs.append(len(sa & sb) / max(1, len(sa | sb)))
dd = pd.DataFrame(diffs, columns=["date", "median_abs", "mean_abs"])
print("  逐股 |今日分数 − 昨日分数|：全体中位 %.5f  全体均值 %.5f（%d 个相邻日对）"
      % (dd["median_abs"].median(), dd["mean_abs"].median(), len(dd)))
print("  对照：截面标准差中位 %.5f —— 所以一天的移动量约等于 %+.0f%% 个标准差"
      % (sd.median(), dd["median_abs"].median() / sd.median() * 100))
print("  过线名单日间重合度（Jaccard，1=完全相同）中位 %.2f  平均 %.2f"
      % (np.median(jaccs), float(np.mean(jaccs))))

print()
print("### 4. 特征端：同一只股票相邻两天的特征值差多少（面板取 3 列）")
fc = ["momentum_5d", "rsi_14", "volatility_10d"]
panel = pd.read_parquet("data/cache/factor_panel.parquet", columns=["date", "symbol"] + fc)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["symbol", "date"])
for c in fc:
    d1 = panel.groupby("symbol")[c].diff().abs().dropna()
    print("  %-15s 相邻两日 |Δ| 中位 %.5f  p90 %.5f  最大 %.5f  当天截面 σ %.5f"
          % (c, d1.median(), d1.quantile(.9), d1.max(), panel[c].std()))
print("  （这三列都是滚动窗口类特征：窗口每天前进一格，输入必然每天变）")
