# -*- coding: utf-8 -*-
"""horizon=5 现行默认档的分数分布/过线只数/分桶实测，供 README 换掉 h20 旧数字。

临时脚本。
"""
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
H = int(cfg["model"]["horizon"])
BUY = float(cfg["position_policy"]["buy_score"])
STRONG = float(cfg["position_policy"]["strong_score"])

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
s = pred.set_index(["date", "symbol"])["prediction"].sort_index()

print("样本外分数行数: %s | 覆盖 %d 个交易日 × %d 只"
      % (f"{len(s):,}", s.index.get_level_values("date").nunique(),
         s.index.get_level_values("symbol").nunique()))

q = lambda p: float(s.quantile(p))
print("  分数分布(horizon=%d): 中位 %+.2f%% | 均值 %+.2f%% | σ %.2f%% | "
      "P90 %+.2f%% | P99 %+.2f%%" %
      (H, q(.5) * 100, float(s.mean()) * 100, float(s.std()) * 100,
       q(.90) * 100, q(.99) * 100))
print("  阈值分位: buy %.5f = P%.3f | strong %.5f = P%.3f" %
      (BUY, 100 * float((s < BUY).mean()), STRONG, 100 * float((s < STRONG).mean())))

# ---- 每日过线只数 ----
wide = s.unstack("symbol")
print("  日截面中位数的中位 %+.2f%% | 日均过买线只数 %.0f | 全样本 P50 %+.2f%%"
      % (float(wide.median(axis=1).median()) * 100,
         float((wide >= BUY).sum(axis=1).mean()), q(.5) * 100))
cnt_buy = (wide >= BUY).sum(axis=1)
cnt_str = (wide >= STRONG).sum(axis=1)
print("  过线只数/日: 买线 均值 %.0f 中位 %.0f 最小 %d(=%s) 为 0 的天数 %d" %
      (cnt_buy.mean(), cnt_buy.median(), int(cnt_buy.min()),
       pd.Timestamp(cnt_buy.idxmin()).date(), int((cnt_buy == 0).sum())))
print("               强线 均值 %.0f 中位 %.0f 最小 %d 为 0 的天数 %d" %
      (cnt_str.mean(), cnt_str.median(), int(cnt_str.min()),
       int((cnt_str == 0).sum())))

# ---- 分位分桶 vs 实际前向收益 ----
import pathlib

paths = sorted(pathlib.Path(cfg["cache"]["directory"]).glob("daily/*.parquet"))
frames = []
for p in paths:
    try:
        df = pd.read_parquet(p, columns=["日期", "收盘"])
    except Exception:
        continue
    if df.empty:
        continue
    df = df.sort_values("日期")
    df["sym"] = p.stem
    for h in {5, 20}:
        df[f"fwd{h}"] = df["收盘"].shift(-h) / df["收盘"] - 1.0
    frames.append(df[["sym", "日期", "fwd5", "fwd20"]])
fw = pd.concat(frames, ignore_index=True)
fw["日期"] = pd.to_datetime(fw["日期"])
print("  前向收益表: %s 行(全缓存股票)" % f"{len(fw):,}")

fwd = fw.set_index(["日期", "sym"])[["fwd5", "fwd20"]]
fwd.index.names = ["date", "symbol"]
pct = wide.rank(axis=1, pct=True).stack().rename("pct")
j = pd.concat([s.rename("score"), pct], axis=1).join(fwd, how="inner")
edges = [(1.00, .988, "top 1.2%(≈12只)"), (.988, .95, "1.2–5%"), (.95, .90, "5–10%"),
         (.90, .80, "10–20%"), (.80, .50, "20–50%"), (.50, -0.01, "50–100%")]
print("\n  按每日分数名次分桶(pct=1 为当日最高分):")
print("  %-18s %10s %10s %10s %10s" % ("分桶", "n", "均5日", "均20日", "5日胜率"))
for hi, lo, name in edges:
    sub = j[(j["pct"] > lo) & (j["pct"] <= hi)]
    print("  %-18s %10s %+9.2f%% %+9.2f%% %9.0f%%" %
          (name, f"{len(sub):,}", sub["fwd5"].mean() * 100,
           sub["fwd20"].mean() * 100, (sub["fwd5"] > 0).mean() * 100))
