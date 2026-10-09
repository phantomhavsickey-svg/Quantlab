# -*- coding: utf-8 -*-
"""副本分数分布 → 重标定三条分位线(P84/P94/P97),口径照 research/fill_and_threshold.py §1。

    python research/regime_lines.py

输出新线数值后,手工写回 config.yaml 的 position_policy(本脚本只读不写)。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import yaml

from data.cache import CacheManager

H = 5
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
preds_df = pd.read_parquet("data/cache/predictions.parquet")
preds_df["date"] = pd.to_datetime(preds_df["date"])

cache = CacheManager(cfg["cache"]["directory"])
bars = {}
for s in preds_df["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df

closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
fwd = (closes.shift(-H) / closes - 1).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
df = preds_df.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
print("配对样本: %d 行 (%d 个交易日)" % (len(df), df["date"].nunique()))

# 生产分支的绝对线写死在脚本里(不读 config:config 一改,对照就没了,见 docs/入门.md §4)
PROD = dict(buy=0.01354, sell=0.00875, strong=0.01807)
pct = lambda v: (df.prediction < v).mean() * 100
print("生产分支绝对线在新分布上的分位: buy %.5f=P%.2f | sell %.5f=P%.2f | strong %.5f=P%.2f"
      % (PROD["buy"], pct(PROD["buy"]),
         PROD["sell"], pct(PROD["sell"]),
         PROD["strong"], pct(PROD["strong"])))
cur = cfg["position_policy"]
print("(现行 config 值 buy %.5f / sell %.5f / strong %.5f)"
      % (cur["buy_score"], cur["sell_score"], cur["strong_score"]))

new_buy = float(df.prediction.quantile(0.94))
new_sell = float(df.prediction.quantile(0.84))
new_strong = float(df.prediction.quantile(0.97))
print("新 P84 = %.5f | 新 P94 = %.5f | 新 P97 = %.5f" % (new_sell, new_buy, new_strong))

per_day = (df[df.prediction >= new_buy].groupby("date").size()
           .reindex(sorted(df.date.unique()), fill_value=0))
print("新买线以上候选只数: 中位 %.0f | 最低 %d (%s) | 不足 12 只的日子 %.1f%%"
      % (per_day.median(), per_day.min(), str(per_day.idxmin().date()),
         (per_day < 12).mean() * 100))
