# -*- coding: utf-8 -*-
"""分年收益的三条同轴对照：现行默认档 vs 中证1000 指数 vs 池内 1000 只等权。

存在的理由有两个：

1. 文档里"2025 跟住指数、跑输等权 15.95pp"这类断言要三位小数才判得动，
   `research/h5_arms.py` 只打到 1 位。
2. 报告里的"基准收益 12.18%"与"分年基准自己复利"对不上，差的是一个交易日：
   引擎把基准归一到**净值首日**（不计当天涨幅），而按"年内日收益复利"切片会把
   2023-02-01 当天的 +1.638% 算进去。本脚本两个口径都打出来，免得以后再猜。

跑法（指数已缓存时实测 9 秒，首次要下一份指数行情；只读缓存，不改生产代码）：

    python research/annual_baseline.py        # 从任意目录都能跑
"""
import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import pandas as pd
import yaml

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = cb.get("initial_capital", 1_000_000)
pm = PerformanceMetrics()

cache = CacheManager(cfg["cache"]["directory"])
pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
PR = pred.set_index(["date", "symbol"])["prediction"]

bars = {}
for s in pred["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df.sort_values("日期")

BM_END = max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")
BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", BM_END).set_index("日期")["收盘"]

# 池内等权：当日有行情的名字等权，日内再平衡，不含任何费率
EW = pd.DataFrame({s: df.set_index("日期")["收盘"].sort_index()
                   for s, df in bars.items()}).pct_change().mean(axis=1).dropna()

eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                     max_positions=cb["max_positions"],
                     cost_model=TransactionCostModel(
                         cm["commission_rate"], cm["min_commission"],
                         cm["stamp_tax_rate"], cm["slippage_rate"],
                         cm.get("stamp_tax_schedule")),
                     lot_size=int(cm.get("lot_size", 100)),
                     policy=policy_from_config(cfg),
                     overlay=overlay_from_config(cfg))
with contextlib.redirect_stdout(io.StringIO()):
    r = eng.run(bars, scores_from_predictions(PR, build_tradable_mask(
        bars, pred["date"].unique())), benchmark_prices=BM)
eq = r["equity_curve"]
rets = eq.pct_change().dropna()
LO, HI = eq.index.min(), eq.index.max()


def yearly(ser):
    w = ser[(ser.index >= LO) & (ser.index <= HI)]
    return w.groupby(w.index.year).apply(lambda x: float((1 + x).prod() - 1))


strat, byr, ewr = yearly(rets), yearly(BM.pct_change().dropna()), yearly(EW)
cum = lambda s: float((1 + s[(s.index >= LO) & (s.index <= HI)]).prod()) - 1

print("=" * 92)
print("分年对照（净值区间 %s ~ %s，%d 个交易日；年内日收益复利）"
      % (LO.date(), HI.date(), len(rets)))
print("  %-6s %-11s %-11s %-11s %-13s %-13s" % ("年", "策略(费后)", "中证1000", "池内等权",
                                                "策略−基准", "策略−等权"))
for y in strat.index:
    print("  %-6d %+9.3f%% %+9.3f%% %+9.3f%% %+11.2fpp %+11.2fpp" % (
        y, strat[y] * 100, byr[y] * 100, ewr[y] * 100,
        (strat[y] - byr[y]) * 100, (strat[y] - ewr[y]) * 100))
print("  %-6s %+9.3f%% %+9.3f%% %+9.3f%% %+11.2fpp %+11.2fpp" % (
    "全窗口", cum(rets) * 100, cum(BM.pct_change().dropna()) * 100, cum(EW) * 100,
    (cum(rets) - cum(BM.pct_change().dropna())) * 100,
    (cum(rets) - cum(EW)) * 100))
print()
print("口径对照（同一个基准的两种算法，差的就是 2023-02-01 当天那一跳）：")
m = r["metrics"] if "metrics" in r else {}
print("  引擎/报告口径  基准累计 %.2f%%  超额 %.2fpp（基准归一到净值首日 %s 收盘，"
      "不计当日涨幅）" % (m.get("benchmark_cumulative_return", float("nan")) * 100,
                          m.get("excess_return", float("nan")) * 100, LO.date()))
print("  本脚本切片口径 基准累计 %.2f%%（含 %s 当日 %+.3f%%）"
      % (cum(BM.pct_change().dropna()) * 100, LO.date(),
         float(BM.pct_change().dropna().loc[LO]) * 100))
print("  池内等权口径：当日有行情的名字等权、日内再平衡、**不含费率**；"
      "折年化 %.2f%%" % (((1 + cum(EW)) ** (252 / len(EW[(EW.index >= LO) & (EW.index <= HI)]))
                          - 1) * 100))
print("done")
