# -*- coding: utf-8 -*-
"""只读复算原分支 6 个种子的基线回测数字(生产线),供成对对照。

    python research/rebaseline_base.py

数据与预测全部从原分支 G:\\个人量化\\Quantlab 只读加载(该分支一字不改,
PYTHONDONTWRITEBYTECODE=1 运行);BM 传 None 跳过基准下载,不产生任何写路径。
输出建议重定向到 logs/rebaseline_base.log。
"""
import contextlib
import dataclasses as dc
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_ROOT = r"G:\个人量化\Quantlab"
os.chdir(BASE_ROOT)
sys.path.insert(0, BASE_ROOT)

import pandas as pd
import yaml

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from data.cache import CacheManager
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

SEEDS = [42, 7, 13, 0, 1, 2]
PATHS = {
    42: "data/cache/predictions.parquet",
    0: "data/cache/_probe_cache_seed0/predictions.parquet",
    1: "data/cache/_probe_cache_seed1/predictions.parquet",
    2: "data/cache/_probe_cache_seed2/predictions.parquet",
    7: "data/cache/_probe_seed7/predictions.parquet",
    13: "data/cache/_probe_seed13/predictions.parquet",
}

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])
pol = policy_from_config(cfg)
print("生产线 buy %.5f / sell %.5f / strong=add %.5f"
      % (pol.buy_score, pol.sell_score, pol.strong_score))

pred42 = pd.read_parquet(PATHS[42])
pred42["date"] = pd.to_datetime(pred42["date"])
uni = sorted(pred42["symbol"].unique())
bars_all = {}
for s in uni:
    df = cache.get_daily(s)
    if df is not None:
        bars_all[s] = df.sort_values("日期")

for seed in SEEDS:
    pdf = pd.read_parquet(PATHS[seed])
    pdf["date"] = pd.to_datetime(pdf["date"])
    PR = pdf.set_index(["date", "symbol"])["prediction"]
    SC = scores_from_predictions(PR, build_tradable_mask(bars_all, pdf["date"].unique()))
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule") or None),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=dc.replace(pol),
                         overlay=overlay_from_config(cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars_all, SC, benchmark_prices=None)
    eq, tr = r["equity_curve"], r["trades"]
    yr = eq.groupby(eq.index.year).agg(["first", "last"])
    print("  base seed%-3d 累计 %+8.2f%%  笔数 %4d  成本 %7.0f 元  平均目标仓位 %.1f%%  平均只数 %.1f"
          % (seed, pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum(),
             (r["execution"].get("policy_mean_gross_weight") or 0) * 100,
             r["execution"].get("policy_mean_names") or 0))
    print("              分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                                           for y, (f, l) in yr.iterrows()))
