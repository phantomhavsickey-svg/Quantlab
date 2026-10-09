# -*- coding: utf-8 -*-
"""把某个 probe 预测文件跑一遍回测，看它买了什么（重点：300641）。

    python research/probe_arm_trades.py data/cache/_probe_backup_20260930_seed7/predictions.parquet
"""
import contextlib
import dataclasses as dc
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

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

PATH = sys.argv[1]
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
cache = CacheManager(cfg["cache"]["directory"])
pm = PerformanceMetrics()

pred = pd.read_parquet(PATH)
pred["date"] = pd.to_datetime(pred["date"])
panel = pd.read_parquet("data/cache/factor_panel.parquet", columns=["symbol"])
bars_all = {}
for s in sorted(set(panel["symbol"]) | set(pred["symbol"])):
    df = cache.get_daily(s)
    if df is not None:
        bars_all[s] = df.sort_values("日期")
try:
    BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
        cb.get("benchmark", "000852"), "20210101",
        max(d["日期"].max() for d in bars_all.values()).strftime("%Y%m%d")).set_index("日期")["收盘"]
except Exception:
    BM = None

PR = pred.set_index(["date", "symbol"])["prediction"]
SC = scores_from_predictions(PR, build_tradable_mask(bars_all, pred["date"].unique()))
eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                     max_positions=cb["max_positions"],
                     cost_model=TransactionCostModel(
                         cm["commission_rate"], cm["min_commission"], cm["stamp_tax_rate"],
                         cm["slippage_rate"], cm.get("stamp_tax_schedule") or None),
                     lot_size=int(cm.get("lot_size", 100)),
                     policy=dc.replace(policy_from_config(cfg)),
                     overlay=overlay_from_config(cfg))
with contextlib.redirect_stdout(io.StringIO()):
    r = eng.run(bars_all, SC, benchmark_prices=BM)
eq, tr = r["equity_curve"], r["trades"]
tr["year"] = pd.to_datetime(tr["date"]).dt.year
print("### %s" % PATH)
print("  累计 %+.2f%% | 笔数 %d | 成本 %.0f 元" % (pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum()))
print("  分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                            for y, (f, l) in eq.groupby(eq.index.year).agg(["first", "last"]).iterrows()))
s = tr.query("side=='sell'")
t = s[s["symbol"] == "300641"].sort_values("date")
print("  300641 卖出笔 %d" % len(t))
for _, x in t.iterrows():
    print("    %s  %+.0f 元 (%+.2fpp)" % (x["date"].strftime("%Y-%m-%d"), x["realized_pnl"],
                                          x["realized_pnl"] / CAP * 100))
y24 = s[s["year"] == 2024]
print("  2024 卖出合计 %+.0f 元 (%+.2fpp)，最好3笔: %s"
      % (y24["realized_pnl"].sum(), y24["realized_pnl"].sum() / CAP * 100,
         "  ".join("%s %+.0f" % (x["symbol"], x["realized_pnl"])
                   for _, x in y24.nlargest(3, "realized_pnl").iterrows())))
