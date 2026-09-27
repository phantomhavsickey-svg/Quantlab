# -*- coding: utf-8 -*-
"""目标波动扫档里"地板 scale_floor=0.30 顶住了多少个评估日"。

README 结论 4(d) 那句 "地板已经顶住 N/45" 是个派生断言，不能靠上一参数版本记着，
所以在这里现算。判据只看波动通道：clip(target/realized, floor, 1) 触地板
<=> realized_ann >= target/scale_floor，与 A 门控的 ×0.50 无关（两者叠在同一个
cap 上，用 cap 反推会把"被门控压到地板以下"误算成"触地板"）。

    python research/floor_bind.py        # 五个目标波动档 × 一次全量回测，约 1~2 分钟
"""
import contextlib
import dataclasses
import io
import os
import sys

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from backtest.cost import TransactionCostModel          # noqa: E402
from backtest.engine import BacktestEngine              # noqa: E402
from data.cache import CacheManager                     # noqa: E402
from data.downloader import DataDownloader              # noqa: E402
from models.predictor import Predictor, scores_from_predictions  # noqa: E402
from utils.exposure import overlay_from_config          # noqa: E402
from utils.market_rules import build_tradable_mask      # noqa: E402
from utils.position_policy import policy_from_config    # noqa: E402

TARGETS = (0.08, 0.10, 0.12, 0.15, 0.20)

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = cb.get("initial_capital", 1_000_000)

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

SC = scores_from_predictions(PR, build_tradable_mask(bars, pred["date"].unique()))
pol = policy_from_config(cfg)
ovl0 = overlay_from_config(cfg)
FLOOR = ovl0.scale_floor
BASE = 1 - cb.get("min_cash_ratio", 0.0)   # 仅用于打印,真正的基线在引擎里取 max_total_pct
print("scale_floor = %.2f | 补仓线 add_score_line = %.5f | 清仓线 = %.5f" %
      (FLOOR, pol.add_score_line, pol.sell_score))
print("触地板判据: 全池等权年化已实现波动 >= target / %.2f" % FLOOR)
for tv in TARGETS:
    ovl = dataclasses.replace(ovl0, vol_target_ann=tv)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, SC, benchmark_prices=BM)
    ex = r.get("execution") or {}
    er = ex.get("exposure")
    if er is None:
        print("目标波动 %.0f%%: 引擎没给暴露层诊断,跳过" % (tv * 100))
        continue
    rv = pd.to_numeric(er["realized_vol"], errors="coerce").dropna()
    bound = int((rv >= tv / FLOOR - 1e-12).sum())
    cap = pd.to_numeric(er["cap"], errors="coerce")
    print("目标波动 %2.0f%%: 评估 %d 次(有波动读数 %d) | 触地板 %d/%d = %.1f%%"
          " | cap 均值 %.1f%% 最低 %.1f%% | 门控 %d 次 | 已实现波动中位 %.1f%%"
          % (tv * 100, len(er), len(rv), bound, len(er), 100.0 * bound / len(er),
             cap.mean() * 100, cap.min() * 100,
             int(ex.get("n_gated_evals", -1)), float(rv.median()) * 100))
