# -*- coding: utf-8 -*-
"""现行 horizon=5 预测上的三档对照：Top-50 等权 / 只开带位 / 带位 + 暴露层 A+B。

README 里那张全档实测表是在 horizon=20 的分数上跑的，换标签后要重出这三行才能对齐口径。

    python research/h5_arms.py             # 从任意目录都能跑
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
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import numpy as np
import pandas as pd
import yaml

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import Predictor, scores_from_predictions
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

# 指数区间跟着行情面板走：写死终点会把分年基准列的最后一个月截掉
BM_END = max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")
BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", BM_END).set_index("日期")["收盘"]

SC = scores_from_predictions(PR, build_tradable_mask(bars, pred["date"].unique()))
topk = Predictor(None, top_k=cb["max_positions"],
                 position_sizing=cb["position_sizing"]
                 ).generate_signals_from_series(PR, tradable=(
    build_tradable_mask(bars, pred["date"].unique())))
pol0 = policy_from_config(cfg)
ovl0 = overlay_from_config(cfg)


def run(tag, pol, ovl, sig=None):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, sig if sig is not None else SC, benchmark_prices=BM)
    eq, tr = r["equity_curve"], r["trades"]
    cost = float(tr["cost"].sum()) if "cost" in tr.columns else 0.0
    ex = r["execution"]
    rets = eq.pct_change().dropna()
    mon = (1 + rets).resample("ME").prod() - 1
    mp = rets.index.to_period("M")
    r24 = rets[rets.index.year != 2024]
    b3, b5 = set(mon.nlargest(3).index), set(mon.nlargest(5).index)
    p3 = {pd.Period(x, "M") for x in b3}
    p5 = {pd.Period(x, "M") for x in b5}

    def cut(x):
        nav = (1 + x).cumprod()
        return "累计 %+7.1f%% Sharpe %5.2f 回撤 %+6.2f%%" % (
            (nav.iloc[-1] - 1) * 100,
            x.mean() / x.std() * np.sqrt(252),
            (nav / nav.cummax() - 1).min() * 100)

    print("  %-22s %s | %s | %s | %s" % (tag, cut(rets), cut(r24),
                                         cut(rets[~mp.isin(p3)]),
                                         cut(rets[~mp.isin(p5)])))
    nn = ex.get("policy_mean_names")
    tail = "" if pol is None or nn is None else " 仓位 %.1f%% 只数 %.1f" % (
        (ex.get("policy_mean_gross_weight") or 0) * 100, nn)
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    print("  %-22s 引擎口径: 累计 %.2f%% 年化 %.2f%% 波动 %.2f%% Sharpe %.3f "
          "回撤 %.2f%% | 笔数 %d 成本 %.2f%%%s" %
          ("", pm.cumulative_return(eq) * 100, ann * 100, vol * 100,
           (ann - pm.risk_free_rate) / vol,
           pm.max_drawdown(eq)["drawdown"] * 100, len(tr),
           cost / CAP * 100, tail))
    yr = rets.groupby(rets.index.year).apply(lambda x: float((1 + x).prod() - 1))
    # 基准与策略同锚：日收益在年内复利，切片用净值曲线的日子（与 verify_backtest 同口径）
    b = BM.pct_change()
    b = b[(b.index >= eq.index.min()) & (b.index <= eq.index.max())].dropna()
    byr = b.groupby(b.index.year).apply(lambda x: float((1 + x).prod() - 1))
    print("  %-22s 分年: %s" % ("", " ".join(
        "%d %+6.1f%%(基准 %+5.1f%%)" % (y, yr[y] * 100, byr.get(y, float("nan")) * 100)
        for y in yr.index)))
    er = ex.get("exposure")
    if er is not None:
        g = er.index[er["gated"]]
        print("  %-22s IC 门控触发 %d 次: %s | 上限区间 %.0f%%~%.0f%%" % ("",
              len(g), ", ".join(pd.Timestamp(x).strftime("%Y-%m") for x in g),
              er["cap"].min() * 100, er["cap"].max() * 100))
    return r


print("=" * 96)
print("horizon=5 现行分数上的三档（100 万本金，月度评估，同一份 predictions.parquet）")
run("Top-50 等权（策略关）", None, None, sig=topk)
run("带位（无暴露层）", pol0, None)
run("带位 + A+B（现行默认）", pol0, ovl0)

print()
print("暴露层归因：A/B 各自 vs '直接静态少下注'（同一份分数、同一套费率，只换暴露来源）")
run("只开 A（IC 门控，不缩波动）", pol0, dc.replace(ovl0, vol_target_ann=0.0))
run("只开 B（目标波动，无门控）", pol0, dc.replace(ovl0, ic_window_days=0))
run("静态上限 45%（等于 B 的平均值）", dc.replace(
    pol0, max_total_pct=0.45, max_names=7), None)
run("静态上限 30% + 只数 4", dc.replace(
    pol0, max_total_pct=0.30, max_names=4, add_reserve_weight=0.02), None)
print()
print("目标波动 10% 附近扫一遍（同一份分数，A 门控保持开启，只改 vol_target_ann）")
for tv in (0.08, 0.10, 0.12, 0.15, 0.20):
    run(f"B 目标波动 {tv:.0%}", pol0, dc.replace(ovl0, vol_target_ann=tv))
print("  对照：09-22 那版 horizon=20 同一口径 → 等权 +30.95%/0.220，"
      "带位 +123.16%/0.797，A+B +121.60%/1.224")
