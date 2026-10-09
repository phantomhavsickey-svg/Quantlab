# -*- coding: utf-8 -*-
"""把一条 walk-forward 净值拆开：收益集中在哪几只名字、哪几笔、哪一年。

    python research/arm_decomp.py <predictions.parquet> <pool_panel.parquet> [tag]

pool_panel 决定股票池（口径照 research/panel_arms.py 与 why_162pp.py：只取面板里的 symbol）；
日线缓存与标签用当前这一份（旧缓存已被覆盖过，这一点与 panel_arms.py 相同）。
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

PRED, POOL = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(PRED)
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])

pred = pd.read_parquet(PRED)
pred["date"] = pd.to_datetime(pred["date"])
panel = pd.read_parquet(POOL, columns=["symbol"])
bars_all = {}
for s in sorted(set(panel["symbol"])):
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
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
                         cm["commission_rate"], cm["min_commission"],
                         cm["stamp_tax_rate"], cm["slippage_rate"],
                         cm.get("stamp_tax_schedule") or None),
                     lot_size=int(cm.get("lot_size", 100)),
                     policy=dc.replace(policy_from_config(cfg)),
                     overlay=overlay_from_config(cfg))
with contextlib.redirect_stdout(io.StringIO()):
    r = eng.run(bars_all, SC, benchmark_prices=BM)
eq, tr = r["equity_curve"], r["trades"]
tr["date"] = pd.to_datetime(tr["date"])
tr["year"] = tr["date"].dt.year

print("### %s | 预测 %s | 池 %s" % (TAG, PRED, POOL))
print("  A. 累计 %+.2f%% | 笔数 %d | 成本 %.0f 元 | 平均目标仓位 %.1f%% | 期末持仓 %d 只"
      % (pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum(),
         (r["execution"].get("policy_mean_gross_weight") or 0) * 100,
         int(r["execution"].get("policy_final_names") or 0)))
yr = eq.groupby(eq.index.year).agg(["first", "last"])
print("     分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                              for y, (f, l) in yr.iterrows()))
s = tr.query("side=='sell'")
tot = s["realized_pnl"].sum()
nav = (float(eq.iloc[-1]) / float(eq.iloc[0]) - 1) * CAP
bcost = tr.query("side=='buy'")["cost"].sum()
scost = tr.query("side=='sell'")["cost"].sum()
print("     净值变动 %+.0f 元（=累计收益×本金） | 卖出已实现合计 %+.0f 元 | 差 %+.0f 元"
      % (nav, tot, nav - tot))
print("     首日净值 %.6f | 末日净值 %.6f | 买入成本 %.0f 元 | 卖出成本 %.0f 元"
      % (float(eq.iloc[0]), float(eq.iloc[-1]), bcost, scost))
g = s.groupby("symbol")["realized_pnl"].agg(["sum", "count"]).sort_values("sum", ascending=False)
print("  B. 卖出已实现盈亏按名字 top12（占本金 pp）：")
for sym, row in g.head(12).iterrows():
    print("     %s  %+10.0f 元 (%+.2fpp)  %d 笔" % (sym, row["sum"], row["sum"] / CAP * 100, row["count"]))
print("  C. 集中度（sell 总盈亏 %+.0f 元 = %+.2fpp）：" % (tot, tot / CAP * 100))
for k in (1, 2, 3, 5, 10):
    kk = g.head(k)["sum"].sum()
    print("     top%-2d 合计 %+10.0f 元 = 卖出总盈亏的 %6.1f%% = 本金 %+.2fpp | 去掉后剩 %+.0f 元"
          % (k, kk, kk / tot * 100 if tot else float("nan"), kk / CAP * 100, tot - kk))
for y in sorted(s["year"].unique()):
    sy = s[s["year"] == y]
    gy = sy.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
    top3 = "  ".join("%s %+.0f" % (a, b) for a, b in gy.head(3).items())
    print("  D. %d 年卖出 %+.0f 元 (%+.2fpp)，前 3: %s" % (y, sy["realized_pnl"].sum(),
                                                          sy["realized_pnl"].sum() / CAP * 100, top3))
print("  E. 交易过的名字 %d 只（赚过 %d 只 / 亏过 %d 只）；单笔最好 %+.0f / 最差 %+.0f 元"
      % (g.shape[0], int((g["sum"] > 0).sum()), int((g["sum"] < 0).sum()),
         s["realized_pnl"].max(), s["realized_pnl"].min()))
