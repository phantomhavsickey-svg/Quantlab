# -*- coding: utf-8 -*-
"""成本落地审计：滑点/佣金/印花税到底有没有进损益，费前费后差多少。

    python research/cost_audit.py

四件事：
  1) 逐笔恒等式：流水里记的 cost 是不是恰好等于 佣金(含 5 元下限) + 印花税(按成交日
     取档，仅卖出) + 滑点(双边)。少一项就是"算了没扣"，多一项就是重复计提。
     印花税这里按 config 的 schedule **独立重算一遍**，不复用 cost 模型的代码。
  2) 每次只免掉一项费率的五档对照（同一份 predictions.parquet、同一套仓位规则）。
     读法要小心：免掉一项只省 0.5%~2.7% 本金，若期末净值差出十几个点，那部分是
     路径敏感（成交笔数都变了），不能当成"这项成本的价值"。
  3) 记账方式：引擎是 `现金 -= amount + cost`，滑点按成交额计提，不是把成交价抬高
     0.1%。两者代数等价（cost = rate×amount = rate×qty×price），差别只在整手取整：
     股数按未抬高的开盘价算，价格口径最多少买一手。末行给出这个零头有多大。
  4) 印花税跨档：回测从 2023-02 起算，2023-08-28 那半年是万十，现行档万五。
     分档生效后这里报出跨档卖出实际多收了多少（不分档就是少收这么多）。
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
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
COMM, MINCOMM, STAMP, SLIP = (cm["commission_rate"], cm["min_commission"],
                              cm["stamp_tax_rate"], cm["slippage_rate"])
SCHED = cm.get("stamp_tax_schedule") or {}
# 独立于 TransactionCostModel 的另一份实现，用来做恒等式对照
_sched = sorted((pd.Timestamp(k), float(v)) for k, v in SCHED.items())


def stamp_rate(d):
    """成交日 d 适用的印花税率（取"生效日 ≤ d"里最新那档；早于全部则取最早档）。"""
    if not _sched:
        return STAMP
    if pd.Timestamp(d) < _sched[0][0]:
        return _sched[0][1]
    r = STAMP
    for eff, x in _sched:
        if pd.Timestamp(d) >= eff:
            r = x
    return r


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

# 指数区间跟着行情面板走（本脚本打印的都是费用数字，不读基准；写死终点只是留给以后的坑）
BM = DataDownloader(cache, **(cfg.get("download") or {})) \
    .download_index_daily(cb.get("benchmark", "000852"), "20210101",
                          max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")) \
    .set_index("日期")["收盘"]
SC = scores_from_predictions(PR, build_tradable_mask(bars, pred["date"].unique()))
POL, OVL = policy_from_config(cfg), overlay_from_config(cfg)


def run(label, **kw):
    """跑一遍现行默认档，可覆盖个别费率。"""
    rate = dict(commission=COMM, floor=MINCOMM, stamp=STAMP, slip=SLIP)
    rate.update(kw)
    model = TransactionCostModel(rate["commission"], rate["floor"],
                                 rate["stamp"], rate["slip"],
                                 None if rate["stamp"] == 0 else SCHED)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"], cost_model=model,
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=dc.replace(POL), overlay=OVL)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, SC, benchmark_prices=BM)
    eq, tr = r["equity_curve"], r["trades"]
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    cost = float(tr["cost"].sum()) if "cost" in tr.columns else 0.0
    gross = (r["execution"].get("policy_mean_gross_weight") or 0) * 100
    print("  %-22s 累计 %7.2f%%  年化 %6.2f%%  Sharpe %5.3f  回撤 %6.2f%%  "
          "笔数 %3d  成本占本金 %5.2f%%  仓位 %5.1f%%  期末净值 %.4f"
          % (label, pm.cumulative_return(eq) * 100, ann * 100,
             (ann - pm.risk_free_rate) / vol,
             pm.max_drawdown(eq)["drawdown"] * 100, len(tr), cost / CAP * 100,
             gross, eq.iloc[-1]))
    return eq, tr, cost


print("=" * 108)
print("0) 每次只免掉一项费率 —— 这些档位的差里，多少是「成本」、多少是「路径敏感」")
A = {}
A["现行"] = run("费后（现行默认）")
A["免滑点"] = run("只免滑点", slip=0.0)
A["免佣金"] = run("只免佣金", commission=0.0, floor=0.0)
A["免印花税"] = run("只免印花税", stamp=0.0)
A["全零"] = run("费率全零", commission=0.0, floor=0.0, stamp=0.0, slip=0.0)
tr, c_net = A["现行"][1], A["现行"][2]
nav = {k: v[0].iloc[-1] for k, v in A.items()}
sav = {k: (c_net - A[k][2]) / CAP * 100 for k in ("免滑点", "免佣金", "免印花税")}
print("  各档期末净值/笔数: " + "  ".join("%s %.4f/%d" % (k, nav[k], len(A[k][1]))
                                        for k in A))
print("  → 免掉任一项只省 %.1f%%~%.1f%% 本金，但期末净值跨度 %.4f（= %.0f 元 ≈ %.1f pp 累计），"
      "笔数还从 %d 变到 %d\n    所以档位间小于 10pp 的差不能读成「更好/更差」"
      % (min(sav.values()), max(sav.values()),
         max(nav.values()) - min(nav.values()),
         (max(nav.values()) - min(nav.values())) * CAP,
         (max(nav.values()) - min(nav.values())) * 100,
         min(len(v[1]) for v in A.values()), max(len(v[1]) for v in A.values())))
print("  费后 vs 费率全零：累计 %.2f%% - %.2f%% = 摩擦 %.2fpp（名义成本合计 %.2f%% 本金）"
      % (pm.cumulative_return(A["现行"][0]) * 100,
         pm.cumulative_return(A["全零"][0]) * 100,
         (pm.cumulative_return(A["全零"][0])
          - pm.cumulative_return(A["现行"][0])) * 100, c_net / CAP * 100))

print()
print("=" * 108)
print("1) 逐笔恒等式：流水 cost == 佣金(含下限) + 印花税(按成交日取档) + 滑点(双边)?")
print("   （下面所有费率单位 bp = 万分之一：佣金万三 = 3bp，千一滑点 = 10bp）")
exp_sell = (np.maximum(tr["amount"] * COMM, MINCOMM)
            + tr["amount"].to_numpy() * np.array([stamp_rate(d) for d in tr["date"]])
            + tr["amount"] * SLIP)
exp_buy = (np.maximum(tr["amount"] * COMM, MINCOMM) + tr["amount"] * SLIP)
exp = np.where(tr["side"] == "sell", exp_sell, exp_buy)
d = np.abs(exp - tr["cost"].to_numpy())
print("  笔数 %d  最大逐笔差 %.6f 元  合计 记入 %s 元 / 应为 %s 元"
      % (len(tr), d.max(), format(tr["cost"].sum(), ",.0f"), format(exp.sum(), ",.0f")))
for side, nom in (("buy", COMM + SLIP), ("sell", COMM + STAMP + SLIP)):
    t = tr[tr["side"] == side]
    amt, c = float(t["amount"].sum()), float(t["cost"].sum())
    floor = int((t["amount"] * COMM < MINCOMM).sum())
    print("  %-4s 笔数 %3d 成交额 %s 元 实收 %s 元 实测 %5.2fbp 名义(现行档) %5.2fbp "
          "佣金触 5 元下限 %d 笔 最小单 %s 元"
          % (side, len(t), format(amt, ",.0f"), format(c, ",.0f"),
             c / amt * 10000, nom * 10000, floor, format(float(t["amount"].min()), ",.0f")))
print("  单笔金额: 中位 %s 元 | 10 分位 %s | 90 分位 %s | 5 元下限只咬得到 %s 元以下的单"
      % (format(tr["amount"].median(), ",.0f"),
         format(tr["amount"].quantile(.1), ",.0f"),
         format(tr["amount"].quantile(.9), ",.0f"),
         format(MINCOMM / COMM, ",.0f")))

print()
print("=" * 108)
print("2) 价格口径 vs 现金口径：把开盘价抬高 0.1% 会少买多少股？")
b = tr[tr["side"] == "buy"]
lots = (b["amount"] * SLIP) / b["price"] / cm.get("lot_size", 100)
print("  滑点折成股数: 中位 %.3f 手 / 最大 %.3f 手 → 只有 ≥1 手才改变成交股数，"
      "本档 %d/%d 笔达到一整手（两种口径逐笔股数相同）"
      % (lots.median(), lots.max(), int((lots >= 1).sum()), len(lots)))

print()
print("=" * 108)
print("3) 印花税跨档（万十 → 万五，2023-08-28）")
rate_before = stamp_rate(pd.Timestamp("2023-08-25"))
rate_after = stamp_rate(pd.Timestamp("2023-08-28"))
m = TransactionCostModel(stamp_tax_schedule=SCHED)
agree = (m.stamp_rate_on("2023-08-25") == rate_before
         and m.stamp_rate_on("2023-08-28") == rate_after
         and m.stamp_rate_on("1995-01-01") == rate_before)
early = tr[(tr["side"] == "sell") & (tr["date"] < pd.Timestamp("2023-08-28"))]
extra = float(early["amount"].sum()) * (rate_before - rate_after)
print("  schedule %s → 2023-08-25 适用 %.0fbp、2023-08-28 适用 %.0fbp"
      "（含早于首档的日期；模型与独立算法一致: %s）" % (SCHED, rate_before * 10000,
                                                rate_after * 10000, agree))
print("  跨档前卖出 %d 笔 / %s 元 → 分档比「整段按现行档」多收 %s 元"
      "（总成本的 %.1f%%、占本金 %.3f%%）"
      % (len(early), format(float(early["amount"].sum()), ",.0f"), format(extra, ",.0f"),
         extra / c_net * 100, extra / CAP * 100))
by_year = tr.assign(y=pd.to_datetime(tr["date"]).dt.year) \
            .groupby("y").agg(成交额=("amount", "sum"), 成本=("cost", "sum"))
print("  分年成本占初始资金: " + "  ".join("%d %.2f%%" % (y, r["成本"] / CAP * 100)
                                          for y, r in by_year.iterrows()))
