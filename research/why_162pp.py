# -*- coding: utf-8 -*-
"""把 +200.92% 与 +39.10% 这两条净值逐笔拆开：钱差在哪些名字、哪些年份上。

    python research/why_162pp.py

两份 predictions.parquet 之外全部相同：同一份日线缓存、同一套费率、同一份 config。
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
BUY = float(cfg["position_policy"]["buy_score"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])

SRC = {"旧 09-25": ("data/cache/backup_20260930/predictions.parquet",
                    "data/cache/backup_20260930/factor_panel.parquet"),
       "新 09-30": ("data/cache/predictions.parquet",
                    "data/cache/factor_panel.parquet")}
preds = {}
for k, (p, fp) in SRC.items():
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    preds[k] = d
    # main.py 的 bars 池取自 factor_panel 的 symbol 列，两臂各自对齐自己那一份
    preds[k + "_univ"] = set(pd.read_parquet(fp, columns=["symbol"])["symbol"])

union = sorted(set(preds["旧 09-25"]["symbol"]) | set(preds["新 09-30"]["symbol"]))
bars_all = {}
for s in union:
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars_all[s] = df.sort_values("日期")
try:
    BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
        cb.get("benchmark", "000852"), "20210101",
        max(d["日期"].max() for d in bars_all.values()).strftime("%Y%m%d")).set_index("日期")["收盘"]
except Exception as e:
    print("基准取不到(%s)，用 None —— 不影响净值，只影响对照列" % e)
    BM = None

OUT = {}
for k, d in preds.items():
    if k.endswith("_univ"):
        continue
    bars = {s: v for s, v in bars_all.items() if s in preds[k + "_univ"]}
    PR = d.set_index(["date", "symbol"])["prediction"]
    SC = scores_from_predictions(PR, build_tradable_mask(bars, d["date"].unique()))
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
        OUT[k] = eng.run(bars, SC, benchmark_prices=BM)

eq = {k: v["equity_curve"] for k, v in OUT.items()}
tr = {k: v["trades"] for k, v in OUT.items()}
for t in tr.values():
    t["year"] = pd.to_datetime(t["date"]).dt.year

print("### 1. 两条净值")
for k in SRC:
    ser = eq[k]
    yr = ser.groupby(ser.index.year).agg(["first", "last"])
    print("  %-8s 累计 %+.2f%%  笔数 %d  成本 %.0f 元  交易过的名字 %d 只"
          % (k, pm.cumulative_return(eq[k]) * 100, len(tr[k]),
             tr[k]["cost"].sum(), tr[k]["symbol"].nunique()))
    print("      分年 " + "  ".join("%d %+.2f%%" % (y, (l / f - 1) * 100)
                                   for y, (f, l) in yr.iterrows()))

print()
print("### 2. 已实现盈亏按年（只算卖出笔，单位元）")
for y in sorted(tr["旧 09-25"]["year"].unique()):
    o = tr["旧 09-25"].query("year==@y and side=='sell'")["realized_pnl"]
    n = tr["新 09-30"].query("year==@y and side=='sell'")["realized_pnl"]
    print("  %d  旧 %12.0f (%d 笔)  新 %12.0f (%d 笔)  差 %+12.0f 元 = 占本金 %+.2fpp"
          % (y, o.sum(), len(o), n.sum(), len(n), n.sum() - o.sum(),
             (n.sum() - o.sum()) / CAP * 100))
print("  合计 旧 %+.0f  新 %+.0f  差 %+.0f 元（= 占本金 %+.2fpp）"
      % (tr["旧 09-25"].query("side=='sell'")["realized_pnl"].sum(),
         tr["新 09-30"].query("side=='sell'")["realized_pnl"].sum(),
         tr["新 09-30"].query("side=='sell'")["realized_pnl"].sum()
         - tr["旧 09-25"].query("side=='sell'")["realized_pnl"].sum(),
         (tr["新 09-30"].query("side=='sell'")["realized_pnl"].sum()
          - tr["旧 09-25"].query("side=='sell'")["realized_pnl"].sum()) / CAP * 100))

print()
print("### 3. 两条流水重合多少（按 股票+买卖方向 计）")
so = set(zip(tr["旧 09-25"]["symbol"], tr["旧 09-25"]["side"]))
sn = set(zip(tr["新 09-30"]["symbol"], tr["新 09-30"]["side"]))
print("  旧 %d 个名字-方向对，新 %d 个，两边都有 %d 个（重合度 %.2f）"
      % (len(so), len(sn), len(so & sn), len(so & sn) / max(1, len(so | sn))))
print("  只在旧里出现过的：", " ".join(sorted(s for s, _ in so - sn))[:400])
print("  只在新里出现过的：", " ".join(sorted(s for s, _ in sn - so))[:400])

print()
print("### 4. 卖出笔已实现盈亏的头尾集中度")
for k in SRC:
    s = tr[k].query("side=='sell'").sort_values("realized_pnl")
    tot = s["realized_pnl"].sum()
    top5 = s.tail(5)["realized_pnl"].sum()
    bot5 = s.head(5)["realized_pnl"].sum()
    print("  %-8s 最好5笔 %+11.0f 元（占该条合计 %.1f%%）| 最差5笔 %+11.0f 元 | 单笔中位 %+.0f"
          % (k, top5, top5 / tot * 100, bot5, s["realized_pnl"].median()))
    print("      最好5笔: " + "  ".join("%s %s %+.0f" % (r["date"].strftime("%Y-%m-%d"),
          r["symbol"], r["realized_pnl"]) for _, r in s.tail(5).iterrows()))
    print("      最差5笔: " + "  ".join("%s %s %+.0f" % (r["date"].strftime("%Y-%m-%d"),
          r["symbol"], r["realized_pnl"]) for _, r in s.head(5).iterrows()))

print()
print("### 5. 按名字汇总已实现盈亏：差额落在哪几只（占本金 pp）")
o = tr["旧 09-25"].query("side=='sell'").groupby("symbol")["realized_pnl"].sum()
n = tr["新 09-30"].query("side=='sell'").groupby("symbol")["realized_pnl"].sum()
g = pd.DataFrame({"旧": o, "新": n}).fillna(0.0)
g["差_元"] = g["新"] - g["旧"]
g["差_pp"] = g["差_元"] / CAP * 100
g = g.sort_values("差_元")
tot = g["差_元"].sum()
print("  参与过的名字 %d 只，差额合计 %+.0f 元（= %+.2fpp）" % (len(g), tot, tot / CAP * 100))
for r, lab in ((3, "前 3 只"), (5, "前 5 只"), (10, "前 10 只")):
    print("  %s 占了总差额的 %.1f%%" % (lab, g.head(r)["差_元"].sum() / tot * 100))
print("  拉低最多的 12 只：")
for s, r in g.head(12).iterrows():
    print("    %s  旧 %+9.0f 元 (%+.2fpp)  新 %+9.0f 元 (%+.2fpp)  差 %+9.0f 元 (%+.2fpp)"
          % (s, r["旧"], r["旧"] / CAP * 100, r["新"], r["新"] / CAP * 100,
             r["差_元"], r["差_pp"]))
print("  反向拉高的前 5 只：")
for s, r in g.tail(5).iloc[::-1].iterrows():
    print("    %s  旧 %+9.0f 元 (%+.2fpp)  新 %+9.0f 元 (%+.2fpp)  差 %+9.0f 元 (%+.2fpp)"
          % (s, r["旧"], r["旧"] / CAP * 100, r["新"], r["新"] / CAP * 100,
             r["差_元"], r["差_pp"]))
print("  只在一条流水里赚/亏过的名字数：旧独有 %d 只（合计 %+.0f 元）、新独有 %d 只（合计 %+.0f 元）"
      % (int((g["新"] == 0).sum()), g.loc[g["新"] == 0, "旧"].sum(),
         int((g["旧"] == 0).sum()), g.loc[g["旧"] == 0, "新"].sum()))

print()
print("### 6. 逐个调仓日的买入名单（同一天买进了谁）")
do = tr["旧 09-25"].query("side=='buy'").groupby("date")["symbol"].apply(set)
dn = tr["新 09-30"].query("side=='buy'").groupby("date")["symbol"].apply(set)
days = sorted(set(do.index) | set(dn.index))
inter = tot_u = 0
n_diff_days = 0
for d in days:
    a, b = do.get(d, set()), dn.get(d, set())
    inter += len(a & b); tot_u += len(a | b)
    if a != b:
        n_diff_days += 1
print("  买入日 %d 个：名单完全相同 %d 个、有差异 %d 个；逐日并集重合度 %.2f"
      % (len(days), len(days) - n_diff_days, n_diff_days, inter / max(1, tot_u)))
for d in days:
    a, b = do.get(d, set()), dn.get(d, set())
    if a == b:
        continue
    print("    %s  旧 %s | 新 %s  %s"
          % (d.strftime("%Y-%m-%d"), " ".join(sorted(a)) or "-",
             " ".join(sorted(b)) or "-",
             "共有 " + " ".join(sorted(a & b)) if a & b else "共有 无"))

print()
print("### 7. 平均暴露与单笔规模（为什么几只名字就能撬动整条净值）")
for k in SRC:
    ex = OUT[k]["execution"]
    b = tr[k].query("side=='buy'")
    print("  %-8s 平均目标仓位 %.1f%%  建仓笔数 %d  单笔买入中位 %.0f 元 = 本金 %.2f%%  最大一笔 %.0f 元"
          % (k, (ex.get("policy_mean_gross_weight") or 0) * 100, len(b),
             b["amount"].median(), b["amount"].median() / CAP * 100, b["amount"].max()))

print()
print("### 8. 差额按名字归属（同一只买卖过 / 只在旧里 / 只在新里）")
both = g[(g["旧"] != 0) & (g["新"] != 0)]
oo = g[g["新"] == 0]
nn = g[g["旧"] == 0]
print("  两臂都交易过的名字 %d 只，贡献差额 %+.0f 元 = %+.2fpp（占 %.1f%%）"
      % (len(both), (both["新"] - both["旧"]).sum(),
         (both["新"] - both["旧"]).sum() / CAP * 100,
         (both["新"] - both["旧"]).sum() / tot * 100))
print("  只在旧臂交易过 %d 只，贡献 %+.0f 元 = %+.2fpp（占 %.1f%%）"
      % (len(oo), -oo["旧"].sum(), -oo["旧"].sum() / CAP * 100, -oo["旧"].sum() / tot * 100))
print("  只在新臂交易过 %d 只，贡献 %+.0f 元 = %+.2fpp（占 %.1f%%）"
      % (len(nn), nn["新"].sum(), nn["新"].sum() / CAP * 100, nn["新"].sum() / tot * 100))
print("  最大单笔：300641 旧臂 2024-04-01 买入、2024-05-06 卖出，"
      "开盘 2.81 -> 18.49 = +558.0%（实测 data/cache/daily/300641.parquet）")
