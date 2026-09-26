# -*- coding: utf-8 -*-
"""为什么把清仓线抬到 P84 之后，按笔胜率反而掉了？这张表把"胜率"拆开算。

口径与 `research/sell_line.py` / `research/h5_arms.py` 完全一致：同一份
`data/cache/predictions.parquet`（horizon=5）、100 万本金、月度评估、同一套费率，
**一次只动 `sell_score`**（对照臂的旧值 0.0 写死在这里，不读 config —— config 已经搬到 P84，
读回来的话"旧线"和"新线"是同一个数，差就算不出来）。

    python research/win_rate_bridge.py     # 约 2 分钟（2 次回测）；只读缓存，不改生产代码

四个问题按顺序回答：
  §1 两档各自的胜率 / 利润因子 / 买卖笔数 —— 用的是 `backtest/metrics.py` 里同一支函数，
     所以这里的数字就是报告里那几个，不是我另算了一套
  §2 一笔买卖要跨过多少成本（实测费率 × 中位成交额）
  §3 按「完整来回」（同一只股票从建仓到清空算一段）重算胜率，并按持有交易日分桶
  §4 每笔卖出已实现盈亏的分布（中位、P10/P25/P75/P90 + 「小亏」区间占比）

结论写在 `docs/entry-位置管理.md` §8-D。
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

preds_df = pd.read_parquet("data/cache/predictions.parquet")
preds_df["date"] = pd.to_datetime(preds_df["date"])
preds = preds_df.set_index(["date", "symbol"])["prediction"]

cache = CacheManager(cfg["cache"]["directory"])
bars = {}
for s in preds_df["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df.sort_values("日期")

end = max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")
bm = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", end).set_index("日期")["收盘"]
tradable = build_tradable_mask(bars, preds.index.get_level_values("date").unique())
SC = scores_from_predictions(preds, tradable)
pol0 = policy_from_config(cfg)
ovl0 = overlay_from_config(cfg)
if pol0 is None:
    raise SystemExit("position_policy.enabled=false，没有可对照的基准档")

CAL = pd.DatetimeIndex(sorted(set().union(*[set(d["日期"]) for d in bars.values()])))
POS = {d: i for i, d in enumerate(CAL)}


def run(pol):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl0)
    with contextlib.redirect_stdout(io.StringIO()):
        return eng.run(bars, SC, benchmark_prices=bm)


def episodes(tr):
    """把成交流水拼成「完整来回」：同一只股票从持仓 0 变正、再回到 0 的一段。

    引擎每笔卖出的 realized_pnl 已经扣了这笔的卖出费 + 按比例分摊的买入成本，
    所以把一段里的卖出全加起来 = 这一段扣费后的净盈亏。
    """
    out, opened = [], {}
    for r in tr.sort_values(["date", "side"], ascending=[True, False]).itertuples():
        sym, sh = r.symbol, int(r.shares)
        if r.side == "buy":
            st = opened.get(sym)
            if st is None:
                opened[sym] = dict(entry=r.date, shares=sh, pnl=r.realized_pnl,
                                   cost=r.cost, exit=r.date)
            else:
                st["shares"] += sh
                st["cost"] += r.cost
                st["pnl"] += r.realized_pnl
        else:
            st = opened.get(sym)
            if st is None:
                continue
            st["shares"] -= sh
            st["cost"] += r.cost
            st["pnl"] += r.realized_pnl
            st["exit"] = r.date
            if st["shares"] <= 0:
                out.append(st)
                opened.pop(sym)
    out.extend(opened.values())      # 收尾还持着的：按最后一次成交日截断算一段
    ep = pd.DataFrame(out)
    ep["days"] = [POS[e] - POS[x] for e, x in zip(ep["exit"], ep["entry"])]
    return ep


ARMS = (("旧档：清仓线 0（09-26 上午那一版）", 0.0),
        (f"现行：清仓线 P84 = {pol0.sell_score:.5f}", pol0.sell_score))
res = {}
for tag, sv in ARMS:
    res[tag] = run(dc.replace(pol0, sell_score=sv))

print("=" * 92)
print("### 1. 两档的报告口径（同一份分数、同一套费率，只动 sell_score）")
for tag, _ in ARMS:
    tr = res[tag]["trades"]
    sells = tr[tr["side"] == "sell"]
    buys = tr[tr["side"] == "buy"]
    print(f"  {tag}")
    print(f"     笔数 {len(tr)}（买 {len(buys)} / 卖 {len(sells)}）| "
          f"按笔胜率 {pm.win_rate_by_trade(tr) * 100:.2f}% | "
          f"利润因子 {pm.profit_factor(tr):.2f} | "
          f"成本占本金 {tr['cost'].sum() / CAP * 100:.2f}%")
    print(f"     每笔卖出已实现盈亏：中位 {sells['realized_pnl'].median():+,.0f} 元 | "
          f"均值 {sells['realized_pnl'].mean():+,.0f} 元 | "
          f"为正的 {(sells['realized_pnl'] > 0).sum()} 笔 / {len(sells)} 笔")

print()
print("=" * 92)
print("### 2. 一单要跨过多少成本才谈得上「赚」（实测费率，含滑点）")
hurdle = {}
for tag, _ in ARMS:
    tr = res[tag]["trades"]
    b, s = tr[tr["side"] == "buy"], tr[tr["side"] == "sell"]
    bp = b["cost"].sum() / b["amount"].sum() * 1e4
    sp = s["cost"].sum() / s["amount"].sum() * 1e4
    hurdle[tag] = (bp + sp) / 100
    print(f"  {tag}")
    print(f"     买入侧 {bp:.2f}bp + 卖出侧 {sp:.2f}bp = 一个来回 {(bp + sp):.2f}bp "
          f"= {hurdle[tag]:.3f}%（价格没涨过这个数，卖出就是亏）")
    print(f"     中位单笔成交额 {tr['amount'].median():,.0f} 元 → "
          f"中位一单的固定成本 {tr['amount'].median() * (bp + sp) / 1e4:,.0f} 元")

print()
print("=" * 92)
print("### 3. 按「完整来回」（同一只股票从建仓到清空算一段）重算，并按持有交易日分桶")
for tag, _ in ARMS:
    ep = episodes(res[tag]["trades"])
    short, long = ep[ep.days <= 21], ep[ep.days > 21]
    print(f"  {tag}")
    print(f"     完整来回 {len(ep)} 段 | 段胜率 {(ep.pnl > 0).mean() * 100:.2f}% | "
          f"持有交易日中位 {ep.days.median():.0f} 天 | ≤21 天的占 "
          f"{(ep.days <= 21).mean() * 100:.1f}%")
    print(f"     ≤21 天那 {len(short)} 段：胜率 {(short.pnl > 0).mean() * 100:.2f}% | "
          f"单段盈亏中位 {short.pnl.median():+,.0f} 元 | 合计 {short.pnl.sum():+,.0f} 元")
    print(f"     >21 天那 {len(long)} 段：胜率 {(long.pnl > 0).mean() * 100:.2f}% | "
          f"单段盈亏中位 {long.pnl.median():+,.0f} 元 | 合计 {long.pnl.sum():+,.0f} 元")
    for lo, hi, lab in ((0, 21, "≤21 天(一轮月度评估)"), (22, 63, "22~63 天"),
                        (64, 9999, ">63 天")):
        sub = ep[(ep.days >= lo) & (ep.days <= hi)]
        if len(sub):
            print(f"        {lab:<22} {len(sub):3d} 段 | 胜率 {(sub.pnl > 0).mean() * 100:5.1f}% | "
                  f"中位 {sub.pnl.median():+9,.0f} 元 | 最大一段 {sub.pnl.max():+,.0f} | "
                  f"最小一段 {sub.pnl.min():+,.0f}")

print()
print("=" * 92)
print("### 4. 每笔卖出的已实现盈亏分布（元；已扣买入成本与双边费用）")
for tag, _ in ARMS:
    s = res[tag]["trades"]
    s = s[s["side"] == "sell"]["realized_pnl"]
    small_loss = ((s > -1000) & (s < 0))
    small_win = ((s > 0) & (s < 1000))
    print(f"  {tag}")
    print(f"     P10 {s.quantile(.10):+,.0f} | P25 {s.quantile(.25):+,.0f} | "
          f"中位 {s.median():+,.0f} | P75 {s.quantile(.75):+,.0f} | "
          f"P90 {s.quantile(.90):+,.0f}")
    print(f"     最好 {s.max():+,.0f} | 最差 {s.min():+,.0f} | "
          f"小亏（−1,000~0 元）{small_loss.sum()} 笔 = 占卖出 {small_loss.mean() * 100:.1f}% | "
          f"小赚（0~+1,000 元）{small_win.sum()} 笔 = {small_win.mean() * 100:.1f}%")
print()
print("读法：按笔胜率数的是「卖出那一笔亏没亏」。一单要涨过 §2 那条成本线才算赚，")
print("而成本线是固定的；把持仓时间缩短，等于把「涨幅」这个分子变小 —— 胜率就往下掉。")
