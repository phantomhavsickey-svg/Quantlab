# -*- coding: utf-8 -*-
"""清仓线（sell_score）档位实测：现行 P94 建仓线之下，把清仓线从 0 一路上抬。

回答的问题有两个：清仓线抬到分数分布的哪一档，代价开始超过收益；以及同一处改动在
五种暴露层上分别值多少（§4，因为有正有负，所以它不是普适结论）。README
「清仓线重标定」那段、`docs/entry-位置管理.md` §8-C 与 config.yaml 里 sell_score 的注释出自这里。

    python research/sell_line.py         # 约 4 分钟（16 次回测）；只读缓存，不改生产代码

口径与 `research/h5_arms.py` 一致：同一份 data/cache/predictions.parquet
（horizon=5）、100 万本金、月度评估、同一套费率，一次只动 sell_score。
最后两档把暴露层摘掉复核 —— 同一改动在"带暴露层"与"纯带位"两档上的**回撤方向相反**，
这一条必须留在档案里，否则容易把 P84 当成无条件更好的线。

⚠ 抬到 P90 以上会让清仓线逼近甚至等于建仓线（0.01354），那是退化配置：
建仓后任何回撤都立刻触发清仓。本脚本把它当"上界在哪儿"的证据跑一次，不是候选档。
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
CAP = cb.get("initial_capital", 1_000_000)
H = int(cfg["model"]["horizon"])
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
    raise SystemExit("config.yaml 的 position_policy.enabled=false，没有可对照的基准档")

# ==================== 1. 分位数：与建仓线同一把尺子 ====================
closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
fwd = (closes.shift(-H) / closes - 1).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
df = preds_df.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
pct = lambda v: (df.prediction < v).mean()

print("### 1. 清仓线候选（%d 日预测收益，配对样本 %s，与建仓线同一分布）"
      % (H, f"{len(df):,}"))
for q in (.84, .90, .94):
    print("  P{:<4.0f}= {:.5f}".format(q * 100, df.prediction.quantile(q)))
print("  现行建仓线 {:.5f} = P{:.2f} | 现行清仓线 {:.5f} = P{:.2f}".format(
    pol0.buy_score, pct(pol0.buy_score) * 100,
    pol0.sell_score, pct(pol0.sell_score) * 100))
print()

# ============ 1b. 机制：月末信号日过建仓线的名字，下一轮还剩多少在清仓线上方 ============
# 回按月度评估：这一轮建仓之后下一次能动作就是下个月末，所以"一步"= 相邻两个月末信号日。
alld = pd.DatetimeIndex(sorted(set(SC.index.get_level_values("date"))))
ev = list(pd.Series(alld, index=alld).groupby(alld.to_period("M")).max())
rows = []
for i, t in enumerate(ev[:-1]):
    cur = SC.loc[t]["score"]
    nxt = SC.loc[ev[i + 1]]["score"]
    hit = cur[cur >= pol0.buy_score]
    if not len(hit):
        continue
    d = nxt.reindex(hit.index)
    rows.append(pd.DataFrame({
        "at_entry": hit, "next_score": d, "delta": d - hit}))
w = pd.concat(rows)
print("### 1b. 过建仓线那批名字的下一轮分数（%d 对相邻月末、%s 个名字-轮观测）"
      % (len(rows), f"{len(w):,}"))
print("  过线分数中位 {:.5f} | 下一轮分数变化中位 {:+.5f}（= {:+.2f}pp）"
      "| P25 {:+.5f} | P75 {:+.5f}".format(
          w.at_entry.median(), w.delta.median(), w.delta.median() * 100,
          w.delta.quantile(.25), w.delta.quantile(.75)))
for lab, v in (("建仓线", pol0.buy_score), ("清仓线(现行 P84)", pol0.sell_score),
               ("清仓线若抬到 P90", 0.01095)):
    below = (w.next_score < v).mean()
    print("  下一轮跌破{} {:.5f} 的比例 {:.1%}".format(lab, v, below))
print("  → 建仓线只比清仓线高 {:.5f}：一轮分数的中位回撤就够跨过它，".format(
    pol0.buy_score - pol0.sell_score))
print("    这就是把清仓线抬到 P84 后 `trim`(减仓) 几乎不再发生、`exit`(直接清仓) 翻倍的原因。")
print()


def run(tag, sell, ovl):
    pol = dc.replace(pol0, sell_score=sell)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, SC, benchmark_prices=bm)
    eq, tr, ex = r["equity_curve"], r["trades"], r["execution"]
    cost = float(tr["cost"].sum()) if "cost" in tr.columns else 0.0
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    acts = ex.get("policy_actions") or {}
    print("%-30s 累计 %7.2f%% 年化 %6.2f%% 波动 %5.2f%% Sharpe %5.3f 回撤 %5.2f%%"
          " | 笔数 %3d 成本 %4.2f%% 仓位 %4.1f%% 只数 %3.1f" %
          (tag, pm.cumulative_return(eq) * 100, ann * 100, vol * 100,
           (ann - pm.risk_free_rate) / vol, pm.max_drawdown(eq)["drawdown"] * 100,
           len(tr), cost / CAP * 100,
           (ex.get("policy_mean_gross_weight") or 0) * 100,
           ex.get("policy_mean_names") or 0))
    print("%-30s 动作 %s" % ("", {k: acts.get(k, 0) for k in
                                ("entry", "add", "trim", "exit_by_trim", "exit",
                                 "de_gross")}))
    yr = rets.groupby(rets.index.year).apply(lambda x: float((1 + x).prod() - 1))
    print("%-30s 分年 %s" % ("", " ".join("%d %+5.1f%%" % (y, v * 100)
                                          for y, v in yr.items())))


print("### 2. 带暴露层 A+B（现行档）：一次只动 sell_score")
for tag, sv in (("清仓线 0（09-26 上午那一版）", 0.0),
                ("清仓线 P84 = 0.00875 ← 现行", 0.00875),
                ("清仓线 P90 = 0.01095", 0.01095),
                ("清仓线 = 建仓线 0.01354", 0.01354)):
    run(tag, sv, ovl0)

print()
print("### 3. 摘掉暴露层复核（纯带位，其余参数不变）")
for tag, sv in (("清仓线 0", 0.0), ("清仓线 P84 = 0.00875", 0.00875)):
    run(tag, sv, None)


# ==================== 4. 同一处改动在五种暴露层上的边际 ====================
# 这一节存在的唯一理由：一句"清仓线抬到 P84 值多少"在五个不同的暴露层档位上给出五个
# 有正有负的答案（现行只值几 pp，静态上限那两档一个 −67pp 一个 +74pp）。少了这张表，
# §2 那一行很容易被读成"清仓线越紧越好"这种普适结论。
# 对照臂的旧值 0.0 写死在这里，不读 config —— config 已经搬到 P84，读回来的话
# "旧线"和"新线"是同一个数，边际算不出来。
def brief(pol, ovl):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, SC, benchmark_prices=bm)
    eq, tr, ex = r["equity_curve"], r["trades"], r["execution"]
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    return dict(cum=pm.cumulative_return(eq) * 100,
                sharpe=(ann - pm.risk_free_rate) / vol,
                dd=pm.max_drawdown(eq)["drawdown"] * 100,
                n=len(tr), cost=float(tr["cost"].sum()) / CAP * 100,
                gross=(ex.get("policy_mean_gross_weight") or 0) * 100,
                names=ex.get("policy_mean_names") or 0)


print()
print("### 4. 清仓线 0 → 现行（P84）在五种暴露层上的边际")
print("     （config 的 sell_score 现为 {:.5f} = P{:.2f}；本节拿它和写死的 0 对照）"
      .format(pol0.sell_score, pct(pol0.sell_score) * 100))
for tag, pol, ovl in (
        ("现行 A+B", pol0, ovl0),
        ("只开 A（IC 门控，不缩波动）", pol0, dc.replace(ovl0, vol_target_ann=0.0)),
        ("只开 B（目标波动，无门控）", pol0, dc.replace(ovl0, ic_window_days=0)),
        ("静态上限 45% / 名额 7",
         dc.replace(pol0, max_total_pct=0.45, max_names=7), None),
        ("静态上限 30% / 名额 4",
         dc.replace(pol0, max_total_pct=0.30, max_names=4,
                    add_reserve_weight=0.02), None)):
    a, b = brief(dc.replace(pol, sell_score=0.0), ovl), brief(dc.replace(pol, sell_score=pol0.sell_score), ovl)
    print("  %-26s 累计 %+7.2f%% → %+7.2f%%  边际 %+7.2fpp | Sharpe %.3f → %.3f"
          " | 回撤 %5.2f%% → %5.2f%%（%+5.2fpp）| 笔数 %3d → %3d | 成本 %4.2f%% → %4.2f%%"
          % (tag, a["cum"], b["cum"], b["cum"] - a["cum"], a["sharpe"], b["sharpe"],
             a["dd"], b["dd"], b["dd"] - a["dd"], a["n"], b["n"], a["cost"], b["cost"]))
