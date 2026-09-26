# -*- coding: utf-8 -*-
"""两项改动的成对对照：开盘成交判定 × 建仓线分位数（P84 → P94）。

    python research/fill_and_threshold.py

四档跑在同一份 predictions.parquet 上，一次只动一个变量，所以每档与上一档的差
就是那个变量的影响：

    A0  P84 建仓线 + 收盘涨跌幅判定   ← 2026-09-25 以前的生产口径
    A1  P84 建仓线 + 开盘涨跌幅判定   ← 只换成交判定
    A2  P94 建仓线 + 收盘涨跌幅判定   ← 只换建仓线
    A3  P94 建仓线 + 开盘涨跌幅判定   ← 两项都改（新默认档）

"收盘涨跌幅判定"是旧实现：撮合价是次日开盘价，判定却看当天收盘 vs 前收，于是
开盘跌停、收盘拉回的那天会按开盘价把卖单成交掉。本脚本把它作为对照臂临时挂回
`backtest.engine.can_fill`，生产代码里只保留开盘口径。
"""
import warnings
warnings.filterwarnings("ignore")
import contextlib
import dataclasses as dc
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import Predictor, scores_from_predictions
import backtest.engine as bt_engine
from backtest.engine import BacktestEngine
from backtest.cost import TransactionCostModel
from utils.market_rules import (at_limit_down, at_limit_up,
                                build_tradable_mask, is_suspended)
from utils.position_policy import policy_from_config
from utils.exposure import overlay_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = cb.get("initial_capital", 1_000_000)
H = int(cfg["model"]["horizon"])
BUY_Q, STRONG_Q = 0.94, 0.97        # 建仓线要落到的分位数；强线保持"最前 3%"

preds_df = pd.read_parquet("data/cache/predictions.parquet")
preds_df["date"] = pd.to_datetime(preds_df["date"])
preds = preds_df.set_index(["date", "symbol"])["prediction"]

cache = CacheManager(cfg["cache"]["directory"])
bars = {}
for s in preds_df["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df

tradable = build_tradable_mask(bars, preds.index.get_level_values("date").unique())
scores = scores_from_predictions(preds, tradable=tradable)
topk = Predictor(None, top_k=cb["max_positions"],
                 position_sizing=cb["position_sizing"]
                 ).generate_signals_from_series(preds, tradable=tradable)
end = max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")
bm = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", end).set_index("日期")["收盘"]
base_pol = policy_from_config(cfg)
overlay = overlay_from_config(cfg)
if base_pol is None:
    raise SystemExit("config.yaml 的 position_policy.enabled=false，没有可对照的基准档")

# ==================== 1. 分数分位：P94 落在哪个数值上 ====================

closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
fwd = (closes.shift(-H) / closes - 1).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
df = preds_df.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
pct = lambda v: (df.prediction < v).mean()          # 该数值在池中的分位点

print(f"### 1. 分数分布（{H} 日预测收益，样本 {len(df):,}）")
for q in (.50, .8442, .90, .94, .97, .99):
    print(f"  P{q*100:.2f} = {df.prediction.quantile(q):+.5f}")
print(f"  现行建仓线 {base_pol.buy_score:.5f} = P{pct(base_pol.buy_score)*100:.2f}"
      f" | 现行强线 {base_pol.strong_score:.5f} = P{pct(base_pol.strong_score)*100:.2f}")

NEW_BUY = float(df.prediction.quantile(BUY_Q))
NEW_STRONG = float(df.prediction.quantile(STRONG_Q))
# 迁移前那条线的**数值**（当时在旧分数分布上是 P84.57 / P96.95）。config.yaml 已经改到
# P94/P97，所以"旧线"必须写死，否则 A0/A2 两档会变成同一档，这个 2×2 对照就塌了。
OLD_BUY, OLD_STRONG = 0.00892, 0.01794
print(f"  → 新建仓线 {NEW_BUY:.5f} (P{BUY_Q*100:.0f}) / 新强线 {NEW_STRONG:.5f} "
      f"(P{STRONG_Q*100:.0f}，沿用'最前 3% 给满仓'的语义)")
print(f"  → 对照用的旧建仓线 {OLD_BUY:.5f} (本次分布上 P{pct(OLD_BUY)*100:.2f}) / "
      f"旧强线 {OLD_STRONG:.5f} (P{pct(OLD_STRONG)*100:.2f})")

per_day = (df[df.prediction >= NEW_BUY].groupby("date").size()
           .reindex(sorted(df.date.unique()), fill_value=0))
per_day_old = (df[df.prediction >= OLD_BUY].groupby("date").size()
               .reindex(sorted(df.date.unique()), fill_value=0))
print(f"\n### 2. 建仓线以上的候选只数（12 个新仓名额会不会招不满）")
for tag, s in ((f"旧线 {OLD_BUY:.5f}", per_day_old),
               (f"新线 {NEW_BUY:.5f}", per_day)):
    print(f"  {tag}: " + "  ".join(
        f"{y} 最低 {int(v.min())}/中位 {v.median():.0f}"
        f"/候选不足 12 只的日子占 {(v < 12).mean()*100:.0f}%"
        for y, v in s.groupby(lambda x: x.year)))

# ==================== 3. 四档对照 ====================

_orig_can_fill = bt_engine.can_fill


def close_based_can_fill(row, symbol, side, prev_close=None):
    """旧口径：拿当日收盘涨跌幅判涨跌停（只作对照臂用）。"""
    if row is None:
        return False, "无当日日线"
    if is_suspended(row):
        return False, "停牌/无成交"
    chg = row.get("涨跌幅")
    if chg is None or pd.isna(chg):
        return True, "ok"
    if side == "buy" and at_limit_up(symbol, chg):
        return False, f"涨停 {float(chg):+.2f}%"
    if side == "sell" and at_limit_down(symbol, chg):
        return False, f"跌停 {float(chg):+.2f}%"
    return True, "ok"


def run(label, pol, legacy_fill):
    bt_engine.can_fill = close_based_can_fill if legacy_fill else _orig_can_fill
    eng = BacktestEngine(initial_capital=CAP,
                         rebalance_frequency=cb["rebalance_frequency"],
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=overlay)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = eng.run(bars, scores, benchmark_prices=bm)
    bt_engine.can_fill = _orig_can_fill
    m, ex, eq, tr = r["metrics"], r["execution"], r["equity_curve"], r["trades"]
    # 分年口径与 verify_backtest.py / h5_arms.py 统一：年内日收益复利，基准同锚在净值区间
    yr = eq.pct_change().dropna().groupby(lambda i: i.year).apply(
        lambda x: float((1 + x).prod() - 1))
    _r = bm.pct_change()
    yrb = _r[(_r.index >= eq.index.min()) & (_r.index <= eq.index.max())].dropna() \
        .groupby(lambda i: i.year).apply(lambda x: float((1 + x).prod() - 1))
    print(f"\n{label}")
    print(f"  累计 {m['cumulative_return']:+7.2%}  年化 {m['annual_return']:+6.2%}"
          f"  波动 {m['annual_volatility']:6.2%}  Sharpe {m['sharpe_ratio']:5.2f}"
          f"  回撤 {m['max_drawdown']:6.2%}  Calmar {m.get('calmar_ratio', float('nan')):5.2f}")
    print(f"  日胜率 {m['win_rate']:.2%}  按笔胜率 {m.get('win_rate_by_trade', float('nan')):.2%}"
          f"  利润因子 {m.get('profit_factor', float('nan')):5.2f}"
          f"  超额 {m.get('excess_return', float('nan')):+.2%}"
          f"  IR {m.get('information_ratio', float('nan')):.2f}")
    print(f"  笔数 {m['total_trades']:4d}  成本 {tr['cost'].sum()/CAP:5.2%} 本金"
          f"  仓位 {ex.get('policy_mean_gross_weight', float('nan')):.1%}"
          f"  只数 {ex.get('policy_mean_names', float('nan')):.1f}"
          f"  期末 {ex.get('policy_final_names')} 只")
    print(f"  买入被挡 {sum(ex['blocked_buy'].values())} 笔 {dict(ex['blocked_buy']) or '无'}"
          f"  |  卖出被挡 {sum(ex['blocked_sell'].values())} 笔"
          f" {dict(ex['blocked_sell']) or '无'}")
    print(f"  分年 " + "  ".join(f"{y} {v:+.1%}(基准 {yrb.get(y, float('nan')):+.1%})"
                                 for y, v in yr.items()))
    return r


print("\n### 3. 四档对照（同一份预测、同一份费率、同一套暴露层）")
P84 = dc.replace(base_pol, buy_score=OLD_BUY, strong_score=OLD_STRONG)
P94 = dc.replace(base_pol, buy_score=round(NEW_BUY, 5),
                 strong_score=round(NEW_STRONG, 5))
run("A0  P84 线 + 收盘涨跌幅判定（09-25 以前的生产口径）", P84, True)
run("A1  P84 线 + 开盘涨跌幅判定（只换成交判定）", P84, False)
run("A2  P94 线 + 收盘涨跌幅判定（只换建仓线）", P94, True)
run("A3  P94 线 + 开盘涨跌幅判定  ← 两项都改", P94, False)
