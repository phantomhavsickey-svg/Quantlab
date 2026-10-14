# -*- coding: utf-8 -*-
"""换方向回测对照:0=现行信号(基线),A=模型预测取反,B=合成因子取反。

    python research/flip_arms.py

口径照 main.py / research/panel_arms.py:同一回测引擎、同一成本、同一暴露层,
只换信号列,并把带位阈值按各臂信号自己的分位数(P84/P94/P97)重标。
生产 predictions.parquet 只读不写。
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
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
logger.remove()   # 静音引擎 DEBUG 日志,只留本脚本自己的输出

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
mcfg, cb, cm = cfg["model"], cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
panel = pd.read_parquet("data/cache/factor_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
fcols = [c for c in panel.columns if c not in ("date", "symbol")]

print("### 换方向对照 flip_arms")
print("预测 %d 行 %d 日 %d 只 %s~%s | 面板 %d 因子"
      % (len(pred), pred["date"].nunique(), pred["symbol"].nunique(),
         pred["date"].min().date(), pred["date"].max().date(), len(fcols)))

# 合成因子:每日各因子百分位排名的均值(截面等权),再取反
sub = panel[panel["date"].isin(set(pred["date"]))].copy()
sub["comp"] = sub.groupby("date")[fcols].rank(pct=True).mean(axis=1)

arms = {}
arms["0 现行"] = pred[["date", "symbol", "prediction", "forward_return"]].copy()
a = pred[["date", "symbol", "prediction", "forward_return"]].copy()
a["prediction"] = -a["prediction"]
arms["A 模型取反"] = a
b = pred[["date", "symbol", "forward_return"]].merge(
    sub[["date", "symbol", "comp"]], on=["date", "symbol"], how="left")
b["prediction"] = 0.5 - b["comp"]   # 平移常数:排序与阈值相对位置不变,只为绕开"线不能为负"自检
arms["B 因子取反"] = b.dropna(subset=["prediction"])[
    ["date", "symbol", "prediction", "forward_return"]]


def daily_ic(df):
    def f(g):
        if len(g) < 20:
            return np.nan
        return g["prediction"].rank().corr(g["forward_return"].rank())
    s = df.dropna(subset=["forward_return"]).groupby("date").apply(f)
    s = s.dropna()
    return s.mean(), len(s)


for tag, df in arms.items():
    ic, n = daily_ic(df)
    q = df.dropna(subset=["forward_return"])["prediction"].quantile(
        [0.03, 0.06, 0.16, 0.84, 0.94, 0.97])
    print("\n-- %s: %d 条 | 日均 RankIC %+.4f (%d 日)"
          % (tag, len(df), ic, n))
    print("   P84 %.5f  P94 %.5f  P97 %.5f  (镜像 P16 %.5f)"
          % (q[0.84], q[0.94], q[0.97], q[0.06]))

# 行情与基准(照 panel_arms)
bars_all = {}
for s in sorted(pred["symbol"].unique().tolist()):
    df = cache.get_daily(s)
    if df is not None:
        bars_all[s] = df.sort_values("日期")
print("\n行情 %d 只" % len(bars_all))
try:
    BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
        cb.get("benchmark", "000852"), "20210101",
        max(d["日期"].max() for d in bars_all.values()).strftime("%Y%m%d")).set_index("日期")["收盘"]
except Exception as e:
    print("  基准取不到(%s)" % e)
    BM = None


def run_arm(tag, sig_df, thr, arm_id):
    PR = sig_df.set_index(["date", "symbol"])["prediction"]
    SC = scores_from_predictions(PR, build_tradable_mask(bars_all, sig_df["date"].unique()))
    pol = dc.replace(policy_from_config(cfg), **thr) if thr else policy_from_config(cfg)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule") or None),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=overlay_from_config(cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars_all, SC, benchmark_prices=BM)
    eq, tr = r["equity_curve"], r["trades"]
    os.makedirs("data/cache/_probe_flip", exist_ok=True)
    tr.to_parquet("data/cache/_probe_flip/%s_trades.parquet" % arm_id)
    dd = pm.max_drawdown(eq)
    dr = eq.pct_change().dropna()
    print("\n### %s" % tag)
    print("  阈值 buy %.5f / sell %.5f / strong %.5f / add %.5f / floor %.5f"
          % (pol.buy_score, pol.sell_score, pol.strong_score,
             pol.add_score_line, pol.step_score_floor))
    print("  累计 %+8.2f%%  夏普 %5.2f  回撤 %.2f%%  笔数 %4d  成本 %7.0f 元  平均目标仓位 %.1f%%  平均持仓 %.1f 只"
          % (pm.cumulative_return(eq) * 100, pm.sharpe_ratio(dr) or float("nan"),
             dd["drawdown"] * 100,
             len(tr), tr["cost"].sum(),
             (r["execution"].get("policy_mean_gross_weight") or 0) * 100,
             r["execution"].get("policy_mean_names", 0.0)))
    yr = eq.groupby(eq.index.year).agg(["first", "last"])
    print("  分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                               for y, (f, l) in yr.iterrows()))
    return tr


def thr_from(sig_df, floor_ratio=None):
    q = sig_df.dropna(subset=["forward_return"])["prediction"].quantile([0.84, 0.94, 0.97])
    thr = {"buy_score": float(q[0.94]), "sell_score": float(q[0.84]),
           "strong_score": float(q[0.97]), "add_score_line": float(q[0.97])}
    if floor_ratio is not None:
        thr["step_score_floor"] = floor_ratio * (float(q[0.97]) - float(q[0.94]))
    return thr


FLOOR_RATIO = 0.005 / (0.01807 - 0.01354)   # 生产 floor 相对 P97−P94 的倍数 ≈1.10
T0 = run_arm("0 现行信号(基线)", arms["0 现行"], None, "0")
TA = run_arm("A 模型预测取反", arms["A 模型取反"], thr_from(arms["A 模型取反"]), "A")
TB = run_arm("B 合成因子取反", arms["B 因子取反"], thr_from(arms["B 因子取反"], FLOOR_RATIO), "B")

# ==================== 选股差异 ====================
import itertools

print("\n### 选股差异")
trades = {"0 现行": T0, "A 模型取反": TA, "B 因子取反": TB}
sets, pnl = {}, {}
for tag, tr in trades.items():
    bu = tr[tr["side"] == "buy"]
    sets[tag] = set(bu["symbol"])
    sp = (tr[tr["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
          .sort_values(ascending=False))
    pnl[tag] = sp
    tot = float(sp.sum())
    print(f"-- {tag}: 买入 {len(bu)} 笔/{len(sets[tag])} 只 | 卖出已实现合计 {tot:+,.0f} 元 "
          f"| top2 占 {sp.head(2).sum()/tot*100:.0f}% top10 占 {sp.head(10).sum()/tot*100:.0f}%")
    print("   卖利前 5: " + " | ".join(f"{s} {v:+,.0f}" for s, v in sp.head(5).items()))

for a, b2 in itertools.combinations(trades, 2):
    inter = sets[a] & sets[b2]
    print(f"-- {a} 与 {b2}: 共买 {len(inter)} 只 | 只在 {a} {len(sets[a]-sets[b2])} 只 "
          f"| 只在 {b2} {len(sets[b2]-sets[a])} 只 | 交集大小/并集大小 {len(inter)/len(sets[a]|sets[b2]):.2f}")

bm = {}
for tag, tr in trades.items():
    bu = tr[tr["side"] == "buy"].copy()
    bu["m"] = bu["date"].dt.to_period("M")
    bm[tag] = bu.groupby("m")["symbol"].apply(set)
for a, b2 in itertools.combinations(trades, 2):
    months = [m for m in bm[a].index.intersection(bm[b2].index)
              if bm[a][m] | bm[b2][m]]
    js = [len(bm[a][m] & bm[b2][m]) / len(bm[a][m] | bm[b2][m]) for m in months]
    print(f"-- {a} vs {b2}: 同月有买入 {len(months)} 个月, 月内买入名单交集/并集中位 {np.median(js):.2f}")

only0 = (pnl["0 现行"].reindex(sorted(sets["0 现行"] - sets["B 因子取反"]))
         .dropna().sort_values(ascending=False))
onlyB = (pnl["B 因子取反"].reindex(sorted(sets["B 因子取反"] - sets["0 现行"]))
         .dropna().sort_values(ascending=False))
both0b = pnl["0 现行"].reindex(sorted(sets["0 现行"] & sets["B 因子取反"])).dropna()
print(f"-- 只在 0 买入且卖出过的 {len(only0)} 只 = {only0.sum():+,.0f} 元; "
      "前 3: " + " | ".join(f"{s} {v:+,.0f}" for s, v in only0.head(3).items()))
print(f"-- 只在 B 买入且卖出过的 {len(onlyB)} 只 = {onlyB.sum():+,.0f} 元; "
      "前 3: " + " | ".join(f"{s} {v:+,.0f}" for s, v in onlyB.head(3).items()))
print(f"-- 0 与 B 共买且卖出过的 {len(both0b)} 只 = {both0b.sum():+,.0f} 元: "
      + " | ".join(f"{s} {v:+,.0f}" for s, v in both0b.items()))
