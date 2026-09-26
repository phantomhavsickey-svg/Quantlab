# -*- coding: utf-8 -*-
"""分数带位策略的阈值标定 + 全档实测（README「阈值标定与全档实测」的表出自这里，
§5 是组合级暴露层 A/B 的同口径对比）。

    python research/policy_grid.py

不改任何生产代码：读 config.yaml 的 position_policy 当基准档，其余档位用
dataclasses.replace 派生。predictions.parquet 与因子都不重算，重训之后直接重跑
本脚本即可重新标定。耗时主要在缓存载入（约 1 分钟），每个档位本身几秒。

⚠ 这一版整套是在 horizon=20 的旧分数上跑的（README 已标成"h20 旧表"），且它打印的
"分年"用的是**年内首末收盘价相除**的旧口径（漏掉每年第一个交易日的跳空）——与
`verify_backtest.py` / `h5_arms.py` / `fill_and_threshold.py` 现在的"年内日收益复利"
口径不同，不要跨表比逐年数字。
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
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import numpy as np
import pandas as pd
import yaml

from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import Predictor, scores_from_predictions
from backtest.engine import BacktestEngine
from backtest.cost import TransactionCostModel
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = cb.get("initial_capital", 1_000_000)
# 标签周期必须跟着模型走： predictions.parquet 里的分数预测的是 horizon 日收益,
# 拿固定的 20 日收益去评它,分位点和分桶表都是在说另一件事。
H = int(cfg["model"]["horizon"])

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

tradable = build_tradable_mask(
    bars, preds.index.get_level_values("date").unique())
scores = scores_from_predictions(preds, tradable=tradable)
topk = Predictor(None, top_k=cb["max_positions"],
                 position_sizing=cb["position_sizing"]
                 ).generate_signals_from_series(preds, tradable=tradable)
end = max(df["日期"].max() for df in bars.values()).strftime("%Y%m%d")
bm = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", end).set_index("日期")["收盘"]
base_pol = policy_from_config(cfg)
if base_pol is None:
    raise SystemExit("config.yaml 的 position_policy.enabled=false，没有基准档可扫")


# ==================== 1. 分数分布：阈值该定在哪 ====================

closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
fwd = (closes.shift(-H) / closes - 1).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
df = preds_df.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
df["pct"] = df.groupby("date")["prediction"].rank(ascending=False, method="first") \
    / df.groupby("date")["prediction"].transform("size")
print(f"### 1. 分数 = 预测 {H} 日收益率的截面分布")
print("  " + "  ".join(f"P{int(q*100)} {df.prediction.quantile(q):+.2%}"
                       for q in (.5, .85, .95, .99))
      + f"  | 均值 {df.prediction.mean():+.2%} σ {df.prediction.std():.2%}"
      f" | 实际 {H} 日均益 {df.fwd.mean():+.2%} | 样本 {len(df):,}")
lab = ["top1.2%", "1.2-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
df["b"] = pd.cut(df.pct, [0, .012, .05, .10, .20, .50, 1.0], labels=lab)
g = df.groupby("b", observed=True).agg(均益=("fwd", "mean"),
                                       胜率=("fwd", lambda x: (x > 0).mean()))
print(f"\n按每日分数名次分桶 → 组内实际 {H} 日收益（头部越平，越说明分数线不是 alpha 旋钮）")
print("  " + "  ".join(f"{k} {v.均益:+.2%}/{v.胜率:.0%}" for k, v in g.iterrows()))
for y, sub in df.groupby(lambda i: df.date[i].year):
    gg = sub.groupby("b", observed=True).fwd.mean()
    print(f"  {y}: " + "  ".join(f"{k}:{v:+.2%}" for k, v in gg.items()))
a = df[df.pct <= .012].groupby(df.date.dt.to_period("M")).fwd.mean()
b = df[df.pct <= .05].groupby(df.date.dt.to_period("M")).fwd.mean()
d = (a - b).dropna()
print(f"  Top-12 减 Top-50 月差：均值 {d.mean():+.2%}，为正 {int((d > 0).sum())}/{len(d)} 个月，"
      f"区间 {d.min():+.2%} ~ {d.max():+.2%}")

print("\n### 2. 建仓线下每日过线只数（分数线会不会自己变成暴露约束？）")
for th in (0.01, 0.02, 0.03, 0.05):
    per = df[df.prediction >= th].groupby("date").size() \
        .reindex(sorted(df.date.unique()), fill_value=0)
    yr = per.groupby(lambda x: x.year).agg(["min", "median"])
    print(f"  buy_score {th:.2f}: 全样本日均 {per.mean():.0f} 只 | "
          + " ".join(f"{y} 最低 {int(r['min'])}/中位 {r['median']:.0f}"
                     for y, r in yr.iterrows()))


# ==================== 3. 档位实测 ====================

def run(label, pol, freq="monthly", top=False, score_df=None, quiet=True,
        ovl=None):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency=freq,
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule")),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=None if top else pol, overlay=ovl)
    buf = io.StringIO()                      # 引擎自带 print() 摘要，扫档时太吵
    with contextlib.redirect_stdout(buf if quiet else sys.stdout):
        r = eng.run(bars, topk if top else (scores if score_df is None else score_df),
                    benchmark_prices=bm)
    m, ex, eq, tr = r["metrics"], r["execution"], r["equity_curve"], r["trades"]
    yr = eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    gr = ex.get("policy_mean_gross_weight")
    nm = ex.get("policy_mean_names")
    extra = ""
    if "exposure_mean_cap" in ex:
        extra = (f" |仓位上限均 {ex['exposure_mean_cap']:.0%}"
                 f"/最低 {ex['exposure_min_cap']:.0%}"
                 f"|门控 {ex['n_gated_evals']}/{len(ex['exposure'])}")
    print(f"{label:32s} 累计 {m['cumulative_return']:+7.1%} 年化 {m['annual_return']:+6.1%} "
          f"波动 {m['annual_volatility']:5.1%} Sharpe {m['sharpe_ratio']:5.2f} "
          f"回撤 {m['max_drawdown']:6.2%} "
          f"({str(m.get('max_drawdown_peak_date'))[:10]}→"
          f"{str(m.get('max_drawdown_trough_date'))[:10]}) "
          f"笔数 {m['total_trades']:5d} 成本 {tr['cost'].sum()/CAP:5.2%} "
          + (f"仓位 {gr:.0%} 只数 {nm:.1f}" if gr is not None else "Top-50 等权")
          + extra
          + "  分年 " + " ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
    return r


print("\n### 3. 回测档位（同一份 predictions，未重训）")
run("Top-50 等权（enabled:false 口径）", None, top=True)
for th in (0.01, 0.02, 0.03, 0.05):
    run(f"buy={th:.2f} sell={base_pol.sell_score:.2f} 12只",
        dc.replace(base_pol, buy_score=th, strong_score=th + 0.02))
run("buy=0.03 sell=0.01", dc.replace(base_pol, sell_score=0.01))
for n in (20, 25):
    run(f"buy=0.03 {n}只(base3%)",
        dc.replace(base_pol, base_weight=0.03, min_hold_weight=0.02, max_names=n))
run("buy=0.03 周度评估", base_pol, freq="weekly")
run("buy=0.03 周度 20只(base3%)",
    dc.replace(base_pol, base_weight=0.03, min_hold_weight=0.02, max_names=20),
    freq="weekly")
for cap in (0.80, 0.60):
    run(f"buy=0.03 总仓位上限{cap:.0%}",
        dc.replace(base_pol, max_total_pct=cap, add_reserve_weight=0.05,
                   max_names=int((cap - 0.05) / 0.0501)))

print("\n### 4. 组合级趋势门控（目前代码里还没有，靠摘掉分数模拟）")
dates = sorted(scores.index.get_level_values("date").unique())
for n in (100, 200):
    off = (bm < bm.rolling(n).mean()).reindex(dates).fillna(False).astype(bool)
    sc2 = scores.copy()
    sc2.loc[sc2.index.get_level_values("date").map(dict(zip(dates, off))), "score"] = -1.0
    print(f"  MA{n}: 空仓日 {off.mean():.1%}，趋势翻转 "
          f"{int((np.diff(off.astype(int)) == 1).sum())} 次")
    run(f"buy=0.03 + 跌破 MA{n} 强制空仓", base_pol, score_df=sc2)

print("\n### 5. 组合级暴露层 A/B（utils/exposure.py，分数线与名额沿用基准档）")
from utils.exposure import OverlayConfig


def OC(**kw):
    """默认两条通道都关，只留传进来的那条。"""
    d = dict(vol_target_ann=0.0, ic_window_days=0)
    d.update(kw)
    return OverlayConfig(enabled=True, **d)


def diag(tag, r):
    ex = r["execution"]
    er = ex["exposure"]
    g = [str(d)[:7] for d, gg in er["gated"].items() if gg]
    print(f"  {tag} 诊断: 仓位上限均值 {ex['exposure_mean_cap']:.1%}"
          f"(最低 {ex['exposure_min_cap']:.1%})，IC 门控触发 "
          f"{ex['n_gated_evals']}/{len(er)} 次评估: "
          + (" ".join(g[:24]) if g else "从未触发"))
    rv = ex["mean_realized_vol"]
    print(f"    全池已实现波动均值 "
          + ("未启用" if rv != rv else f"{rv:.1%}")
          + f"，窗口 RankIC 均值 "
          f"{ex['mean_window_ic']:+.4f}/最低 {ex['min_window_ic']:+.4f}；上限最低的 "
          "5 次评估 " + " ".join(f"{str(i)[:10]} {v:.0%}"
                                 for i, v in er["cap"].nsmallest(5).items()))


run("基准：无暴露层", base_pol)
r_a = run("A IC 门控 60 日", base_pol, ovl=OC(ic_window_days=60))
for tv in (0.08, 0.10, 0.15, 0.20):
    run(f"B 目标波动 {tv:.0%}", base_pol, ovl=OC(vol_target_ann=tv))
for w in (40, 120):
    run(f"A 门控窗 {w} 日", base_pol, ovl=OC(ic_window_days=w))
r_ab = {}
for tv in (0.10, 0.12, 0.15):
    r_ab[tv] = run(f"A + B 目标波动 {tv:.0%}", base_pol,
                   ovl=OC(vol_target_ann=tv, ic_window_days=60))
# 对照组:B 是不是只是"静态把仓位压低"的马甲 —— 若同上限的静态档打出同样的
# Sharpe/回撤,那波动缩放就没有增量信息,不该为它多养一层代码。名额按同口径
# 收缩到装得进上限(与 §3 的静态档写法一致)。
for cap in (0.45, 0.30):
    static = dc.replace(base_pol, max_total_pct=cap, add_reserve_weight=0.05,
                        max_names=int((cap - 0.05) / 0.0501))
    run(f"对照:静态上限 {cap:.0%}({static.max_names}只)", static)
    run(f"对照:静态上限 {cap:.0%}({static.max_names}只) + A 门控",
        static, ovl=OC(ic_window_days=60))

diag("A(60)", r_a)
for tv, r in r_ab.items():
    diag(f"A+B{tv:.0%}", r)
