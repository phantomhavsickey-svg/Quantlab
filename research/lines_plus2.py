# -*- coding: utf-8 -*-
"""把三条分位线各抬 2 个百分点，补仓线"关 / 开在原档 P97 / 跟着抬到 P99"三种状态全交叉，
再叠四种暴露层 —— 一张 60 格的大对照。

回答的三个问题（按用户提的顺序）：
  1. **"所有档位提高两个百分点"本身值多少钱**：清仓线 P84→P86、建仓线 P94→P96、
     建仓满仓线 P97→P99。既一起抬（L4），也一条一条单独抬（L1/L2/L3），这样才知道
     收益差是哪条线给的。
  2. **抬完之后那条绝对补仓线还开不开**：开的话是留在原来那档 P97（= 0.01807 这个
     死数不动），还是跟着升到 P99（= 0.02888）。两档都跑，不假设"跟着升才对"。
  3. **同一处改动在松/紧的仓位约束下分别值多少**：四种暴露层（现行 A+B、静态 45%/7 只、
     静态 30%/4 只、纯带位）。上一轮的教训就是只看现行层会读错 —— 补仓线在现行层
     +38.83pp、在 30%/4 层 −65.97pp（`logs/add_line.log` §4、`logs/add_line_2024.log` §0）。

口径与 `research/add_line.py` / `research/sell_line.py` / `research/h5_arms.py` 完全一致：
同一份 `data/cache/predictions.parquet`（`model.horizon: 5`）、100 万本金、月度评估、
同一套费率（佣金+印花税+滑点）、所有主数字**费后**；一次只动写明白的那几条线，
名额 / 单票上限 / Δ 步长 / 减仓条件一律沿用 config 现值。

⚠ 两条读数注意事项（脚本 §1、§5 当场把数打出来）：
  1. **抬建仓线会让 Δ 跟着变大**：Δ = max(0.005, 建仓分数 − 建仓线)，建仓线抬到 P96
     之后"首次触发补仓所需的最低分数"= 建仓线 + 0.005 也一起上移，所以"台阶①"和
     "绝对线②"谁更松会换边 —— 补仓线的效果不是上一轮的复刻。
  2. **抬线会把候选池削薄**：建仓线从 P94 到 P96，评估日过线只数直接掉一截，12 个
     名额可能招不满；§5 给每个线档的"过线只数中位/最低/不足 12 只的日子占比"。
     欠配（想买买不满、钱花不出去）是这组对照最容易读反的地方。

跑法：

    python research/lines_plus2.py > logs/lines_plus2.log 2>/dev/null
    # 60 次回测；本机同码连跑 3 次，实测脚本自计 88.5 / 121.9 / 148.1 秒（墙钟 1.5~2.5 分钟，
    # 随机器负载波动），除最后一行的秒数外**逐字节一致**（215 行里只有 1 行不同）。
    # stderr 是引擎进度条，必须丢掉，否则盖掉对照表。
"""
import contextlib
import dataclasses as dc
import io
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

logger.remove()          # 60 次回测的 DEBUG 会盖掉对照表，日志自己打印
T0 = time.time()
RUNS = {"n": 0}

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

# ==================== 1. 尺子：把"抬两个百分点"翻译成分数 ====================
# 与 calib_stats.py §1 同口径（预测与前向 h 日实际收益成对出现的观测），这里配对样本
# 与非配对样本行数相同（854,763），所以文档里引用的 P84/P94/P97 就是这一把尺。
close = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
fwd = (close.shift(-H) / close - 1.0).stack().rename("actual").reset_index()
fwd.columns = ["date", "symbol", "actual"]
fwd["date"] = pd.to_datetime(fwd["date"])
pair = preds_df.merge(fwd[["date", "symbol", "actual"]], on=["date", "symbol"], how="inner")
S = pair["prediction"]
pct = lambda v: (S < v).mean() * 100  # noqa: E731
Q = {p: round(float(np.percentile(S, p)), 5) for p in (84, 86, 94, 96, 97, 99)}

print("=" * 104)
print("### 1. 尺子：分位线 → 分数（配对样本 %s 行 / %d 个信号日 / %d 只，horizon=%d）"
      % (f"{len(S):,}", pair["date"].nunique(), pair["symbol"].nunique(), H))
for p in (84, 86, 94, 96, 97, 99):
    print(f"    P{p:<4} = {Q[p]:.5f}")
print(f"  config 现行三线：清仓 {pol0.sell_score:.5f}（P{pct(pol0.sell_score):.2f}）、"
      f"建仓 {pol0.buy_score:.5f}（P{pct(pol0.buy_score):.2f}）、"
      f"满仓 strong {pol0.strong_score:.5f}（P{pct(pol0.strong_score):.2f}）")
print(f"  换算：+2 个百分点 = 清仓线 {Q[84]:.5f}→{Q[86]:.5f}（+{(Q[86]-Q[84])*1e4:.2f}bp）、"
      f"建仓线 {Q[94]:.5f}→{Q[96]:.5f}（+{(Q[96]-Q[94])*1e4:.2f}bp）、"
      f"满仓线 {Q[97]:.5f}→{Q[99]:.5f}（+{(Q[99]-Q[97])*1e4:.2f}bp）")
print(f"  ⚠ 顶端是稀的：P94→P96 只差 {(Q[96]-Q[94])*1e4:.0f}bp，而 P97→P99 差 "
      f"{(Q[99]-Q[97])*1e4:.0f}bp —— 同样'抬 2 个百分点'，满仓线那一抬比另外两条狠得多。")

LINES = (
    ("L0 现行三线 P84/P94/P97", Q[84], Q[94], Q[97]),
    ("L1 只抬清仓线 → P86", Q[86], Q[94], Q[97]),
    ("L2 只抬建仓线 → P96", Q[84], Q[96], Q[97]),
    ("L3 只抬满仓线 → P99", Q[84], Q[94], Q[99]),
    ("L4 三条全抬 +2pp", Q[86], Q[96], Q[99]),
)
ADDS = (("关", 0.0), ("开 P97", Q[97]), ("开 P99", Q[99]))

print()
print("  补仓的两条标准谁更松（①台阶：建仓分数 ≥ 建仓线 + Δ 下限 "
      f"{pol0.step_score_floor:.5f}；②绝对线）—— 逐档实算：")
for tag, sv, bv, tv in LINES:
    first = bv + pol0.step_score_floor
    for lab, lv in ADDS:
        if lv == 0:
            print(f"    {tag:<24} {lab:<7} 只剩①，①首触最低分数 = {first:.5f}"
                  f" = P{pct(first):.2f}")
        else:
            print(f"    {tag:<24} {lab:<7} ②={lv:.5f}（P{pct(lv):.2f}）vs ①首触 "
                  f"{first:.5f}（P{pct(first):.2f}）→ "
                  f"{'②更松，能在没涨够一档时补' if lv < first else '②更严，几乎只在①成立时一起成立'}")
print("  （②只触发、不推进参考分数，所以它不改变减仓条件；建仓满仓线 strong 与补仓线")
print("   在 L0/L3/L4 上会取到同一个数值，那是**两条不同规则**共用一把尺子，不是重复参数。）")

# ==================== 候选池：抬线会不会把名额招不满 ====================
alld = pd.DatetimeIndex(sorted(set(SC.index.get_level_values("date"))))
EV = list(pd.Series(alld, index=alld).groupby(alld.to_period("M")).max())
NAMES = pol0.max_names


def pool(line):
    n = pd.Series([len(SC.loc[t]["score"][lambda x: x >= line]) for t in EV],
                  index=EV)
    return n


print()
print(f"### 1b. 评估日过建仓线只数（名额 {NAMES} 只，45 个月末评估日）")
for lab, bv in (("现行 P94", Q[94]), ("抬到 P96", Q[96])):
    n = pool(bv)
    print(f"    {lab} 线 {bv:.5f}：中位 {n.median():5.1f} 只、最低 {n.min():4.0f} 只"
          f"（{n.idxmin():%Y-%m}）、不足 {NAMES} 只的评估日 {(n < NAMES).mean() * 100:4.1f}%"
          f"、平均 {n.mean():5.1f} 只")
for lab, tv in (("现行满仓线 P97", Q[97]), ("抬到 P99", Q[99])):
    n = pool(tv)
    print(f"    {lab} {tv:.5f}：中位 {n.median():5.1f} 只、最低 {n.min():4.0f} 只"
          f"（{n.idxmin():%Y-%m}）、不足 {NAMES} 只的评估日 {(n < NAMES).mean() * 100:4.1f}%")


# ==================== 跑一次回测取指标 ====================
def brief(pol, ovl):
    RUNS["n"] += 1
    eng_ = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                          max_positions=cb["max_positions"],
                          cost_model=TransactionCostModel(
                              cm["commission_rate"], cm["min_commission"],
                              cm["stamp_tax_rate"], cm["slippage_rate"],
                              cm.get("stamp_tax_schedule")),
                          lot_size=int(cm.get("lot_size", 100)),
                          policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng_.run(bars, SC, benchmark_prices=bm)
    eq, tr, ex = r["equity_curve"], r["trades"], r["execution"]
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    acts = ex.get("policy_actions") or {}
    sells = tr[tr["side"] == "sell"]
    return dict(
        cum=pm.cumulative_return(eq) * 100, ann=ann * 100, vol=vol * 100,
        sharpe=(ann - pm.risk_free_rate) / vol,
        dd=pm.max_drawdown(eq)["drawdown"] * 100,
        n=len(tr), nb=int((tr["side"] == "buy").sum()), ns=len(sells),
        cost=float(tr["cost"].sum()) / CAP * 100,
        gross=(ex.get("policy_mean_gross_weight") or 0) * 100,
        names=ex.get("policy_mean_names") or 0,
        acts=acts, wr=pm.win_rate_by_trade(tr) * 100, pf=pm.profit_factor(tr),
        med_sell=float(sells["realized_pnl"].median()) if len(sells) else float("nan"),
        max_amt=float(tr["amount"].max()),
        blocked=int(ex.get("n_blocked_buy") or 0),
        blocked_by=dict(ex.get("blocked_buy") or {}),
        yr=rets.groupby(rets.index.year).apply(lambda x: float((1 + x).prod() - 1)))


def line_row(tag, m, ref=None):
    a = m["acts"]
    d = "" if ref is None else (f"  Δ{m['cum'] - ref:+7.2f}pp")
    return (f"{tag:<26} 累计 {m['cum']:7.2f}%{d}  Sharpe {m['sharpe']:5.3f} "
            f"回撤 {m['dd']:5.2f}%  年化 {m['ann']:5.2f}%  仓位 {m['gross']:4.1f}% "
            f"只数 {m['names']:4.1f}  建仓 {a.get('entry', 0):3d} 补 {a.get('add', 0):2d} "
            f"减 {a.get('trim', 0):2d} 一档即清 {a.get('exit_by_trim', 0):2d} 清 {a.get('exit', 0):3d} "
            f"降杠杆 {a.get('de_gross', 0):2d}  胜率 {m['wr']:5.2f}%  成本 {m['cost']:4.2f}%")


LAYERS = (
    ("现行 A+B（IC 门控 + 目标波动）", pol0, ovl0),
    ("静态上限 45% / 名额 7",
     dc.replace(pol0, max_total_pct=0.45, max_names=7), None),
    ("静态上限 30% / 名额 4",
     dc.replace(pol0, max_total_pct=0.30, max_names=4, add_reserve_weight=0.02), None),
    ("纯带位（无暴露层）", pol0, None),
)

R = {}


def pol_for(sv, bv, tv, lv, base):
    return dc.replace(base, sell_score=sv, buy_score=bv, strong_score=tv,
                      add_score_line=lv)


for li, (ltag, pol, ovl) in enumerate(LAYERS, 1):
    print()
    print("=" * 104)
    print(f"### 2.{li} 暴露层【{ltag}】：5 种线档 × 3 种补仓线 = 15 格"
          f"（Δ 的基准 = 本层 L0 + 补仓线关）")
    ref = None
    for ltag2, sv, bv, tv in LINES:
        for atag, lv in ADDS:
            m = brief(pol_for(sv, bv, tv, lv, pol), ovl)
            R[(ltag, ltag2, atag)] = m
            if ltag2 == LINES[0][0] and atag == "关":
                ref = m["cum"]
            print(line_row(f"{ltag2[2:]:<22} 补仓线{atag:<6}", m,
                           None if ref is None else ref))
            if ltag == LAYERS[0][0]:
                print(f"{'':<27}分年 " + " ".join(
                    f"{y} {v * 100:+5.1f}%" for y, v in m["yr"].items())
                      + f"  最大单笔 {m['max_amt']:,.0f} 元  买入被挡 {m['blocked']} 笔"
                      + f" {m['blocked_by'] or '无'}"
                      + f"  每笔卖出中位 {m['med_sell']:+,.0f} 元")

# ==================== 3. 跨层一览（决定用的那张表） ====================
print()
print("=" * 104)
print("### 3. 跨层一览：每格相对**本层 L0+补仓线关**的边际（累计 pp / 回撤 pp）")
hdr = f"{'线档':<26}" + "".join(f"{ltag[:12]:<18}" for ltag, _, _ in LAYERS)
print(hdr)
for ltag2, sv, bv, tv in LINES:
    for atag, lv in ADDS:
        cells = []
        for ltag, _, _ in LAYERS:
            a = R[(ltag, LINES[0][0], "关")]
            b = R[(ltag, ltag2, atag)]
            cells.append(f"{b['cum'] - a['cum']:+7.2f}/{b['dd'] - a['dd']:+5.2f}")
        print(f"{ltag2[2:] + ' 补' + atag:<26}" + "".join(f"{c:<18}" for c in cells))
print(f"  绝对值（累计%）：")
for ltag2, _, _, _ in LINES:
    for atag, _ in ADDS:
        print(f"    {ltag2[2:] + ' 补' + atag:<26}" + "".join(
            f"{R[(ltag, ltag2, atag)]['cum']:8.2f}%" for ltag, _, _ in LAYERS))

print()
print("### 3b. Sharpe 一览（同一 15×4 网格）")
print(hdr)
for ltag2, _, _, _ in LINES:
    for atag, _ in ADDS:
        print(f"{ltag2[2:] + ' 补' + atag:<26}" + "".join(
            f"{R[(ltag, ltag2, atag)]['sharpe']:8.3f} " for ltag, _, _ in LAYERS))

# ==================== 4. 拆开读：抬线的钱从哪来 ====================
print()
print("=" * 104)
print("### 4. 单独抬每一条线，各自值多少（现行层 + 最紧层，补仓线一律关）")
for ltag2, _, _, _ in LINES:
    a = R[(LAYERS[0][0], LINES[0][0], "关")]
    b = R[(LAYERS[0][0], ltag2, "关")]
    c = R[(LAYERS[2][0], LINES[0][0], "关")]
    d = R[(LAYERS[2][0], ltag2, "关")]
    print(f"  {ltag2:<26} 现行层 累计 {b['cum'] - a['cum']:+7.2f}pp、回撤 "
          f"{b['dd'] - a['dd']:+5.2f}pp、只数 {a['names']:.1f}→{b['names']:.1f}、"
          f"建仓 {a['acts'].get('entry', 0)}→{b['acts'].get('entry', 0)}、"
          f"清仓 {a['acts'].get('exit', 0)}→{b['acts'].get('exit', 0)}"
          f" || 30%/4 层 累计 {d['cum'] - c['cum']:+7.2f}pp、回撤 "
          f"{d['dd'] - c['dd']:+5.2f}pp")
print("  （L4 的边际 ≠ L1+L2+L3 之和：名额、总仓位、Δ 都是耦合约束，交互项在表里读。）")

print()
print("### 4b. 抬线的符号一致性（补仓线一律关，边际 = 本层该线档 − 本层 L0；噪声带 ±6.7pp）")
for ltag2, _, _, _ in LINES[1:]:
    ds, dd = [], []
    for ltag, _, _ in LAYERS:
        a = R[(ltag, LINES[0][0], "关")]
        b = R[(ltag, ltag2, "关")]
        ds.append(b["cum"] - a["cum"])
        dd.append(b["dd"] - a["dd"])
    print(f"  {ltag2:<26} " + "  ".join(f"{d:+7.2f}" for d in ds)
          + f"   | 正 {sum(1 for d in ds if d > 0)}/4 层"
          f"、超噪声 {sum(1 for d in ds if abs(d) > 6.7)}/4（负 {sum(1 for d in ds if d < -6.7)}）"
          f"、回撤变差 {sum(1 for x in dd if x > 0.81)}/4")

print()
print("### 5. 补仓线在不同线档上的边际（同层同线档，开线 − 关线）")
for atag in ("开 P97", "开 P99"):
    print(f"  ── 补仓线 {atag}")
    dsum, ddsum = [], []
    for ltag2, _, _, _ in LINES:
        cs = []
        for ltag, _, _ in LAYERS:
            a = R[(ltag, ltag2, "关")]
            b = R[(ltag, ltag2, atag)]
            dsum.append(b["cum"] - a["cum"])
            ddsum.append(b["dd"] - a["dd"])
            cs.append(f"{ltag[:8]}:{b['cum'] - a['cum']:+7.2f}pp"
                      f"(补 {a['acts'].get('add', 0)}→{b['acts'].get('add', 0)} 笔,"
                      f"回撤 {b['dd'] - a['dd']:+.2f})")
        print(f"    {ltag2:<26} " + "  ".join(cs))
    print(f"    → 4 层 × 5 线档 = 20 格里 {sum(1 for d in dsum if d > 0)} 格为正"
          f"；超噪声带 {sum(1 for d in dsum if abs(d) > 6.7)} 格（正 "
          f"{sum(1 for d in dsum if d > 6.7)} / 负 {sum(1 for d in dsum if d < -6.7)}）；"
          f"回撤变差 {sum(1 for d in ddsum if d > 0.81)}/20 格、"
          f"回撤改善 {sum(1 for d in ddsum if d < -0.81)}/20 格")

print()
print("### 6. 噪声带与读数规则")
print("  噪声带（同一份分数只换执行细节的路径差，实测 `logs/h5_arms_p94.log`、"
      "docs/entry-位置管理.md §8-C）：累计 ±6.7pp、回撤 ±0.81pp。")
print("  ⚠ 这条带是在**现行三线**下量出来的；抬线之后同一档的名字集合就变了，")
print("     带内差异仍然读不出方向，但带外的差也不保证是"
      "线档本身的功劳 —— 至少两档口径同向才算。")
print()
print(f"  跑完：{RUNS['n']} 次回测、总耗时 {time.time() - T0:.1f} 秒；"
      "只读缓存，未改任何生产代码。")
