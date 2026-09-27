# -*- coding: utf-8 -*-
"""补仓的两条标准：「涨够一个步长 Δ」与「分数够到 P97 这条绝对线」——**或**关系。

回答的问题：补仓原来只有一个条件（比上次动作的参考分数再涨够 Δ）。给它加第二条
**独立**条件：只要当前分数够到全市场前 3%（P97 = 0.01807）就补一档，不需要涨够 Δ。
任一满足即补（`ladder_up or level_up`），不是"两个都要满足"。减仓 / 清仓条件一字不动
（`trim_step_weight` `trim_price_ttl_days` `sell_score` 全部沿用 config 现值），
而且水平线**只负责触发、不推进参考分数**，所以连"减仓的基准"都没被抬高。
→ 2026-09-27 这条线**已被采纳为 config 默认**（`add_score_line: 0.01807`），
  采纳条件见 config.yaml 注释：只在现行暴露层（名额 12 只 + 总仓位 95%）下成立。

对照臂（A 一律显式写 `add_score_line=0`，**不跟 config 走**）：
  A 线关（= 2026-09-27 采纳**之前**的档）
  B 水平线 = P97 = 0.01807 ← **这就是 config 现行默认**，本脚本 A→B 的边际就是采纳的那一步
  C 水平线 = P99 = 0.02888（更严的水平条件）
  D 连 Δ 一起关（`step_score_floor` = 999）—— 补、减共用的 Δ 被一起杀掉，只是对照
  E 只禁补仓（`add_step_weight` ≈ 0 → 补一档的名义额不足最小交易额）—— **减仓条件照旧**，
    这才是"补仓这件事整体值多少钱"的干净下界

口径与 `research/sell_line.py` / `research/h5_arms.py` 一致：同一份
`data/cache/predictions.parquet`（horizon=5）、100 万本金、月度评估、同一套费率、费后。

⚠ 两处容易读错的地方，脚本里都当场打出来：
  1. `strong_score` 本来就 = P97 = 0.01807，但那条管的是**建仓**给多厚（到 P97 给满
     max_entry_weight），本档动的是**已持仓要不要再加一档**，两条不同规则数值相同纯属同一把尺子。
  2. 水平线是**每轮都会重新判断**的条件：分数连续几个月都待在 P97 上方就会每月补一档，
     一路顶到 `max_position_weight` = 16%。所以"设一条水平线"不等于"到 P97 补一次"。
     实测（§4 逐笔表）：现行暴露层 A+B 的 12 笔补仓落在 12 个不同名字上、全部是第 1 档，
     没有一只补到第 2 次（只有 1 笔因为价格本身已到 16% 上限）；【只开 A】层 38 笔里
     有 6 笔是同一只票的第 2 档、5 笔顶到 16% 上限（其中 4 笔来自第 2 档）。

跑法：

    python research/add_line.py      # 实测 61~63 秒（20 次回测）；只读缓存，不改生产代码
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

import numpy as np  # noqa: F401  (与既有脚本保持同一份导入)
import pandas as pd
import yaml

import backtest.engine as eng
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

# 对照臂的水平线数值写死在这里，不读 config —— 万一哪天 config 采纳了这条线，
# 读回来的话「旧档」和「新档」是同一个数，边际就算不出来了。
LINE_OFF = 0.0
LINE_P97 = 0.01807
LINE_P99 = 0.02888
LINE_NEVER = 999.0
_EPS = 1e-9

print("=" * 96)
print("### 1. 尺子：三条线在同一天截面里的位置（绝对补仓线用的是同一把尺）")
closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
fwd = (closes.shift(-H) / closes - 1).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
df = preds_df.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
pct = lambda v: (df.prediction < v).mean() * 100  # noqa: E731
print(f"  配对样本 {len(df):,}（horizon={H} 日预测收益，能算出真实 5 日收益的那批）")
for q in (.84, .94, .97, .99):
    print(f"    P{q * 100:<4.0f}= {df.prediction.quantile(q):.5f}")
print(f"  清仓线 {pol0.sell_score:.5f} = P{pct(pol0.sell_score):.2f}   "
      f"建仓线 {pol0.buy_score:.5f} = P{pct(pol0.buy_score):.2f}   "
      f"strong_score(建仓满仓线) {pol0.strong_score:.5f} = P{pct(pol0.strong_score):.2f}")
print(f"  本档候选绝对补仓线 = {LINE_P97:.5f} = P{pct(LINE_P97):.2f}、"
      f"{LINE_P99:.5f} = P{pct(LINE_P99):.2f}")
print(f"  config 现行 add_score_line = {pol0.add_score_line:.5f}"
      f"（{'关闭：补仓只看步长 Δ' if pol0.add_score_line <= 0 else '已开'}）")
print()

# ---------- 1b. 两条条件各自的候选池有多大（或关系 = 并集） ----------
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
    rows.append(pd.DataFrame({"at_entry": hit, "next": d, "delta": d - hit}))
w = pd.concat(rows)
dlt = np.maximum(pol0.step_score_floor,
                 pol0.step_multiplier * (w.at_entry - pol0.buy_score))
need = w.at_entry + dlt
ladder = w["next"] >= need
print(f"### 1b. 两条条件各放行多少（{len(rows)} 对相邻月末、{len(w):,} 个「过建仓线名字-轮」观测）")
print(f"  过建仓线分数中位 {w.at_entry.median():.5f} | 下一轮分数变化中位 "
      f"{w.delta.median():+.5f}（= {w.delta.median() * 100:+.2f}pp）")
print(f"    条件①涨够一档（分数 ≥ 建仓分 + Δ）      {(ladder).mean() * 100:5.1f}%")
for lab, lv in ((f"条件②达 P97 {LINE_P97:.5f}", LINE_P97),
                (f"条件②达 P99 {LINE_P99:.5f}", LINE_P99),
                ("条件②达 999（永不）", LINE_NEVER)):
    level = w["next"] >= lv
    print(f"    {lab:<28} {level.mean() * 100:5.1f}%"
          f" | ①或②（并集） {(ladder | level).mean() * 100:5.1f}%"
          f" | 只由②放行（①不成立） {(~ladder & level).mean() * 100:5.1f}%")
print("  （口径说明：这里按「建仓分数」起算 Δ，是候选池的**上界**；真实状态机的参考分数")
print("   每补一次就上调 n×Δ，只有台阶路径会上调，绝对线触发的补仓不动参考分数。）")
print()


def brief(pol, ovl, watch_line=None):
    """跑一次回测，返回摘要；watch_line 不为 None 时顺带做补仓触发归因。"""
    calls = []
    if watch_line is not None:
        real = eng.policy_plan

        def wrapper(scores, held, prices, sts, tv, pol_arg, **kw):
            snap = dict(sts)
            pp = real(scores, held, prices, sts, tv, pol_arg, **kw)
            for s, it in pp.intents.items():
                if it.kind != "add":
                    continue
                st = snap.get(s)
                calls.append(dict(
                    sym=s, date=kw.get("asof"),
                    ladder=(st is not None and it.score >= st.ref_score + st.step - _EPS),
                    level=watch_line > 0 and it.score >= watch_line - _EPS,
                    score=it.score, adds=(st.adds + 1 if st else 1),
                    ref=(st.ref_score if st else float("nan")),
                    step=(st.step if st else float("nan")),
                    frm=it.weight - it.delta_value / tv, to=it.weight,
                    value=it.delta_value))
            return pp
        eng.policy_plan = wrapper
    try:
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
    finally:
        if watch_line is not None:
            eng.policy_plan = real  # noqa: F821
    eq, tr, ex = r["equity_curve"], r["trades"], r["execution"]
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    acts = ex.get("policy_actions") or {}
    sells = tr[tr["side"] == "sell"]
    buys = {}   # 代码 → [(成交日, 金额, 股数)]，用于「计划补仓 vs 实际成交」逐笔核对
    tb = tr[tr["side"] == "buy"].sort_values("date")
    for s, g in tb.groupby("symbol"):
        buys[str(s)] = [(d, float(a), int(q)) for d, a, q in
                        zip(g["date"], g["amount"], g["shares"])]
    out = dict(
        cum=pm.cumulative_return(eq) * 100, ann=ann * 100, vol=vol * 100,
        sharpe=(ann - pm.risk_free_rate) / vol,
        dd=pm.max_drawdown(eq)["drawdown"] * 100,
        n=len(tr), nb=int((tr["side"] == "buy").sum()), ns=len(sells),
        cost=float(tr["cost"].sum()) / CAP * 100,
        gross=(ex.get("policy_mean_gross_weight") or 0) * 100,
        names=ex.get("policy_mean_names") or 0,
        acts=acts, wr=pm.win_rate_by_trade(tr) * 100, pf=pm.profit_factor(tr),
        med_sell=float(sells["realized_pnl"].median()),
        max_amt=float(tr["amount"].max()),
        blocked=ex.get("blocked_buy") or {},
        buys=buys,
        yr=rets.groupby(rets.index.year).apply(lambda x: float((1 + x).prod() - 1)))
    if watch_line is not None:
        out["add_calls"] = calls
    return out


def show(tag, m):
    a = m["acts"]
    print(f"{tag:<34} 累计 {m['cum']:7.2f}% 年化 {m['ann']:6.2f}% 波动 {m['vol']:5.2f}%"
          f" Sharpe {m['sharpe']:5.3f} 回撤 {m['dd']:5.2f}%")
    print(f"{'':<34} 笔数 {m['n']:3d}（买 {m['nb']} / 卖 {m['ns']}） 成本 {m['cost']:4.2f}%"
          f" 仓位 {m['gross']:4.1f}% 只数 {m['names']:3.1f} 最大单笔 {m['max_amt']:,.0f} 元")
    print(f"{'':<34} 动作 " + " ".join(
        f"{k}={a.get(k, 0)}" for k in ("entry", "add", "trim", "exit_by_trim",
                                       "exit", "de_gross")))
    print(f"{'':<34} 按笔胜率 {m['wr']:5.2f}% 利润因子 {m['pf']:4.2f} "
          f"每笔卖出中位 {m['med_sell']:+,.0f} 元 | 分年 " + " ".join(
              f"{y} {v * 100:+5.1f}%" for y, v in m["yr"].items()))


def adds_table(m, indent="    "):
    """每一次补仓触发的**具体分数**，逐笔核对「计划 → 实际成交」。"""
    cs = m.get("add_calls") or []
    if not cs:
        print(f"{indent}这一档一次补仓都没触发。")
        return
    cs = sorted(cs, key=lambda c: (str(c["date"]), str(c["sym"])))
    print(f"{indent}成交日      代码    触发分数   台阶基准+Δ  靠哪条腿      "
          f"权重(占总权益)   计划金额 → 实际成交")
    filled, miss = 0, []
    for c in cs:
        sym = str(c["sym"])
        day = pd.Timestamp(c["date"])
        need = c["ref"] + c["step"]
        leg = ("Δ+线" if c["ladder"] and c["level"] else
               ("①Δ" if c["ladder"] else "②只有水平线"))
        hit = [x for x in m["buys"].get(sym, []) if pd.Timestamp(x[0]) == day]
        if hit:
            filled += 1
            got = "、".join(f"{a:,.0f} 元/{q} 股" for _, a, q in hit)
        else:
            got = "未成交"
            miss.append((sym, day))
        print(f"{indent}{day:%Y-%m-%d}  {sym}  {c['score']:.5f}  "
              f"{c['ref']:.5f}+{c['step']:.5f}={need:.5f}  {leg:<12} "
              f"{c['frm'] * 100:5.2f}% → {c['to'] * 100:5.2f}%(第{c['adds']}档)  "
              f"{c['value']:9,.0f} → {got}")
    second = sum(1 for c in cs if c["adds"] >= 2)
    capped = sum(1 for c in cs if c["to"] >= 0.16 - 1e-6)
    print(f"{indent}计划 {len(cs)} 笔 → 当日成交 {filled} 笔"
          + (f"、未成交 {len(miss)} 笔（{miss}）" if miss else "")
          + f" | 买入被挡合计 {sum(m['blocked'].values())} 笔 {m['blocked'] or '无'}"
          + f" | 触发分数中位 {pd.Series([c['score'] for c in cs]).median():.5f}")
    print(f"{indent}其中同一只票的第 2 档及以上 {second} 笔、把权重顶到 16% 单票上限 "
          f"{capped} 笔 | 补过的名字 {len(set(c['sym'] for c in cs))} 个"
          f"（占 {len(cs)} 笔 → 平均每只 {len(cs) / max(1, len(set(c['sym'] for c in cs))):.2f} 次）")


ARMS = (("A 补仓线关（采纳前档）", dict(add_score_line=LINE_OFF), LINE_OFF),
        (f"B 或：水平线 = P97 {LINE_P97:.5f}（= 现行默认）",
         dict(add_score_line=LINE_P97), LINE_P97),
        (f"C 或：水平线 = P99 {LINE_P99:.5f}", dict(add_score_line=LINE_P99), LINE_P99),
        ("E 只禁补仓（add_step_weight≈0）", dict(add_score_line=0.0,
                                                  add_step_weight=1e-9), 0.0),
        ("D 连 Δ 一起关（step_floor=999）", dict(add_score_line=0.0,
                                                  step_score_floor=999.0), 0.0))
# E 与 D 的差别很重要:E 只让补仓触发不了(补一档的名义额不足最小交易额),
# Δ 仍然管着减仓 —— 所以 E 才是「补仓这件事值多少钱」的干净下界。
# D 把 trim 也一起杀掉了(Δ 是补/减共用的),留在表里做对照,读它等于读「补+减」的总价。


def arm_pol(kw):
    return dc.replace(pol0, **kw)


print("=" * 96)
print("### 2. 带暴露层 A+B（现行档）：一次只动 add_score_line"
      "（A = 关 = 2026-09-27 采纳前，B = 开 P97 = 现行默认）")
R2 = {}
for tag, kw, lv in ARMS:
    R2[tag] = brief(arm_pol(kw), ovl0, watch_line=lv)
    show(tag, R2[tag])

print()
print("=" * 96)
print("### 3. 摘掉暴露层复核（纯带位，其余参数不变）")
R3 = {}
for tag, kw, lv in ARMS:
    R3[tag] = brief(arm_pol(kw), None, watch_line=lv)
    show(tag, R3[tag])

# ==================== 4. 同一处改动在五种暴露层上的前后对照 ====================
# 和清仓线那节同样的理由：一句「补仓多一条线值多少」在不同暴露层上可能有正有负。
# 少了这张表，§2 那一行很容易被读成「这条线永远更好」。
# 这一节每一层都把 A/B 两档的**完整指标行**和**逐笔补仓分数**打出来，
# 因为「边际 +38.83pp」这种一句话回答不了「到底哪一只、哪一天、什么分数在补」。
print()
print("=" * 96)
print(f"### 4. 五种暴露层，每层「无水平线 → 水平线 = P97」的前后完整表现 + 逐笔分数"
      f"（config 现行 add_score_line = {pol0.add_score_line:.5f}）")
LAYERS = (
    ("现行 A+B（IC 门控 + 目标波动）", pol0, ovl0),
    ("只开 A（IC 门控，不缩波动）", pol0, dc.replace(ovl0, vol_target_ann=0.0)),
    ("只开 B（目标波动，无门控）", pol0, dc.replace(ovl0, ic_window_days=0)),
    ("静态上限 45% / 名额 7", dc.replace(pol0, max_total_pct=0.45, max_names=7), None),
    ("静态上限 30% / 名额 4",
     dc.replace(pol0, max_total_pct=0.30, max_names=4, add_reserve_weight=0.02), None))
L4 = {}
for tag, pol, ovl in LAYERS:
    a = brief(dc.replace(pol, add_score_line=LINE_OFF), ovl, watch_line=LINE_OFF)
    b = brief(dc.replace(pol, add_score_line=LINE_P97), ovl, watch_line=LINE_P97)
    L4[tag] = (a, b)
    print()
    print(f"  ── 暴露层【{tag}】" + "─" * max(2, 66 - len(tag) * 2))
    show("    A 无水平线（现行）", a)
    show("    B 水平线 = P97", b)
    print(f"    边际：累计 {b['cum'] - a['cum']:+7.2f}pp（{a['cum']:.2f}% → {b['cum']:.2f}%）"
          f" | Sharpe {b['sharpe'] - a['sharpe']:+.3f}（{a['sharpe']:.3f} → {b['sharpe']:.3f}）"
          f" | 回撤 {b['dd'] - a['dd']:+5.2f}pp（{a['dd']:.2f}% → {b['dd']:.2f}%）"
          f" | 补仓 {a['acts'].get('add', 0)} → {b['acts'].get('add', 0)}"
          f" | 最大单笔 {a['max_amt']:,.0f} → {b['max_amt']:,.0f} 元")
    adds_table(b)
print()
print("  噪声带（同一份分数只换执行细节的路径差，实测见 logs/h5_arms_p94.log 与 "
      "docs/entry-位置管理.md §8-C）：累计 ±6.7pp、回撤 ±0.81pp。")
print("  落在这两条带里的差别读不出方向，只能读「结构变了多少」（补仓笔数、单票权重）。")
print()
print("  五层边际一览（累计口径，从最松到最紧）：")
for tag, (a, b) in sorted(L4.items(), key=lambda kv: kv[1][1]["cum"] - kv[1][0]["cum"],
                          reverse=True):
    d = b["cum"] - a["cum"]
    print(f"    {tag:<28} {d:+7.2f}pp  补仓 {a['acts'].get('add', 0)} → "
          f"{b['acts'].get('add', 0)} 笔  Sharpe {a['sharpe']:.3f} → {b['sharpe']:.3f}  "
          f"回撤 {b['dd'] - a['dd']:+.2f}pp  {'在噪声带内' if abs(d) <= 6.7 else '超出噪声带'}")

print()
print("=" * 96)
print("### 5. 结构差异：这条水平线到底改了什么（E/D 是「补仓值多少钱」的两个下界）")
for lab, R in (("带暴露层 A+B", R2), ("纯带位", R3)):
    a, b, c, e, d = (R[t] for t, _, _ in ARMS)
    print(f"  {lab}：补仓 A {a['acts'].get('add', 0)} → B {b['acts'].get('add', 0)}"
          f" → C {c['acts'].get('add', 0)}（E 只禁补 {e['acts'].get('add', 0)} / "
          f"D 连 Δ 一起关 {d['acts'].get('add', 0)}） | 建仓 {a['acts'].get('entry', 0)} → "
          f"{b['acts'].get('entry', 0)} | 平均仓位 {a['gross']:.1f}% → {b['gross']:.1f}% |"
          f" 平均只数 {a['names']:.1f} → {b['names']:.1f} | 最大单笔 "
          f"{a['max_amt']:,.0f} → {b['max_amt']:,.0f} 元")
    print(f"    减仓**规则**一字未动（`trim_step_weight`/`sell_score`/参考分数都不被水平线推进）；"
          f"下面是不动的规则下**触发次数**：trim A {a['acts'].get('trim', 0)} → B "
          f"{b['acts'].get('trim', 0)} → C {c['acts'].get('trim', 0)} | exit_by_trim A "
          f"{a['acts'].get('exit_by_trim', 0)} → B {b['acts'].get('exit_by_trim', 0)} → "
          f"C {c['acts'].get('exit_by_trim', 0)}（E {e['acts'].get('trim', 0)}、"
          f"D {d['acts'].get('trim', 0)} ← D 把 Δ 关了所以减仓一起没了）。"
          f"纯带位那行 B 的次数变了是**路径效应**：先补过仓的票仓位更厚，掉下去时一次穿线的概率更大。")
    print(f"    累计 {a['cum']:.2f}% → B {b['cum'] - a['cum']:+.2f}pp → "
          f"C {c['cum'] - a['cum']:+.2f}pp | E 只禁补 {e['cum'] - a['cum']:+.2f}pp、"
          f"D {d['cum'] - a['cum']:+.2f}pp | 回撤 {a['dd']:.2f}% → B "
          f"{b['dd'] - a['dd']:+.2f}pp → C {c['dd'] - a['dd']:+.2f}pp | E "
          f"{e['dd'] - a['dd']:+.2f}pp、D {d['dd'] - a['dd']:+.2f}pp")

print()
print("=" * 96)
print("### 6. 补仓触发归因：每一次 add 是靠哪条标准触发的（带暴露层 A+B）")
for tag, lv in ((ARMS[0][0], LINE_OFF), (ARMS[1][0], LINE_P97), (ARMS[2][0], LINE_P99)):
    m = R2[tag]
    cs = m.get("add_calls") or []
    only_lad = sum(1 for c in cs if c["ladder"] and not c["level"])
    only_lvl = sum(1 for c in cs if c["level"] and not c["ladder"])
    both = sum(1 for c in cs if c["ladder"] and c["level"])
    cap_hits = sum(1 for c in cs if c["adds"] >= 3)
    print(f"  {tag:<30} 补仓触发 {len(cs):3d} 笔：只靠 Δ {only_lad:3d} | "
          f"Δ 与水平线同时成立 {both:3d} | **只靠水平线** {only_lvl:3d} | "
          f"同一只第 3 次以上补仓 {cap_hits:3d} 笔")
    if cs:
        print(f"    触发分数 min/中位/max = {min(c['score'] for c in cs):.5f} / "
              f"{pd.Series([c['score'] for c in cs]).median():.5f} / "
              f"{max(c['score'] for c in cs):.5f}"
              f" | 同一只最多补到第 {max(c['adds'] for c in cs)} 档")
        per = pd.Series([c["sym"] for c in cs]).value_counts()
        print(f"    分散度：{len(cs)} 笔落在 {len(per)} 个名字上 | 补过 1 次的名字 "
              f"{int((per == 1).sum())} 个、2 次 {int((per == 2).sum())} 个、"
              f"≥3 次 {int((per >= 3).sum())} 个 | 最集中的一只 {per.index[0]} 补 {int(per.iloc[0])} 次")
        dts = pd.Series([pd.Timestamp(c["date"]) for c in cs]).dt.strftime("%Y-%m")
        print("    触发月份分布：" + "  ".join(
            f"{k}×{int(v)}" for k, v in dts.value_counts().sort_index().items()))

print()
print("=" * 96)
print("### 7. 两条标准谁更松：一条可以手算的不等式（方向与「闸」时代相反）")
first_add = pol0.buy_score + pol0.step_score_floor
print(f"  条件①首次触发的最低分数 = 建仓线 {pol0.buy_score:.5f} + Δ 下限 "
      f"{pol0.step_score_floor:.5f} = {first_add:.5f} = P{pct(first_add):.2f}")
print(f"  条件②水平线 = {LINE_P97:.5f} = P{pct(LINE_P97):.2f} → "
      f"{'水平线在①下方，所以它是更松的那条腿：能在「没涨够一档」时补仓' if LINE_P97 < first_add else '水平线在①上方，只在涨够一档时一起成立'}"
      f"（差 {LINE_P97 - first_add:+.5f}）")
print(f"  对照：C 档 {LINE_P99:.5f} = P{pct(LINE_P99):.2f} 高于 {first_add:.5f} → 它自己"
      f"很少单独触发（要分数直接冲到 P99），但或关系下条件②**只会放行更多，不会挡任何东西**"
      f"（§6 的归因表里 C 档仍有 {R2[ARMS[2][0]]['acts'].get('add', 0)} 笔补仓，比 A 的 "
      f"{R2[ARMS[0][0]]['acts'].get('add', 0)} 笔多）。")
print("  → 所以「或」的方向与上一版「且」相反：这一版是**放宽**补仓，不是收紧。")
print("    同一只票只要分数继续待在水平线上方，每轮都可能再补一档，直到 16% 单票上限或")
print("    12 只名额 / 95% 总仓位把它卡住。各层实测（§4 逐笔表，括号=第 2 档及以上 / 顶到 16%）：")
for tag, (_, b) in L4.items():
    cs = b.get("add_calls") or []
    n2 = sum(1 for c in cs if c["adds"] >= 2)
    cap = sum(1 for c in cs if c["to"] >= 0.16 - 1e-6)
    print(f"      {tag:<28} 补仓 {len(cs):3d} 笔（第 2 档及以上 {n2} 笔、顶到 16% {cap} 笔）")
print("    → 现行层里这条线基本只补一次；把波动层放松后才会连续补到封顶。")
print()
print("  附：E 档才是「完全不许补仓」的干净下界 —— 把 add_step_weight 压到近 0，")
print("     补一档的名义额不足最小交易额 → 补仓全灭，而 Δ 与减仓条件一字未动。")
print("     D 档（step_score_floor=999）把补、减共用的 Δ 一起关掉，读它等于读「补+减」的总价，")
print("     两档的差额（§5）就是减仓这条规则自己的价钱。")
