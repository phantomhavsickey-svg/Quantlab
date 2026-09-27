# -*- coding: utf-8 -*-
"""静态上限 30% / 名额 4 那一档：P97 补仓线为什么把 2024 从 +42.3% 打成 +1.3%。

`research/add_line.py` §4 留下的坑：同一处改动（补仓加第二条标准 = P97 绝对线，与台阶 Δ
取**或**）在现行暴露层上是 **+38.83pp**，在「静态上限 30% / 名额 4」档上是 **−65.97pp**，
而且那一档的塌陷集中在 2024（分年 +42.3% → +1.3%）。这一档 13 笔补仓到底是哪几笔、
以什么方式把钱亏掉的？不知道这个，就没办法判断"要不要开这条线"。

四笔账 + 一个镜头，每笔都能逐条对上：
  §1 逐笔事后账：加仓那笔的成交价与股数 → 这个名字下一次被清空之前的加权卖价 →
     这一档钱赚/亏多少元、拿了几个交易日。口径：加仓的钱与底仓的钱在账上是混在一起的，
     所以这是**增量那一笔的成色**，不是整只票的盈亏；数字是**费前**，买入费用单列。
  §2 留一法反事实：一次只让一笔补仓"当时没做掉"（把这条意图从 `plan.buys` 与 `intents`
     里摘掉 → 状态机不推进 → 下月条件若还成立就自动补做）。这跟引擎对"开盘涨停买不进"
     的处理**完全同一条路径**，不是另造一套规则。13 笔 = 13 次回测。
  §3 名额挤出：只有 4 个名额、30% 上限，补仓用的钱是从"本来能建仓的那只"手里抢的。
     对 A/B 两档的**建仓事件**（建仓 = 买之前持仓为 0）求差集，看被挤掉的那几只在 A 档
     各自赚了多少。
  §4/§5 用现行暴露层（名额 12 只、上限 95%）做同算法对照 —— 坑不紧的时候这套账长什么样。
  §6/§6b/§7 镜头推到最疼的那一天：2024-04-01 两边各买了什么、收盘后各自被哪个约束
     顶住（名额还是钱），以及「整只票已实现 +X 元」按卖出日归年之后落在哪一年。

跑法：

    python research/add_line_2024.py   # 17 次回测、实测 42~67 秒（本机 5 次 42.2/42.2/43.2/
                                      44.2/67.3，除末行秒数外逐字节一致）；只读缓存、不改生产代码

口径与 `add_line.py` 完全一致：同一份 `data/cache/predictions.parquet`（horizon=5）、
100 万本金、月度评估、同一套费率、费后。⚠ §2 的 13 个边际**加不回** −65.97pp：
每一笔都会改变后面的路径，彼此有交互，只能单独读。
"""
import contextlib
import dataclasses as dc
import io
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
T0 = time.time()
RUNS = {"n": 0}

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

LINE_P97 = 0.01807
_EPS = 1e-9
EXEC_DAY = "2024-04-01"   # §6/§6b/§6c 的镜头对准这一轮（B 档 2024 年最贵的一次让位）
# 出问题那一档：静态上限 30%、只有 4 个名额，补仓预留 2%（add_line.py §4 第 5 层）
POL = dc.replace(pol0, max_total_pct=0.30, max_names=4, add_reserve_weight=0.02)
POLB = dc.replace(POL, add_score_line=LINE_P97)
POL_CUR = dc.replace(pol0, add_score_line=LINE_P97)   # 现行层，用来做对照


def run(pol, ovl, veto=None, watch=True, note_day=None):
    """跑一次回测。veto = {(代码, 成交日)}：这些补仓意图当场摘掉，别的什么都不动。

    note_day = "YYYY-MM-DD"：顺手把那一天 `decide()` 写在 `notes` 里的
    「这只为什么没动」原话捞出来 —— 名额被占掉的理由是引擎自己说的，不是我猜的。
    """
    calls, blocked_n = [], 0
    notes_log = {}
    real = eng.policy_plan

    def wrapper(scores, held, prices, sts, tv, pol_arg, **kw):
        nonlocal blocked_n
        snap = dict(sts)
        pp = real(scores, held, prices, sts, tv, pol_arg, **kw)
        day = pd.Timestamp(kw.get("asof")).date().isoformat()
        if note_day and day == note_day:
            for s, why in pp.notes.items():
                notes_log[str(s)] = why
        for s, it in list(pp.intents.items()):
            if it.kind != "add":
                continue
            st = snap.get(s)
            rec = dict(sym=str(s), date=day, score=it.score,
                       ladder=(st is not None and it.score >= st.ref_score + st.step - _EPS),
                       ref=(st.ref_score if st else float("nan")),
                       step=(st.step if st else float("nan")),
                       adds=(st.adds + 1 if st else 1),
                       frm=it.weight - it.delta_value / tv, to=it.weight,
                       value=it.delta_value)
            if watch:
                calls.append(rec)
            if veto and (rec["sym"], day) in veto:
                pp.intents.pop(s, None)
                pp.plan.buys.pop(s, None)
                pp.actions["add"] = max(0, pp.actions.get("add", 0) - 1)
                blocked_n += 1
        return pp

    eng.policy_plan = wrapper
    RUNS["n"] += 1
    try:
        e = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                           max_positions=cb["max_positions"],
                           cost_model=TransactionCostModel(
                               cm["commission_rate"], cm["min_commission"],
                               cm["stamp_tax_rate"], cm["slippage_rate"],
                               cm.get("stamp_tax_schedule")),
                           lot_size=int(cm.get("lot_size", 100)),
                           policy=pol, overlay=ovl)
        with contextlib.redirect_stdout(io.StringIO()):
            r = e.run(bars, SC, benchmark_prices=bm)
    finally:
        eng.policy_plan = real
    eq, tr, ex = r["equity_curve"], r["trades"], r["execution"]
    rets = eq.pct_change().dropna()
    ann, vol = pm.annual_return(eq), pm.annual_volatility(rets)
    acts = ex.get("policy_actions") or {}
    return dict(
        cum=pm.cumulative_return(eq) * 100,
        sharpe=(ann - pm.risk_free_rate) / vol,
        dd=pm.max_drawdown(eq)["drawdown"] * 100,
        gross=(ex.get("policy_mean_gross_weight") or 0) * 100,
        names=ex.get("policy_mean_names") or 0,
        n=len(tr), acts=acts, tr=tr, eq=eq, notes=notes_log,
        yr={int(y): float((1 + x).prod() - 1) * 100
            for y, x in rets.groupby(rets.index.year)},
        adds=calls, blocked=blocked_n)


def entries(tr):
    """建仓事件 = 买之前持仓为 0 的那笔买入 → {(代码, 日期iso): 金额}。"""
    out = {}
    for s, g in tr.groupby("symbol"):
        run_sh = 0
        for d, side, q, amt in zip(g["date"], g["side"], g["shares"], g["amount"]):
            if side == "buy":
                if run_sh == 0:
                    out[(str(s), pd.Timestamp(d).date().isoformat())] = float(amt)
                run_sh += int(q)
            else:
                run_sh -= int(q)
    return out


def ledger(tr, adds, last_px):
    """每笔加仓 → 这笔钱的成色（费前）：买价 vs 该名字下次清仓前的加权卖价。"""
    by = {str(s): g.sort_values("date") for s, g in tr.groupby("symbol")}
    rows = []
    for c in adds:
        g = by.get(c["sym"])
        if g is None:
            rows.append(dict(c, px=float("nan"), exit_px=float("nan"),
                             pnl=float("nan"), days=0, fee=0.0, closed=None))
            continue
        run_sh, day0, buy_px, sh, fee = 0, None, None, 0, 0.0
        sells = []
        for d, side, price, q, amt, cost in zip(g["date"], g["side"], g["price"],
                                                g["shares"], g["amount"], g["cost"]):
            dd = pd.Timestamp(d).date().isoformat()
            if side == "buy":
                if dd == c["date"] and buy_px is None and run_sh > 0:
                    # 只有"手里已经有货"的那笔买入才是加仓；同日建仓的那笔不算
                    buy_px, sh, day0, fee = float(price), int(q), pd.Timestamp(d), float(cost)
                run_sh += int(q)
            else:
                if buy_px is not None and run_sh > 0:
                    sells.append((min(int(q), run_sh), float(price), pd.Timestamp(d)))
                run_sh -= int(q)
        if buy_px is None:
            rows.append(dict(c, px=float("nan"), exit_px=float("nan"),
                             pnl=float("nan"), days=0, fee=0.0, closed="买没成交"))
            continue
        if run_sh > 0 or not sells:               # 回测结束还留着 → 按最后价盯市
            lp = float(last_px.get(c["sym"], buy_px))
            exit_px, exit_d, closed = lp, None, "未清仓"
        else:
            qsum = sum(q for q, _, _ in sells)
            exit_px = sum(q * p for q, p, _ in sells) / qsum if qsum else buy_px
            exit_d = sells[-1][2]
            closed = "已清仓"
        rows.append(dict(c, px=buy_px, sh=sh, fee=fee, exit_px=exit_px, closed=closed,
                         days=int((exit_d - day0).days) if exit_d is not None else None,
                         pnl=sh * (exit_px - buy_px)))
    return rows


print("=" * 96)
print("### 0. 两档先对齐（静态上限 30% / 名额 4，add_reserve_weight=2%）")
A = run(POL, None, watch=False, note_day=EXEC_DAY)
B = run(POLB, None, note_day=EXEC_DAY)
for tag, m in (("A 不开线（现行补仓条件）", A), ("B 开 P97 线", B)):
    a = m["acts"]
    print(f"  {tag:<22} 累计 {m['cum']:7.2f}%  Sharpe {m['sharpe']:5.3f} 回撤 {m['dd']:5.2f}%"
          f"  仓位 {m['gross']:4.1f}% 只数 {m['names']:3.1f}  补仓 {a.get('add', 0):2d}"
          f"  建仓 {a.get('entry', 0):3d}  减仓 {a.get('trim', 0)}  清仓 {a.get('exit', 0)}"
          f"  分年 " + " ".join(f"{y} {v:+6.1f}%" for y, v in sorted(m["yr"].items())))
print(f"  差额 {B['cum'] - A['cum']:+.2f}pp —— 2024 一年就占了 "
      f"{B['yr'].get(2024, 0) - A['yr'].get(2024, 0):+.1f}pp"
      f"（{A['yr'].get(2024, 0):+.1f}% → {B['yr'].get(2024, 0):+.1f}%）")
closes = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
closes = closes[~closes.index.duplicated(keep="last")].sort_index()
LAST = closes.iloc[-1].to_dict()

print()
print("=" * 96)
print("### 1. B 档这 13 笔补仓的事后账（增量那一笔的成色，费前）")
L = ledger(B["tr"], B["adds"], LAST)
print(f"  {'成交日':<11}{'代码':<8}{'分数':>8}{'买价':>8}{'清仓均价':>9}"
      f"{'这笔盈亏':>11}{'率':>8}{'天数':>6}  {'腿':<10}{'权重':>14}")
tot = 0.0
for r in sorted(L, key=lambda x: (x["date"], x["sym"])):
    tot += r["pnl"] if r["pnl"] == r["pnl"] else 0.0
    rate = (r["exit_px"] / r["px"] - 1) * 100 if r["px"] == r["px"] and r["px"] else float("nan")
    leg = "Δ+线" if r["ladder"] else "只有线"
    days = f"{r['days']:>6d}" if isinstance(r["days"], int) else "     -"
    print(f"  {r['date']:<11}{r['sym']:<8}{r['score']:8.5f}{r['px']:8.2f}"
          f"{r['exit_px']:9.2f}{r['pnl']:+11,.0f}{rate:+7.1f}%{days}  {leg:<10}"
          f"{r['frm'] * 100:5.2f}%→{r['to'] * 100:5.2f}%")
neg = [r for r in L if r["pnl"] == r["pnl"] and r["pnl"] < 0]
pos = [r for r in L if r["pnl"] == r["pnl"] and r["pnl"] > 0]
print(f"  合计 {tot:+,.0f} 元（13 笔里 {len(pos)} 笔赚、{len(neg)} 笔亏；"
      f"亏的合计 {sum(r['pnl'] for r in neg):+,.0f} 元、赚的合计 "
      f"{sum(r['pnl'] for r in pos):+,.0f} 元）")
y24 = [r for r in L if r["date"].startswith("2024")]
print(f"  2024 年那 {len(y24)} 笔：合计 "
      f"{sum(r['pnl'] for r in y24 if r['pnl'] == r['pnl']):+,.0f} 元（"
      + "、".join(f"{r['sym']} {r['pnl']:+,.0f}" for r in y24) + "）")
print("  ⚠ 这笔账只算增量那一档钱本身，不含它挤掉的机会（§3）和它改变的路径（§2）。")

print()
print("=" * 96)
print("### 2. 留一法：一次只把一笔补仓「当时没做掉」，看累计/2024 怎么变")
print("     （正数 = 拿掉它之后更好 = 这笔补仓是亏的；口径与引擎的「涨停买不进」同一条路径）")
print(f"  {'成交日':<11}{'代码':<8}{'分数':>8}  {'累计 Δpp':>9}{'2024 Δpp':>10}"
      f"{'Sharpe Δ':>10}{'回撤 Δpp':>10}  {'拿掉后补仓':>9}")
res = []
for r in sorted(L, key=lambda x: (x["date"], x["sym"])):
    m = run(POLB, None, veto={(r["sym"], r["date"])}, watch=False)
    res.append((r, m))
    print(f"  {r['date']:<11}{r['sym']:<8}{r['score']:8.5f}  "
          f"{m['cum'] - B['cum']:+9.2f}{m['yr'].get(2024, 0) - B['yr'].get(2024, 0):+10.2f}"
          f"{m['sharpe'] - B['sharpe']:+10.3f}{m['dd'] - B['dd']:+10.2f}"
          f"  {m['acts'].get('add', 0):>6d} 笔")
best = sorted(res, key=lambda x: -(x[1]["cum"] - B["cum"]))
print(f"  拿掉最伤的一笔（{best[0][0]['sym']} {best[0][0]['date']}）值 "
      f"{best[0][1]['cum'] - B['cum']:+.2f}pp；13 个边际相加 = "
      f"{sum(m['cum'] - B['cum'] for _, m in res):+.2f}pp"
      f"（对照：13 笔全拿掉 = A 档 = {A['cum'] - B['cum']:+.2f}pp）")
print("  ⚠ 相加 ≠ 总额：每笔都会改后面的路径，交互项很大，只能逐行单独读。")

print()
print("=" * 96)
print("### 3. 名额挤出：4 个名额、30% 上限，补仓的钱是从谁手里抢的")
ea, eb = entries(A["tr"]), entries(B["tr"])
only_a, only_b = sorted(set(ea) - set(eb)), sorted(set(eb) - set(ea))
pnl_a = A["tr"][A["tr"]["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
pnl_b = B["tr"][B["tr"]["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
print(f"  建仓事件 A {len(ea)} 笔 / B {len(eb)} 笔 | 只在 A 出现 {len(only_a)} 笔、"
      f"只在 B 出现 {len(only_b)} 笔")


def _pnl(tbl, sym, default=0.0):
    v = tbl.get(sym, default)
    return float(v) if v == v else default


print(f"  被挤掉的建仓（只在 A 档买了）：合计 "
      f"{sum(_pnl(pnl_a, s) for s, _ in only_a):+,.0f} 元（这些钱 B 档拿去补仓了）")
for s, d in only_a:
    print(f"    {d}  {s}  买 {ea[(s, d)]:>9,.0f} 元 | 该名字 A 档已实现盈亏 "
          f"{_pnl(pnl_a, s):+9,.0f} 元")
print(f"  反过来，只在 B 档出现的建仓：合计 "
      f"{sum(_pnl(pnl_b, s) for s, _ in only_b):+,.0f} 元")
for s, d in only_b:
    print(f"    {d}  {s}  买 {eb[(s, d)]:>9,.0f} 元 | 该名字 B 档已实现盈亏 "
          f"{_pnl(pnl_b, s):+9,.0f} 元")
print(f"  平均只数 A {A['names']:.2f} → B {B['names']:.2f} | 平均仓位 A {A['gross']:.1f}%"
      f" → B {B['gross']:.1f}% | 只数上限 {POL.max_names}、单票上限 {POL.max_position_weight:.0%}")

print()
print("=" * 96)
print("### 4. 对照：现行暴露层那 12 笔补仓的事后账（同样的算法）")
C = run(POL_CUR, ovl0)
LC = ledger(C["tr"], C["adds"], LAST)
totc = sum(r["pnl"] for r in LC if r["pnl"] == r["pnl"])
print(f"  现行层：累计 {C['cum']:.2f}%、补仓 {len(LC)} 笔、增量钱合计 {totc:+,.0f} 元、"
      f"其中亏的 {sum(1 for r in LC if r['pnl'] == r['pnl'] and r['pnl'] < 0)} 笔 "
      f"{sum(r['pnl'] for r in LC if r['pnl'] == r['pnl'] and r['pnl'] < 0):+,.0f} 元")
for r in sorted(LC, key=lambda x: (x["date"], x["sym"])):
    days = f"{r['days']:>6d}" if isinstance(r["days"], int) else "     -"
    print(f"    {r['date']}  {r['sym']}  分数 {r['score']:.5f}  买 {r['px']:7.2f} → "
          f"清 {r['exit_px']:7.2f}  {r['pnl']:+9,.0f} 元  持 {r['days'] if r['days'] is not None else '未清'} 天"
          f"  {r['frm'] * 100:5.2f}%→{r['to'] * 100:5.2f}%")
print(f"  同一个 30%/4 档：{len(L)} 笔、合计 "
      f"{sum(r['pnl'] for r in L if r['pnl'] == r['pnl']):+,.0f} 元"
      f" | 现行层每笔平均 {totc / max(1, len(LC)):+,.0f} 元、"
      f"30/4 档每笔平均 "
      f"{sum(r['pnl'] for r in L if r['pnl'] == r['pnl']) / max(1, len(L)):+,.0f} 元")

print()
print("=" * 96)
print("### 5. 对照：现行暴露层（12 只名额 + A/B 都开）的名额挤出账，同一套算法")
D = run(pol0, ovl0, watch=False)     # 现行层 A 档（不开线）
ea2, eb2 = entries(D["tr"]), entries(C["tr"])
oa, ob = sorted(set(ea2) - set(eb2)), sorted(set(eb2) - set(ea2))
p2a = D["tr"][D["tr"]["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
p2b = C["tr"][C["tr"]["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
print(f"  建仓事件 现行A {len(ea2)} 笔 / 现行B {len(eb2)} 笔 | 只在 A {len(oa)} 笔"
      f"（它们在 A 档合计已实现 "
      f"{sum(_pnl(p2a, s) for s, _ in oa):+,.0f} 元）、只在 B {len(ob)} 笔"
      f"（合计 {sum(_pnl(p2b, s) for s, _ in ob):+,.0f} 元）")
print(f"  对照 30%/4 档：只在 A {len(only_a)} 笔 = "
      f"{sum(_pnl(pnl_a, s) for s, _ in only_a):+,.0f} 元、"
      f"只在 B {len(only_b)} 笔 = {sum(_pnl(pnl_b, s) for s, _ in only_b):+,.0f} 元")
print(f"  → 现行层名额 {pol0.max_names} 只 / 上限 {pol0.max_total_pct:.0%}，"
      f"补仓抢掉的坑 {len(oa)} 个；30%/4 档只有 {POL.max_names} 个坑，抢掉 {len(only_a)} 个。")

print()
print("=" * 96)
print(f"### 6. 把镜头推到最疼的那一天：{EXEC_DAY} 那一轮，两个档各自拿着什么")
alld = pd.DatetimeIndex(sorted(set(SC.index.get_level_values("date"))))
d0 = pd.Timestamp(EXEC_DAY)
evd = alld[alld < d0][-1]


def held_before(tr, day):
    """该执行日**开盘前**的持仓（成交日就是执行日，所以取严格早于它的那些成交）。"""
    out = {}
    for d, s, side, q in zip(tr["date"], tr["symbol"], tr["side"], tr["shares"]):
        if pd.Timestamp(d).date().isoformat() >= day:
            break
        s = str(s)
        out[s] = out.get(s, 0) + (int(q) if side == "buy" else -int(q))
    return {s: q for s, q in out.items() if q > 0}


hb_a, hb_b = held_before(A["tr"], EXEC_DAY), held_before(B["tr"], EXEC_DAY)
cur = SC.loc[evd]["score"].sort_values(ascending=False)
buyed_a = {str(s) for s in A["tr"][(A["tr"]["date"] == pd.Timestamp(EXEC_DAY))
                                       & (A["tr"]["side"] == "buy")]["symbol"]}
buyed_b = {str(s) for s in B["tr"][(B["tr"]["date"] == pd.Timestamp(EXEC_DAY))
                                   & (B["tr"]["side"] == "buy")]["symbol"]}
print(f"  分数截面日 {pd.Timestamp(evd).date()}（{EXEC_DAY} 开盘前的最后一个交易日）"
      f"| 建仓线 {POL.buy_score:.5f}、清仓线 {POL.sell_score:.5f}、名额 {POL.max_names} 只")
print(f"  {'分数排名':<7}{'代码':<8}{'分数':>9}  {'A 档':<16}{'B 档':<16}")
for i, (sym, sc) in enumerate(cur.head(9).items(), 1):
    sym = str(sym)
    a_t = "持有→又买" if (sym in hb_a and sym in buyed_a) else (
        "新建仓" if sym in buyed_a else ("留着" if sym in hb_a else "没进"))
    b_t = "持有→补/减" if (sym in hb_b and sym in buyed_b) else (
        "新建仓" if sym in buyed_b else ("留着" if sym in hb_b else "没进"))
    print(f"  {i:<8}{sym:<8}{sc:9.5f}  {a_t:<17}{b_t:<15}"
          f"{'（≥建仓线）' if sc >= POL.buy_score else '（在建仓线下方）'}")
print(f"  A 当日实际买入：{sorted(buyed_a)} | 持仓（买前）{sorted(hb_a)}")
print(f"  B 当日实际买入：{sorted(buyed_b)} | 持仓（买前）{sorted(hb_b)}")
_s = {str(x) for x in set(map(str, A["tr"][(A["tr"]["date"] == d0) &
                                           (A["tr"]["side"] == "sell")]["symbol"]))}
_s2 = {str(x) for x in set(map(str, B["tr"][(B["tr"]["date"] == d0) &
                                            (B["tr"]["side"] == "sell")]["symbol"]))}
print(f"  买前那两只老仓怎么办了（清仓线 {POL.sell_score:.5f}）：")
for s in sorted(set(hb_a) | set(hb_b)):
    print(f"    {s}  分数 {cur.get(s, float('nan')):7.5f}  "
          f"A 档 {'清掉' if s in _s else '留着'}（买 {s in buyed_a}）  "
          f"B 档 {'清掉' if s in _s2 else '留着'}（买 {s in buyed_b}）  "
          f"{'← 低于清仓线，两边都该清' if cur.get(s, 9) < POL.sell_score else ''}")
ad = [r for r in L if r["date"] == EXEC_DAY]
print(f"  B 在这一天做的事：把 {ad[0]['sym']} 从 {ad[0]['frm'] * 100:.2f}% 补到 "
      f"{ad[0]['to'] * 100:.2f}%（第 {ad[0]['adds']} 档，{ad[0]['value']:,.0f} 元，"
      f"分数 {ad[0]['score']:.5f} ≥ {LINE_P97:.5f}）；"
      f"同期 A 档没这笔补仓，把钱建成了 {sorted(buyed_a - set(hb_a))}（其中 300641 买 "
      f"{ea[('300641', EXEC_DAY)]:,.0f} 元，之后 A 档该名字已实现 "
      f"{_pnl(pnl_a, '300641'):+,.0f} 元 —— 这笔钱是哪一年落袋的见 §7）。")

print()
print("### 6b. 那一天**收盘后**两边各拿着什么：补仓那笔钱把哪个约束顶到了头")
print("     （口径 = 成交后股数 × 当日收盘价 ÷ 当日权益，是**收盘盯市**权重，"
      "和策略开盘下单时的目标权重不完全相同，只用来看约束）")
DAY = EXEC_DAY


def post_round(m, day=d0):
    hp = {}
    for d, s, side, q in zip(m["tr"]["date"], m["tr"]["symbol"],
                             m["tr"]["side"], m["tr"]["shares"]):
        if pd.Timestamp(d) > day:
            break
        s = str(s)
        hp[s] = hp.get(s, 0) + (int(q) if side == "buy" else -int(q))
    return {s: v for s, v in hp.items() if v > 0}


for tag, m, bought in (("A 不开线", A, buyed_a), ("B 开 P97 线", B, buyed_b)):
    hp = post_round(m)
    eq = float(m["eq"][m["eq"].index <= d0].iloc[-1]) * CAP   # 净值 × 本金 = 当日权益
    px = closes.loc[d0]
    rows = sorted(((s, q * float(px[s]) / eq) for s, q in hp.items()),
                  key=lambda x: -x[1])
    tot = sum(w for _, w in rows)
    added_today = {r["sym"] for r in L if r["date"] == DAY} if tag.startswith("B") else set()
    print(f"  {tag}：权益 {eq:,.0f} 元 | 持仓 {len(rows)} 只（名额 {POL.max_names} 个，"
          f"还剩 {POL.max_names - len(rows)}）| 仓位 {tot * 100:.2f}%"
          f"（上限 {POL.max_total_pct:.0%}，还剩 {(POL.max_total_pct - tot) * 100:.2f}pp）"
          f"| 建仓基准一档 {POL.base_weight:.0%}、单票上限 {POL.max_position_weight:.0%}")
    for s, w in rows:
        note = "← 今天补仓补上来的" if s in added_today else (
            "← 今天新建仓" if s in bought else "← 老仓没动")
        print(f"    {s}  分数 {cur.get(s, float('nan')):7.5f}  权重 {w * 100:5.2f}%  {note}"
              f"  市值 {w * eq:,.0f} 元")
    left = (POL.max_total_pct - tot) * 100
    print(f"    → 还塞得下「一整只 5% 建仓」吗："
          f"{'能' if left >= POL.base_weight * 100 else '不能'}"
          f"（还剩 {left:.2f}pp vs 一档 {POL.base_weight * 100:.1f}pp）、"
          f"名额还剩 {POL.max_names - len(rows)} 个")

print()
print(f"### 6c. 引擎自己对那一天写下的「这只为什么没买」（`decide()` 的 notes 原话，"
      f"不是我推的）")
_note_caps = {}
for tag, m in (("A 不开线", A), ("B 开 P97 线", B)):
    hot = {s: w for s, w in m["notes"].items()
           if s in set(cur.head(12).index.astype(str)) or s in {"002993"}}
    print(f"  {tag}（{EXEC_DAY}，分数前 12 名里被拦下的 {len(hot)} 只）：")
    for s in sorted(hot, key=lambda x: -cur.get(x, 0)):
        print(f"    {s}  分数 {cur.get(s, float('nan')):7.5f}  建仓该给 "
              f"{POL.entry_weight(float(cur[s])):.1%} → {hot[s]}")
    _mm = [re.search(r"可用额度 ([\d.]+)%", v) for v in hot.values()]
    _mm = [x for x in _mm if x]
    _note_caps[tag] = float(_mm[0].group(1)) if _mm else float("nan")
_rk = {str(s): i for i, s in enumerate(cur.index, 1)}
_a0 = ad[0]
print(f"  → 引擎给「新钱」留的额度：A {_note_caps['A 不开线']:.1f}% vs B "
      f"{_note_caps['B 开 P97 线']:.1f}%，差 "
      f"{_note_caps['A 不开线'] - _note_caps['B 开 P97 线']:.1f}pp —— "
      f"就是这一档补仓（{_a0['value']:,.0f} 元 = "
      f"{(_a0['to'] - _a0['frm']) * 100:.2f}pp）加上它之前那笔补仓把 688141 "
      f"养到 {_a0['frm'] * 100:.2f}% 的合计占用。")
print(f"  → 于是 B 只能买「分数第 {_rk.get('002993', 0)} 名、建仓 "
      f"{POL.entry_weight(float(cur.get('002993', 0))):.1%}」的 002993，"
      f"而 A 用同样的钱买了分数第 2、第 3 名的 605198 与 300641（各 8%）。")

print()
print("=" * 96)
print("### 7. §3 那句「A 档该名字已实现 +X 元」拆开：这只名字在 A 档被建过几次仓、"
      "钱是哪一年的")
print("     （已实现盈亏按**卖出日**归年；同一个名字可能在后面几轮又被建回来，"
      "所以整只票的数 ≠ 那一次错过的数）")
sell_a = A["tr"][A["tr"]["side"] == "sell"].copy()
sell_a["y"] = pd.to_datetime(sell_a["date"]).dt.year
entry_days_a = {}
for (s, d) in ea:
    entry_days_a.setdefault(s, []).append(d)
print(f"  {'代码':<8}{'A 档建仓日（年后两位-月-日）':<34}{'次数':>4}  "
      f"{'按卖出年拆开的已实现盈亏':<28}{'合计':>12}{'B 档同一名字':>14}")
tot24 = 0.0
for s, d in only_a:
    days = sorted(entry_days_a.get(s, []))
    byyear = sell_a[sell_a["symbol"].astype(str) == s].groupby("y")["realized_pnl"].sum()
    parts = " ".join(f"{y}:{v:+,.0f}" for y, v in byyear.items())
    allv = float(byyear.sum())
    tot24 += float(byyear.get(2024, 0.0))
    print(f"  {s:<8}{(','.join(x[2:] for x in days)):<34}{len(days):>4d}  "
          f"{parts:<28}{allv:>+12,.0f}{_pnl(pnl_b, s):>+14,.0f}")
print(f"  → 这 15 只名字的已实现盈亏里，落在 **2024 年**的部分合计 {tot24:+,.0f} 元；"
      f"整个样本期合计 {sum(_pnl(pnl_a, s) for s, _ in only_a):+,.0f} 元。")
print(f"  → B 档里这些名字已实现合计 {sum(_pnl(pnl_b, s) for s, _ in only_a):+,.0f} 元"
      f"（B 只在个别轮买了别的名字，见 §3 下半张表）。")
_k = [s for s, d in only_a if d == DAY]
_by = sell_a[sell_a["symbol"].astype(str) == "300641"].groupby("y")["realized_pnl"].sum()
print(f"  → {EXEC_DAY} 那天被挤掉的 {len(_k)} 只 {sorted(_k)}：在 A 档各建过 "
      f"{[len(entry_days_a.get(s, [])) for s in _k]} 次仓；300641 的 "
      f"{_by.sum():+,.0f} 元全部记在 "
      f"{sorted(int(y) for y in _by.index)} 年卖出，A 档只买过它 "
      f"{len(entry_days_a.get('300641', []))} 次、B 档一次都没买 "
      f"（B 档该名字已实现 {_pnl(pnl_b, '300641'):+,.0f} 元）。")

print()
print(f"（本脚本共 {RUNS['n']} 次回测、实测 {time.time() - T0:.1f} 秒；"
      f"§0/§4/§5 各 1 档 + §2 留一 13 档）")
