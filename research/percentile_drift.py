# -*- coding: utf-8 -*-
"""三条分数线是「写死的绝对值」还是「每天重算的百分位」？把这两者的关系量出来。

问题背景：config 里 `sell_score=0.00875 / buy_score=0.01354 / strong_score=0.01807` 后面
标着 P84 / P94 / P97。容易误读成"系统每天按当天截面重算分位数"。实际上：

  * 线上跑的时候**只比大小**：`score >= buy_score`、`score < sell_score`，比的是绝对数值；
  * 那三个 P 是**一次性标定**留下的标签：在整份 `predictions.parquet`（854,322 个配对观测）
    上取 P84/P94/P97 得到三个数，抄进 config 就固定了；
  * 所以"百分位"这个标签会随时间**失效**：分数整体水位一漂移，同一个绝对值在当天截面里
    处在的分位就变了 —— 线不动，它对应的百分位在动。

这一节把这个漂移量出来：§2 每天、§3 逐月逐年、§4 逐折（walk-forward 每 6 个月换一次模型，
分数分布本身会变），§5 反过来问"如果按最近的窗口重标，三条线要移到哪儿"。

    python research/percentile_drift.py     # 实测 3.5~4.1 秒，纯读缓存，不跑回测

口径：日截面 = 当日**有分数**的全部名字（896 个信号日 × ~955 只，854,763 行）；
标定用的分布 = 同一份缓存里"5 日后真算得出收益"的配对样本（854,322 行），差 441 行是
最后 5 个交易日没有前向收益。§1 会把这条核对打印出来。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: F401
import pandas as pd
import yaml

from utils.position_policy import policy_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
H = int(cfg["model"]["horizon"])
pol = policy_from_config(cfg)
LINES = (("清仓线", pol.sell_score), ("建仓线", pol.buy_score),
         ("强线/建仓满仓线", pol.strong_score))

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
s = pred["prediction"]

print("=" * 96)
print("### 1. 先核对：线上到底有没有在算分位数")
hits = []
for d in ("live", "paper_trade", "utils", "backtest", "models"):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"):
            continue
        p = os.path.join(d, f)
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if ("quantile(" in line or "percentile(" in line or ".rank(pct" in line):
                hits.append((p, i, line.strip()[:70]))
# 分两类：拿分数算分位（那才是"实时重标"），和拿别的东西算分位（诊断列）
on_score = [h for h in hits if "hold_days" not in h[2]]
print(f"  扫描 live/ paper_trade/ utils/ backtest/ models/ 里所有 .py：")
print(f"  分位数/百分位调用 {len(hits)} 处 —— 其中**对分数**的 {len(on_score)} 处")
for p, i, t in hits:
    tag = "分数" if (p, i, t) in on_score else "非分数（诊断列）"
    print(f"    [{tag}] {p}:{i}: {t}")
ranks = []
for d in ("live", "paper_trade", "utils", "backtest", "models"):
    for f in sorted(os.listdir(d)):
        if not f.endswith(".py"):
            continue
        p = os.path.join(d, f)
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if ".rank(" in line:
                ranks.append(f"{p}:{i}: {line.strip()[:70]}")
print(f"  另有名次/排序调用 {len(ranks)} 处（同样不产出阈值：exposure 那两处是 RankIC 的"
      f"名次相关，predictor 那三处给信号表附 rank/weight 列，带位策略只读 score 列）")
for h in ranks:
    print("    " + h)
print(f"  → 对**分数**取分位数 = {len(on_score)} 处：三条线在 config.yaml 里是**写死的绝对分数**，"
      f"运行路径只做 `score >= 阈值` 的比较。")
print("  标定分布：配对样本 = 有 5 日前向收益的那些行；全样本 = 当日有分数的全部行。")

# 配对样本口径（与 research/calib_stats.py §1 同一把尺）
from data.cache import CacheManager  # noqa: E402

cache = CacheManager(cfg["cache"]["directory"])
bars = {}
for sym in pred["symbol"].unique():
    df = cache.get_daily(sym)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[sym] = df.sort_values("日期")
close = pd.DataFrame({k: v.set_index("日期")["收盘"] for k, v in bars.items()})
close = close[~close.index.duplicated(keep="last")].sort_index()
fwd = (close.shift(-H) / close - 1.0).stack().rename("fwd").reset_index()
fwd.columns = ["date", "symbol", "fwd"]
pair = pred.merge(fwd, on=["date", "symbol"]).dropna(subset=["fwd"])
print(f"  全样本 {len(pred):,} 行 / 配对样本 {len(pair):,} 行 / "
      f"信号日 {pred['date'].nunique()} 天 / {pred['symbol'].nunique()} 只")
for name, v in LINES:
    print(f"    {name:<16} = {v:.5f}   在配对样本上 = "
          f"P{(pair.prediction < v).mean() * 100:.2f}"
          f"   在全样本上 = P{(pred.prediction < v).mean() * 100:.2f}")
print("  → 这两行的数字对得上文档里标的 P84.01 / P94.00 / P97.00，说明尺子没换。")

# ==================== 2. 每天：同一条线在当天截面里排第几 ====================
day = pred.groupby("date")["prediction"]
n_over = day.apply(lambda x: int((x >= pol.buy_score).sum()))   # 当日过建仓线只数
eff = pd.DataFrame({nm: day.apply(lambda x, v=v: float((x < v).mean()) * 100)
                    for nm, v in LINES})
print()
print("=" * 96)
print(f"### 2. 逐日有效百分位（{len(eff)} 个信号日：当天截面里低于这条线的名字占比）")
print("     线是固定的绝对值，这一列却每天在变 —— 这就是「百分位不是实时的、但会漂」")
for nm, v in LINES:
    c = eff[nm]
    print(f"  {nm:<16} {v:.5f}  中位 P{c.median():.2f} | 最常见区间 "
          f"P{c.quantile(.1):.1f}~P{c.quantile(.9):.1f} | 最低 P{c.min():.2f}"
          f"（{c.idxmin().strftime('%Y-%m-%d')}）| 最高 P{c.max():.2f}"
          f"（{c.idxmax().strftime('%Y-%m-%d')}）| 极差 {c.max() - c.min():.1f}pp")
print(f"  同一把尺落到只数上（建仓线）：日均 {n_over.mean():.1f} 只 / 中位 "
      f"{n_over.median():.0f} 只 / 最低 {n_over.min()} 只（{n_over.idxmin():%Y-%m-%d}）"
      f"/ 最高 {n_over.max()} 只（{n_over.idxmax():%Y-%m-%d}）"
      f" —— 设计意图是「最前 6%」，也就是 ~57 只")
print("  对照：把整份缓存合起来看，这三条线分别在 P%.1f / P%.1f / P%.1f —— "
      "标签说的是**整份分布**上的位置，不是任何一天的截面。"
      % tuple((pred.prediction < v).mean() * 100 for _, v in LINES))

# ==================== 3. 逐月 / 逐年 ====================
mon = eff.groupby(eff.index.to_period("M")).mean()
mon_over = n_over.groupby(n_over.index.to_period("M")).mean()   # 月均过线只数
print()
print("=" * 96)
print("### 3. 逐月（按自然月平均：三条线的有效分位 + 过建仓线只数）")
print(f"  {'月份':<10}" + "".join(f"{nm:>18}" for nm, _ in LINES) + f"{'日均过线只数':>14}")
for p in mon.index:
    print(f"  {str(p):<10}" + "".join(f"{mon.loc[p, nm]:>12.2f} (P)" for nm, _ in LINES)
          + f"{mon_over.loc[p]:>14.1f}")
yr = eff.groupby(eff.index.year).mean()
print(f"  月均过建仓线只数：最紧 {mon_over.idxmin()} 月 {mon_over.min():.1f} 只/日，"
      f"最松 {mon_over.idxmax()} 月 {mon_over.max():.1f} 只/日（设计意图 ~57 只/日）")
print("  分年：" + "  ".join(
    f"{y} " + "/".join(f"P{yr.loc[y, nm]:.1f}" for nm, _ in LINES) for y in yr.index))
print("  读法：表里的数 = 「当月平均而言，当天截面里有多少比例的名字低于这条线」，")
print("        也就是那条**写死的线**在这个月的实际分位。同一串数字，不同月份挡掉的")
print("        名字比例不一样 —— 规则没变，规则的松紧变了。")

# ==================== 4. 逐折：重训本身会不会把分布搬走 ====================
CUT = ["2023-01-05", "2023-07-05", "2024-01-05", "2024-07-05",
       "2025-01-06", "2025-07-07", "2026-01-05", "2026-07-06"]  # logs/h5_train.log
print()
print("=" * 96)
print("### 4. 逐折（walk-forward 每 6 个月换一次模型；折边界取自训练日志）")
bounds = [pd.Timestamp(c) for c in CUT] + [pred["date"].max() + pd.Timedelta(days=1)]
for i in range(len(CUT)):
    seg = pred[(pred["date"] >= bounds[i]) & (pred["date"] < bounds[i + 1])]
    if not len(seg):
        continue
    txt = "  ".join(f"{nm} P{(seg.prediction < v).mean() * 100:5.2f}" for nm, v in LINES)
    print(f"  第 {i + 1} 折 {seg['date'].min():%Y-%m-%d}~{seg['date'].max():%Y-%m-%d}"
          f"  {len(seg):7,} 行  中位 {seg.prediction.median():+.5f}  σ "
          f"{seg.prediction.std():.5f}  P97={seg.prediction.quantile(.97):.5f}  | {txt}")
print("  → 每折是一份**新模型**，分数水位本身在动；写死的线在折与折之间的实际分位也不同。")

# ==================== 5. 反过来：按不同窗口重标，三条线会移到哪 ====================
print()
print("=" * 96)
print("### 5. 如果改用某个窗口重标（P84/P94/P97 三条），数值要改成多少")
print(f"  {'窗口':<22}" + "".join(f"{nm:>18}" for nm, _ in LINES))
win = [("全样本（现行标定）", pred),
       ("最近 12 个月", pred[pred.date >= pred["date"].max() - pd.DateOffset(months=12)]),
       ("最近 6 个月", pred[pred.date >= pred["date"].max() - pd.DateOffset(months=6)]),
       ("最近 3 个月", pred[pred.date >= pred["date"].max() - pd.DateOffset(months=3)]),
       ("第 8 折（2026-07 起）", pred[pred.date >= bounds[7]]),
       ("2026 年", pred[pred.date >= pd.Timestamp("2026-01-01")])]
base = {nm: float(pred.prediction.quantile(q))
        for (nm, _), q in zip(LINES, (.84, .94, .97))}
for lab, d in win:
    vals = [float(d.prediction.quantile(q)) for q in (.84, .94, .97)]
    print(f"  {lab:<22}" + "".join(f"{v:>12.5f}" for v in vals)
          + "   " + " ".join(f"{nm[:2]}{v - b:+.5f}"
                             for nm, v, b in zip([n for n, _ in LINES], vals,
                                                 [base[n] for n in base])))
print("  （最后一列 = 与现行三个写死的数之差；`pred.prediction.quantile` 与标定用的"
      "配对样本口径略有差别，全样本/配对口径的差见 §1）")
print()
print("=" * 96)
print("### 6. 结论与复标方法")
print("  1. 百分位**不是**实时算的：三条线是 config.yaml 里的绝对数，改它要改文件 + 重跑回测。")
print("  2. 它们当初是**整份 OOS 分布**上的 P84/P94/P97（research/calib_stats.py §1），"
      "不是任何一天或最近一段的。")
print("  3. 会漂的原因有两个，都在上面量过：" 
      "(a) 分数水位随市场月份漂移（§2/§3）；(b) 每 6 个月换一份模型，分布本身被搬动（§4）。")
print("  4. 什么时候必须重标：改了 `model.horizon` 或因子集 → 重训 → "
      "`python research/calib_stats.py` 读 §1 的分位数，把三条数抄回 config，再跑回测。"
      "没重训也想要新尺子的话，同一份缓存上跑 `research/policy_grid.py` 横扫档位即可。")
print("  5. 重标会改变的是**松紧**（每天有几只过线），不改变 alpha：见 §5 那三列的间距。")
