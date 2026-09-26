# -*- coding: utf-8 -*-
"""P94/P97 两条线的标定口径统计：一份脚本出文档里所有引用到的数字。

    python research/calib_stats.py
"""
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

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
H = int(cfg["model"]["horizon"])

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])

# 与 research/policy_grid.py §1 同口径：预测与"实际前向 h 日收益"成对出现的观测才进分布
from data.cache import CacheManager

cache = CacheManager(cfg["cache"]["directory"])
bars = {}
for s in pred["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df.sort_values("日期")
close = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars.items()})
fwd = close.shift(-H) / close - 1.0
fwd.index = pd.to_datetime(fwd.index)

p = pred.rename(columns={"date": "日期"})
pair = p.merge(fwd.stack().rename("actual").reset_index().rename(
    columns={"level_0": "日期", "level_1": "symbol"}), on=["日期", "symbol"], how="inner")
s = pair["prediction"]
print("### 1. 分数分布（配对口径，样本 %d 行 / %d 个交易日 / %d 只）"
      % (len(s), pair["日期"].nunique(), pair["symbol"].nunique()))
print("  中位 %+.3f%%  均值 %+.3f%%  σ %.3f%%"
      % (s.median() * 100, s.mean() * 100, s.std() * 100))
for q in (50, 84, 90, 94, 96.92, 97, 99):
    print("  P%-6.2f = %+.5f" % (q, np.percentile(s, q)))

BUY, STRONG = (float(cfg["position_policy"]["buy_score"]),
               float(cfg["position_policy"]["strong_score"]))
print("  config: buy_score=%.5f strong_score=%.5f" % (BUY, STRONG))
print("  在配对口径上的分位: buy=P%.2f  strong=P%.2f"
      % ((s < BUY).mean() * 100, (s < STRONG).mean() * 100))

print()
print("### 2. 每日过线只数（12 个新仓名额会不会招不满）")
g = pair.groupby("日期")["prediction"]
for label, thr in (("旧线 0.00892", 0.00892), ("买线 %.5f" % BUY, BUY),
                   ("强线 %.5f" % STRONG, STRONG)):
    cnt = g.apply(lambda x: int((x >= thr).sum()))
    yr = cnt.groupby(cnt.index.year).agg(["min", "median", "mean"])
    print("  %-14s 全样本日均 %5.1f 只 / 中位 %3d / 最低 %3d（%s）| 不足 12 只的日子 %.1f%%"
          % (label, cnt.mean(), cnt.median(), cnt.min(),
             cnt.idxmin().strftime("%Y-%m-%d"), (cnt < 12).mean() * 100))
    print("      分年最低/中位/日均: " + "  ".join(
        "%d %d/%d/%.0f" % (y, r["min"], r["median"], r["mean"]) for y, r in yr.iterrows()))

print()
print("### 3. 过线观测的成色（预测均值 vs 实际前向 %d 日均值 vs 基准）" % H)
base_a, base_p = pair["actual"].mean(), pair["prediction"].mean()
print("  预测截面 σ %.3f%% | 实际前向 %d 日 σ %.2f%% | Pearson(预测,实际) %.4f | "
      "Spearman %.4f"
      % (s.std() * 100, H, pair["actual"].std() * 100,
         pair["prediction"].corr(pair["actual"]),
         pair["prediction"].corr(pair["actual"], method="spearman")))
print("  全体            n=%-8d 预测 %+.3f%%  实际 %+.3f%%  实际>0 %.1f%%"
      % (len(pair), base_p * 100, base_a * 100, (pair["actual"] > 0).mean() * 100))
for label, thr in (("≥ 买线 %.5f" % BUY, BUY), ("≥ 强线 %.5f" % STRONG, STRONG),
                   ("旧买线 0.00892", 0.00892)):
    t = pair[pair["prediction"] >= thr]
    print("  %-16s n=%-8d 预测 %+.3f%%  实际 %+.3f%%  实际>0 %.1f%%  相对全体 %+.3fpp"
          % (label, len(t), t["prediction"].mean() * 100, t["actual"].mean() * 100,
             (t["actual"] > 0).mean() * 100, (t["actual"].mean() - base_a) * 100))

print()
print("### 4. 最新截面（%s）过线只数" % pred["date"].max().date())
last = pred[pred["date"] == pred["date"].max()]
print("  当日 %d 只有分，中位 %+.2f%%  p90 %+.2f%%  最高 %+.5f = %+.2f%%（%s）"
      % (len(last), last["prediction"].median() * 100,
         last["prediction"].quantile(.9) * 100, last["prediction"].max(),
         last["prediction"].max() * 100,
         last.loc[last["prediction"].idxmax(), "symbol"]))
for label, thr in (("买线", BUY), ("强线", STRONG), ("旧买线", 0.00892),
                   ("旧强线", 0.01794)):
    print("  ≥ %s %.5f: %d 只" % (label, thr, int((last["prediction"] >= thr).sum())))

print()
print("### 5. 名单顶端的集中度（月末信号日，超额 = 前 N 只等权 − 当日全池等权）")
H2 = 20
fwd2 = close.shift(-H2) / close - 1.0
score_w = pred.pivot_table(index="date", columns="symbol", values="prediction")
ds = pd.DatetimeIndex(sorted(pred["date"].unique()))
sig = ds.to_series().groupby(ds.to_series().dt.to_period("M")).max().tolist()
rec = []
for d in sig:
    if d not in score_w.index:
        continue
    sc = score_w.loc[d].dropna()
    for h, fw in ((H, fwd), (H2, fwd2)):
        a = fw.loc[d].reindex(sc.index) if d in fw.index else None
        if a is None:
            continue
        ok = a.notna()
        if ok.sum() < 100:
            continue
        sc_ok, a_ok = sc[ok], a[ok]
        for n in (3, 12, 30, 100):
            top = sc_ok.nlargest(n).index
            rec.append(dict(date=d, h=h, n=n, pool=float(a_ok.mean()),
                            ex=float(a_ok[top].mean() - a_ok.mean())))
r = pd.DataFrame(rec)
for (h, n), g in r.groupby(["h", "n"], sort=False):
    e, tot = g["ex"], g["ex"].sum()
    top3 = g.nlargest(3, "ex")["ex"].sum()
    rest = g[~g.index.isin(g.nlargest(3, "ex").index)]["ex"]
    print("  随后 %2d 日 top%-3d n=%2d 月均 %+0.3f%% σ %0.3f%% 中位 %+0.3f%% 为正 %2d/%2d "
          "t=%+0.2f | Σ %+0.1f%% 最好3月占 %5.1f%% 去最好3月后月均 %+0.3f%%"
          % (h, n, len(g), e.mean() * 100, e.std() * 100, e.median() * 100, int((e > 0).sum()), len(e),
             e.mean() / (e.std() / np.sqrt(len(e))), tot * 100,
             (top3 / tot * 100) if tot > 0 else float("nan"), rest.mean() * 100))
print("  （对照：同期全池等权月均 " + "  ".join(
    "%d日 %+0.2f%%" % (h, g["pool"].mean() * 100)
    for h, g in r.groupby("h", sort=False)) + "，信号日 %d 个月末）" % len(sig))

print()
print("### 6. 暴露层 B 的输入分布：全池等权 20 日已实现年化波动（`utils/exposure.py` 同口径）")
ov = cfg["exposure_overlay"]
lb, floor = int(ov["vol_lookback_days"]), float(ov["scale_floor"])
ew = close.pct_change().mean(axis=1)
rv = (ew.rolling(lb).std(ddof=1) * np.sqrt(252)).dropna()
at = rv.reindex([d for d in sig if d in rv.index]).dropna()
print("  评估日样本 %d 个：中位 %.1f%%  最低 %.1f%%（%s）  最高 %.1f%%  均值 %.1f%%"
      % (len(at), at.median() * 100, at.min() * 100,
         at.idxmin().strftime("%Y-%m"), at.max() * 100, at.mean() * 100))
from utils.position_policy import policy_from_config

base = policy_from_config(cfg).max_total_pct
for tv in (0.08, 0.10, 0.12):
    sc = np.clip(tv / at.values, floor, 1.0)
    print("  目标波动 %.0f%% → 缩放中位 %.2f，仓位上限中位 %.1f%%（区间 %.1f%%~%.1f%%），"
          "%d/%d 个评估日撞地板 %.2f" % (
              tv * 100, np.median(sc), np.median(sc) * base * 100,
              sc.min() * base * 100, sc.max() * base * 100,
              int((tv / at.values < floor).sum()), len(at), floor))
