# -*- coding: utf-8 -*-
"""两分支预测的模型层面对照:逐日 RankIC(选股) + 时序相关(择时) + 分数分布拆分。

    python research/regime_vs_base_ic.py

基线 = 原分支 predictions.parquet(只读,可用 argv[1] 覆盖);regime = 本副本。
两文件行键必须完全一致(同一天同股票),否则对照无意义。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

BASE = sys.argv[1] if len(sys.argv) > 1 else \
    r"G:\个人量化\Quantlab\data\cache\predictions.parquet"
REG = r"G:\个人量化\Quantlab_regime\data\cache\predictions.parquet"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd

base = pd.read_parquet(BASE)
reg = pd.read_parquet(REG)
kb = base.set_index(["date", "symbol"]).index
kr = reg.set_index(["date", "symbol"]).index
print("键一致: %s | 行数 %d | 交易日 %d"
      % (kb.equals(kr), len(base), base["date"].nunique()))
if not kb.equals(kr):
    raise SystemExit("行键不一致,对照作废")


def ic_series(df):
    out = {}
    for d, g in df.groupby("date"):
        if len(g) >= 10:
            out[d] = g["prediction"].corr(g["forward_return"], method="spearman")
    return pd.Series(out)


def fmt(s):
    return "中位 %+.4f | 均值 %+.4f | 交易日 %d" % (s.median(), s.mean(), len(s))


def timing(df, by):
    x = df.copy()
    x["per"] = x["date"].dt.to_period(by)
    tam = x.groupby("per").agg(m=("prediction", "median"), r=("forward_return", "mean"))
    return tam["m"].corr(tam["r"], method="spearman")


def dist(df):
    day_med = df.groupby("date")["prediction"].median()
    day_std = df.groupby("date")["prediction"].std()
    return ("全样本 σ=%.5f | 日间(日中位)std=%.5f | 日内 std 中位=%.5f"
            % (df["prediction"].std(), day_med.std(), day_std.median()))


print()
base_ic, reg_ic = ic_series(base), ic_series(reg)
print("基线   逐日 RankIC: %s" % fmt(base_ic))
print("regime 逐日 RankIC: %s" % fmt(reg_ic))
diff = (reg_ic - base_ic).dropna()
from scipy import stats
print("逐日 RankIC 配对差(regime − 基线): 中位 %+.4f | 均值 %+.4f | regime 更好的天数 %.1f%%"
      % (diff.median(), diff.mean(), (diff > 0).mean() * 100))
t, p = stats.ttest_rel(reg_ic[diff.index], base_ic[diff.index])
print("  配对 t(全日 %d 天): t=%+.2f p=%.4f —— 标签重叠 5 天,全日检验高估显著性"
      % (len(diff), t, p))
for k in (5, 10):
    sub = diff.iloc[::k]
    t, p = stats.ttest_1samp(sub, 0)
    print("  每 %d 个交易日取 1 天(%d 天): 均值差 %+.4f, t=%+.2f, p=%.3f"
          % (k, len(sub), sub.mean(), t, p))
print()
print("基线   分数分布: %s" % dist(base))
print("regime 分数分布: %s" % dist(reg))
print()
print("时序相关(日:全池分数中位 vs 当日全池平均前向收益)")
print("  基线   %+.4f | regime %+.4f" % (timing(base, "D"), timing(reg, "D")))
print("时序相关(月:同口径按月)")
print("  基线   %+.4f | regime %+.4f" % (timing(base, "M"), timing(reg, "M")))
