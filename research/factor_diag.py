# -*- coding: utf-8 -*-
"""25 个因子的体检：单因子 IC、因子间相关、模型对因子的使用集中度。

    python research/factor_diag.py

数据范围：data/cache/factor_panel.parquet（当前面板）+ 日线缓存现算的标签。
标签 = 未来 5 日收益（与 config.yaml model.horizon=5 一致）。
单因子 IC = 逐日截面 Spearman（因子值排名 vs 未来 5 日收益排名）。
相关矩阵 = 全样本堆叠的 Pearson 相关（因子已 winsorize+zscore）。
输出 logs/factor_diag.log
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
from loguru import logger

logger.remove()
logger.add("logs/factor_diag.log", format="{message}", encoding="utf-8", mode="w")

from data.cache import CacheManager
from models.dataset import DatasetBuilder

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
mcfg = cfg["model"]

panel = pd.read_parquet("data/cache/factor_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
fcols = [c for c in panel.columns if c not in ("date", "symbol")]
cache = CacheManager(cfg["cache"]["directory"])
data_dict = {}
for sym in panel["symbol"].unique().tolist():
    df = cache.get_daily(sym)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        data_dict[sym] = df
builder = DatasetBuilder(horizon=mcfg["horizon"], model_type=mcfg["type"])
fm = panel.set_index(["date", "symbol"])[fcols]
ds = builder.merge_features_labels(fm, data_dict).dropna()
logger.info("### 0. 样本")
logger.info("  面板 %d 行 × %d 因子；配对后 %d 行，%d 个交易日（%s ~ %s）"
            % (len(panel), len(fcols), len(ds), ds["date"].nunique(),
               ds["date"].min().date(), ds["date"].max().date()))

lab = "forward_return"

# ---------- 1) 单因子逐日 RankIC ----------
logger.info("")
logger.info("### 1. 单因子：逐日截面 RankIC（因子值 vs 未来 5 日收益）")
rows = []
for f in fcols:
    d = ds[["date", f, lab]].dropna()
    d = d.assign(rf=d.groupby("date")[f].rank(), rr=d.groupby("date")[lab].rank())
    ic = d.groupby("date").apply(lambda g: g["rf"].corr(g["rr"]))
    n = ic.count()
    m, s = ic.mean(), ic.std()
    rows.append((f, m, s, m / s, m / (s / np.sqrt(n)), (ic > 0).mean(), n))
ic_tab = pd.DataFrame(rows, columns=["factor", "IC_mean", "IC_std", "ICIR",
                                     "t", "pos_ratio", "days"])
ic_tab = ic_tab.sort_values("IC_mean", key=abs, ascending=False)
logger.info("  %-24s %8s %8s %7s %8s %7s" % ("factor", "IC_mean", "IC_std", "ICIR", "t值", "正比例"))
for _, r in ic_tab.iterrows():
    logger.info("  %-24s %+8.4f %8.4f %+7.3f %+8.2f %6.1f%%"
                % (r["factor"], r["IC_mean"], r["IC_std"], r["ICIR"], r["t"], 100 * r["pos_ratio"]))
logger.info("  |IC_mean| 中位 %.4f，最大 %.4f，最小 %.4f；|IC_mean|≥0.02 的因子 %d 个"
            % (ic_tab["IC_mean"].abs().median(), ic_tab["IC_mean"].abs().max(),
               ic_tab["IC_mean"].abs().min(), (ic_tab["IC_mean"].abs() >= 0.02).sum()))

# ---------- 2) 等权合成因子（全部 25 个，每日排名平均） ----------
d = ds[["date"] + fcols + [lab]].dropna()
comp = d.groupby("date")[fcols].rank().mean(axis=1)
d2 = d[["date", lab]].assign(comp=comp)
d2 = d2.assign(rc=d2.groupby("date")["comp"].rank(), rr=d2.groupby("date")[lab].rank())
ic_c = d2.groupby("date").apply(lambda g: g["rc"].corr(g["rr"]))
logger.info("")
logger.info("### 2. 等权合成（25 个因子每日排名取平均）")
logger.info("  IC 均值 %+.4f | 标准差 %.4f | ICIR %+.3f | 正比例 %.1f%% | 天数 %d"
            % (ic_c.mean(), ic_c.std(), ic_c.mean() / ic_c.std(),
               100 * (ic_c > 0).mean(), ic_c.count()))

# ---------- 3) 因子间相关结构 ----------
logger.info("")
logger.info("### 3. 因子间相关（全样本堆叠 Pearson）")
corr = d[fcols].corr()
off = corr.where(~np.eye(len(fcols), dtype=bool))
absmean = off.abs().stack().mean()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(fcols) for b in fcols[i + 1:]
         if abs(corr.loc[a, b]) > 0.7]
pairs = sorted(pairs, key=lambda x: -abs(x[2]))
ev = np.linalg.eigvalsh(corr.values)[::-1]
pr = ev.sum() ** 2 / (ev ** 2).sum()          # 参与率 = 有效独立方向数
cum = np.cumsum(ev) / ev.sum()
n90 = int(np.searchsorted(cum, 0.90) + 1)
logger.info("  平均 |相关|（非对角）%.3f；|相关|>0.7 的因子对 %d 对" % (absmean, len(pairs)))
for a, b, c in pairs[:12]:
    logger.info("    %-24s %-24s %+.3f" % (a, b, c))
logger.info("  相关矩阵特征值：有效独立方向数（参与率）%.1f / 25；解释 90%% 方差需要 %d 个主方向"
            % (pr, n90))

# ---------- 4) 现在模型对因子的使用集中度 ----------
logger.info("")
logger.info("### 4. 最新模型（models/saved/lgb_regressor_20261009_161240.txt）的因子增益占比")
try:
    import lightgbm as lgb
    booster = lgb.Booster(model_file="models/saved/lgb_regressor_20261009_161240.txt")
    imp = pd.Series(booster.feature_importance("gain"),
                    index=booster.feature_name()).sort_values(ascending=False)
    share = imp / imp.sum()
    logger.info("  %-24s %9s %9s" % ("factor", "gain占比", "累计"))
    cum = 0.0
    for i, (f, v) in enumerate(share.items()):
        cum += v
        logger.info("  %-24s %8.1f%% %8.1f%%" % (f, 100 * v, 100 * cum))
    top5 = share.head(5).sum()
    n_cover = int((share.cumsum() < 0.90).sum() + 1)
    logger.info("  前 5 个因子占 %.1f%% 的总增益；覆盖 90%% 增益需要 %d 个因子" % (100 * top5, n_cover))
except Exception as e:
    logger.info("  模型读取失败: %s" % e)
