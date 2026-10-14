# -*- coding: utf-8 -*-
"""按年分解：原始合成因子的 IC vs 模型预测的 IC vs 两者的相关。

    python research/factor_ic_year.py

- 合成因子 = 25 个因子每日截面排名取平均（不做任何拟合，方向由数据决定）
- 模型预测 = data/cache/predictions.parquet（8 折 walk-forward 样本外）
- IC = 逐日截面 Spearman（分数 vs 未来 5 日收益）
输出 logs/factor_ic_year.log
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
logger.add("logs/factor_ic_year.log", format="{message}", encoding="utf-8", mode="w")

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
ds = builder.merge_features_labels(panel.set_index(["date", "symbol"])[fcols],
                                   data_dict).dropna()
d = ds[["date", "symbol", "forward_return"] + fcols].dropna().copy()
d["comp"] = d.groupby("date")[fcols].rank().mean(axis=1)
d["rc"] = d.groupby("date")["comp"].rank()
d["rr"] = d.groupby("date")["forward_return"].rank()
d["year"] = d["date"].dt.year


def daily_ic(df, xcol):
    return df.groupby("date").apply(lambda g: g[xcol].corr(g["rr"]))


ic_comp = daily_ic(d, "rc")


def tab(ic, name):
    logger.info("  %s" % name)
    logger.info("    %-6s %8s %8s %8s %8s %7s %6s" %
                ("年", "IC均值", "IC中位", "IC_std", "ICIR", "正比例", "天数"))
    for y, g in ic.groupby(ic.index.year):
        logger.info("    %-6d %+8.4f %+8.4f %8.4f %+8.3f %6.1f%% %6d" %
                    (y, g.mean(), g.median(), g.std(), g.mean() / g.std(),
                     100 * (g > 0).mean(), g.count()))
    logger.info("    %-6s %+8.4f %+8.4f %8.4f %+8.3f %6.1f%% %6d" %
                ("全部", ic.mean(), ic.median(), ic.std(),
                 ic.mean() / ic.std(), 100 * (ic > 0).mean(), ic.count()))


logger.info("### 1. 原始合成因子（25 个因子排名平均，无拟合）")
tab(ic_comp, "方向按数据里的实际符号（负 = 分数越高、未来 5 日收益越低）")
logger.info("  反号使用（低分买入）后的 IC = %+.4f" % (-ic_comp.mean()))

# 模型预测
pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
m = pred[["date", "symbol", "prediction", "forward_return"]].dropna().copy()
m["rp"] = m.groupby("date")["prediction"].rank()
m["rr"] = m.groupby("date")["forward_return"].rank()
ic_m = daily_ic(m, "rp")

logger.info("")
logger.info("### 2. 模型预测（predictions.parquet，样本外）")
tab(ic_m, "8 折 walk-forward 预测")

# 模型预测与合成因子的每日排名相关
mg = pred[["date", "symbol", "prediction"]].dropna().copy()
dd = d[["date", "symbol", "comp"]].copy()
j = mg.merge(dd, on=["date", "symbol"], how="inner")
j = j.assign(rc=j.groupby("date")["comp"].rank(), rp=j.groupby("date")["prediction"].rank())
cr = j.groupby("date").apply(lambda g: g["rc"].corr(g["rp"]))
logger.info("")
logger.info("### 3. 模型预测 与 原始合成因子 的每日排名相关（重叠区间 %d 天，%s ~ %s）"
            % (cr.count(), cr.index.min().date(), cr.index.max().date()))
logger.info("    中位 %+.4f | 均值 %+.4f | 最差 %+.4f | 最好 %+.4f"
            % (cr.median(), cr.mean(), cr.min(), cr.max()))
for y, g in cr.groupby(cr.index.year):
    logger.info("    %d 年：中位 %+.4f（%d 天）" % (y, g.median(), g.count()))
