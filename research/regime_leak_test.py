# -*- coding: utf-8 -*-
"""市场模式因子的截断前视测试。

把每只股票的日线截到日期 D 之前,重算 5 个市场模式列,与全样本结果在
"日期 ≤ D" 的公共行上比对:有前视就会出现 O(1e-2) 量级的差异。浮点末位
(pandas rolling 核因数组长度不同产生的 ~1e-16 相对差)不算差异,报告最大相对差。

    python research/regime_leak_test.py
"""
import os
import sys

warnings_flag = True
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import yaml

from data.cache import CacheManager
from factors.market_regime import MARKET_COLS, compute_market_regime

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cache = CacheManager(cfg["cache"]["directory"])
dd = {}
for s in cache.list_cached_symbols():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        dd[s] = df

full = compute_market_regime(dd).set_index("date")
print("full frame:", full.shape)
for cut in ("2022-06-30", "2024-03-29", "2026-05-29"):
    cut_ts = pd.Timestamp(cut)
    part = compute_market_regime(
        {s: df[df["日期"] <= cut_ts] for s, df in dd.items()}).set_index("date")
    common = part.index.intersection(full.index)
    a = part.loc[common, list(MARKET_COLS)].to_numpy(dtype=float)
    b = full.loc[common, list(MARKET_COLS)].to_numpy(dtype=float)
    scale = np.maximum(np.abs(b), 1e-12)
    rel = np.abs(a - b) / scale
    print("cut<=%s | 重叠 %d 行 | 最大相对差 %.3e | 超过 1e-9 的格: %d"
          % (cut, len(common), float(np.nanmax(rel)), int((rel > 1e-9).sum())))
