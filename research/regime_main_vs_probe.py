# -*- coding: utf-8 -*-
"""主流程(seed42)与复算脚本(_probe_regime_seed42)预测逐位一致性检查。

    python research/regime_main_vs_probe.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd

main = pd.read_parquet("data/cache/predictions.parquet")
probe = pd.read_parquet("data/cache/_probe_regime_seed42/predictions.parquet")

lines = []


def log(s):
    print(s)
    lines.append(s)


log("main : %s cols=%s" % (main.shape, list(main.columns)))
log("probe: %s cols=%s" % (probe.shape, list(probe.columns)))

m = main.merge(probe, on=["date", "symbol"], suffixes=("_main", "_probe"))
log("配对行: %d | main 独有 %d | probe 独有 %d"
    % (len(m), len(main) - len(m), len(probe) - len(m)))

d = (m["prediction_main"] - m["prediction_probe"]).abs()
log("最大绝对差: %r | 逐位一致: %s" % (float(d.max()), bool((d == 0).all())))

with open("logs/regime_main_vs_probe.log", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
