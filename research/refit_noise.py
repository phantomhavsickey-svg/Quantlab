# -*- coding: utf-8 -*-
"""折 6 上的对照实验：同一条数据、只换随机种子 / 只打乱行序，分数会差多少。

    python research/refit_noise.py

折 6 的窗口：训练截止 2025-07-07，预测窗口 2025-07-07 ~ 2026-01-05。
选折 6 是因为两臂在它上面的 best_iteration 完全相同（500 -> 500），
所以"两臂分数差"不可能是早停切点造成的，只能是数据/随机性。

变体：
  A 现面板 + 种子 42（config.yaml 默认），早停口径照 main.py:324-326
  B 现面板 + 种子 7
  C 现面板 + 种子 42，但训练行先整体打乱
输出：三个变体两两之间、以及它们与两臂存档分数之间的逐日截面排序相关。
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

from data.cache import CacheManager
from models.dataset import DatasetBuilder
from models.trainer import LightGBMTrainer

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
mcfg = cfg["model"]
WS = pd.Timestamp("2025-07-07")   # 折 6 训练截止 = 预测窗口起点
WE = pd.Timestamp("2026-01-05")

print("### 0. 搭数据集（口径照 main.py:250-281）")
panel = pd.read_parquet("data/cache/factor_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
cache = CacheManager(cfg["cache"]["directory"])
data_dict = {}
for sym in panel["symbol"].unique().tolist():
    df = cache.get_daily(sym)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        data_dict[sym] = df
builder = DatasetBuilder(horizon=mcfg["horizon"], model_type=mcfg["type"])
feature_cols = [c for c in panel.columns if c not in ["date", "symbol"]]
feature_matrix = panel.set_index(["date", "symbol"])[feature_cols]
dataset = builder.merge_features_labels(feature_matrix, data_dict).dropna()
# 不要排序：面板本身是"按股票分组"的行序，main.py 也原样喂给 LightGBM，
# 而 bagging 抽哪些行是按行号来的，行序一换模型就换（这正是本脚本要量的东西）。
feature_cols = [c for c in dataset.columns
                if c not in ["date", "symbol", "label", "forward_return"]]
label_col = "label" if mcfg["type"] == "classifier" else "forward_return"

tr = dataset[dataset["date"] < WS]
pr = dataset[(dataset["date"] >= WS) & (dataset["date"] < WE)]
print("  折 6 训练行 %d，预测行 %d（预测窗口 %s ~ %s）"
      % (len(tr), len(pr), WS.date(), WE.date()))

old_panel = pd.read_parquet("data/cache/backup_20260930/factor_panel.parquet")
old_panel["date"] = pd.to_datetime(old_panel["date"])
old_tr = old_panel[old_panel["date"] < WS]
ko = set(map(tuple, tr[["date", "symbol"]].to_numpy()))
kp = set(map(tuple, old_tr[["date", "symbol"]].to_numpy()))
print("  旧面板同期训练行 %d；两边都有的 %d，只在旧里的 %d，只在新里的 %d"
      % (len(old_tr), len(ko & kp), len(kp - ko), len(ko - kp)))


def fit_predict(X, y, Xp, seed, shuffle_rows=False):
    X2, y2 = X, y
    if shuffle_rows:
        perm = np.random.default_rng(0).permutation(len(X))
        X2, y2 = X.iloc[perm], y.iloc[perm]
    mid = int(len(X2) * 0.85)
    params = dict(mcfg.get("params") or {})
    params["random_state"] = seed
    trainer = LightGBMTrainer(model_type=mcfg["type"], params=params)
    res = trainer.train(X2.iloc[:mid], y2.iloc[:mid],
                        X2.iloc[mid:], y2.iloc[mid:])
    p = trainer.model.predict(Xp)
    return p, res


Xp = pr[feature_cols]
X, y = tr[feature_cols], tr[label_col]
pa, ra = fit_predict(X, y, Xp, 42)
print("  A 变体训练完成: best_iteration=%s" % ra.get("best_iteration"))
pb, rb = fit_predict(X, y, Xp, 7)
print("  B 变体训练完成: best_iteration=%s" % rb.get("best_iteration"))
pc, rc = fit_predict(X, y, Xp, 42, shuffle_rows=True)
print("  C 变体训练完成: best_iteration=%s" % rc.get("best_iteration"))

out = pr[["date", "symbol"]].copy()
out["A"] = pa
out["B"] = pb
out["C"] = pc
for tag, path in (("旧臂", "data/cache/backup_20260930/predictions.parquet"),
                  ("新臂", "data/cache/backup_20261009/predictions.parquet")):
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= WS) & (d["date"] < WE)][["date", "symbol", "prediction"]]
    out = out.merge(d.rename(columns={"prediction": tag}), on=["date", "symbol"], how="left")

print()
print("### 1. 逐日截面排序相关（折 6 的预测窗口，%d 天）" % out["date"].nunique())
P = ["A", "B", "C", "旧臂", "新臂"]
print("        " + "".join("%9s" % c for c in P))
for a in P:
    line = "%6s " % a
    for b in P:
        if a == b:
            line += "%9s" % "-"
            continue
        rs = [g[a].corr(g[b], method="spearman") for _, g in out.groupby("date") if len(g) >= 30]
        line += "%9.4f" % np.nanmedian(rs)
    print(line)
print("  （格子 = 该行变体与该列变体、逐日 Spearman 的中位；对角为空）")
print()
for a, b in (("A", "B"), ("A", "C"), ("A", "新臂"), ("旧臂", "新臂")):
    rs = np.array([g[a].corr(g[b], method="spearman") for _, g in out.groupby("date") if len(g) >= 30])
    print("  %-8s vs %-4s: 中位 %.4f 均值 %.4f 最差 %.4f" % (a, b, np.median(rs), rs.mean(), rs.min()))
