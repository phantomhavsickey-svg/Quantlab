# -*- coding: utf-8 -*-
"""四臂（旧面板 seed42/seed7、新面板 seed7/seed13）在 2024-03-29 的 300641 分数、名次、过建仓线只数。

    python research/seed_score_probe.py
"""
import os

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DAY = pd.Timestamp("2024-03-29")


def find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            r = find_key(v, key)
            if r is not None:
                return r
    return None


LINE = float(find_key(yaml.safe_load(open("config.yaml", encoding="utf-8")), "buy_score"))

ARMS = [
    ("旧面板 seed42", "data/cache/_probe_backup_20260930_seed42/predictions.parquet"),
    ("旧面板 seed7", "data/cache/_probe_backup_20260930_seed7/predictions.parquet"),
    ("新面板 seed7", "data/cache/_probe_seed7/predictions.parquet"),
    ("新面板 seed13", "data/cache/_probe_seed13/predictions.parquet"),
]
print("buy_score = %.5f (config.yaml:95)" % LINE)
for tag, path in ARMS:
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    d = p[p["date"] == DAY].dropna(subset=["prediction"])
    rk = d["prediction"].rank(ascending=False, method="min")
    n = len(d)
    row = d[d["symbol"] == "300641"]
    if not len(row):
        print("%s 2024-03-29: 300641 无预测 (全市场 %d 只)" % (tag, n))
        continue
    s = float(row["prediction"].iloc[0])
    r = int(rk[row.index[0]])
    print("%s 2024-03-29: 全市场 %d 只 | 过建仓线 %d 只 | 300641 分数 %.5f 第 %d 名 (位次 %.1f%%)"
          % (tag, n, int((d["prediction"] >= LINE).sum()), s, r, (n - r + 1) / n * 100))
    top = d.nlargest(5, "prediction")[["symbol", "prediction"]]
    print("   前5: " + "  ".join("%s %.5f" % (a, b) for a, b in top.itertuples(index=False)))
