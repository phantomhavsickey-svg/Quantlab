# -*- coding: utf-8 -*-
"""对照新旧两版因子面板:结构、NaN 掩码、25 列内容、dataset.dropna 口径。

    python research/regime_panel_check.py

旧面板路径默认指向同机并存的原分支(只读),可用 argv[1] 覆盖。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

OLD = sys.argv[1] if len(sys.argv) > 1 else \
    r"G:\个人量化\Quantlab\data\cache\factor_panel.parquet"
NEW = r"G:\个人量化\Quantlab_regime\data\cache\factor_panel.parquet"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd

old = pd.read_parquet(OLD)
new = pd.read_parquet(NEW)
old["date"] = pd.to_datetime(old["date"])
new["date"] = pd.to_datetime(new["date"])
mk = ["mkt_ret_20d", "mkt_vol_20d", "mkt_snr_20d", "mkt_breadth_20d", "mkt_dd_60d"]
base_cols = [c for c in old.columns if c not in ("date", "symbol")]

print("old:", old.shape, "| new:", new.shape)
print("old cols:", len(base_cols), "| new extra:", sorted(set(new.columns) - set(old.columns)))

# 1. 行键与逐日行数
ko = old.set_index(["date", "symbol"]).sort_index()
kn = new.set_index(["date", "symbol"]).sort_index()
print("\nrow keys identical:", ko.index.equals(kn.index))
co, cn = old.groupby("date").size(), new.groupby("date").size()
print("per-date row counts identical:", bool((co == cn.reindex(co.index)).all()))

# 2. 25 列 + date/symbol 内容逐格一致
same = True
for c in base_cols:
    a, b = ko[c].to_numpy(), kn[c].to_numpy()
    both_nan = pd.isna(a) & pd.isna(b)
    eq = np.isclose(a, b, rtol=0, atol=0, equal_nan=True)
    if not (eq | both_nan).all():
        bad = int((~(eq | both_nan)).sum())
        print("  列 %s 有 %d 格不同" % (c, bad))
        same = False
print("25 列内容逐格一致:", same)

# 3. 新列 NaN 结构
nan_rows = kn[kn[mk].isna().any(axis=1)]
print("\n新列 NaN 行数:", len(nan_rows), "| 全部在:", sorted(nan_rows.index.get_level_values('date').unique().astype(str)))

# 4. dataset.dropna 口径对比(用本分支的代码与缓存)
from data.cache import CacheManager
from models.dataset import DatasetBuilder
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cache = CacheManager(cfg["cache"]["directory"])
builder = DatasetBuilder(horizon=cfg["model"]["horizon"], model_type=cfg["model"]["type"])

def build(panel):
    data_dict = {}
    for s in panel["symbol"].unique().tolist():
        df = cache.get_daily(s)
        if df is not None:
            df["日期"] = pd.to_datetime(df["日期"])
            data_dict[s] = df
    fc = [c for c in panel.columns if c not in ["date", "symbol"]]
    ds = builder.merge_features_labels(
        panel.set_index(["date", "symbol"])[fc], data_dict).dropna()
    return ds

ds_o = build(old)
ds_n = build(new)
do = sorted(ds_o["date"].unique())
dn = sorted(ds_n["date"].unique())
print("\ndataset old: %d 条 / %d 日 | new: %d 条 / %d 日" % (len(ds_o), len(do), len(ds_n), len(dn)))
print("dates identical:", do == dn)
if do != dn:
    so, sn = set(do), set(dn)
    print("  only old:", sorted(str(x) for x in (so - sn))[:10])
    print("  only new:", sorted(str(x) for x in (sn - so))[:10])
print("\n第一/最后交易日 old:", do[0].date(), do[-1].date(), "| new:", dn[0].date(), dn[-1].date())
