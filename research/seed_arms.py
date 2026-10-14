# -*- coding: utf-8 -*-
"""同一条数据、同一套代码，只换随机种子，把回测整条跑一遍。

    python research/seed_arms.py

这段脚本回答的问题：+200.92% -> +39.10% 这 161.82pp 里，有多少是"数据版本"造成的？
折 6 的对照已经证明：只换 random_state（42 -> 7）这一件事，就能把逐日分数排序
从 1.0000 拉到 0.8751（实测 logs/refit_noise_20261009.log）。那么整条回测会动多少？

口径完全照 main.py：同一份 factor_panel.parquet、同一份日线缓存、同一套折窗口
（main.py:291-307）、同一个 85/15 早停切法（main.py:324-326）。
预测写到 data/cache/_probe_seed{seed}/predictions.parquet，不动现有 predictions.parquet。
"""
import contextlib
import dataclasses as dc
import io
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

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
from data.cache import CacheManager
from data.downloader import DataDownloader
from models.dataset import DatasetBuilder
from models.predictor import scores_from_predictions
from models.trainer import LightGBMTrainer
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

SEEDS = [int(x) for x in sys.argv[1:]] or [7, 13]

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
mcfg, cb, cm = cfg["model"], cfg["backtest"], cfg["market"]
wf = mcfg["walk_forward"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])

print("### 0. 搭数据集（口径照 main.py:250-281）")
panel = pd.read_parquet("data/cache/factor_panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
data_dict = {}
for sym in panel["symbol"].unique().tolist():
    df = cache.get_daily(sym)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        data_dict[sym] = df
builder = DatasetBuilder(horizon=mcfg["horizon"], model_type=mcfg["type"])
feature_cols = [c for c in panel.columns if c not in ["date", "symbol"]]
dataset = builder.merge_features_labels(
    panel.set_index(["date", "symbol"])[feature_cols], data_dict).dropna()
feature_cols = [c for c in dataset.columns
                if c not in ["date", "symbol", "label", "forward_return"]]
label_col = "label" if mcfg["type"] == "classifier" else "forward_return"
print("  数据集 %d 条 | %d 个交易日" % (len(dataset), dataset["date"].nunique()))

dates = sorted(dataset["date"].unique())
raw_windows, ws = [], dates[0] + pd.DateOffset(months=wf["min_train_months"])
while ws <= dates[-1]:
    raw_windows.append((ws, ws + pd.DateOffset(months=wf["retrain_months"])))
    ws = ws + pd.DateOffset(months=wf["retrain_months"])
windows = []
for ws, we in raw_windows:
    wsd = next((d for d in dates if d >= ws), None)
    if wsd is None:
        break
    wed = next((d for d in dates if d > wsd and d >= we), None)
    if wed is None:
        wed = dates[-1] + pd.Timedelta(days=1)
    windows.append((wsd, wed))
print("  折窗口 %d 个: %s ... %s" % (len(windows), windows[0][0].date(), windows[-1][0].date()))

# 基准与 bars 池
uni = set(panel["symbol"])
bars_all = {}
for s in sorted(uni):
    df = cache.get_daily(s)
    if df is not None:
        bars_all[s] = df.sort_values("日期")
try:
    BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
        cb.get("benchmark", "000852"), "20210101",
        max(d["日期"].max() for d in bars_all.values()).strftime("%Y%m%d")).set_index("日期")["收盘"]
except Exception as e:
    print("  基准取不到(%s)，用 None" % e)
    BM = None

# 参照臂 = 现有 predictions.parquet（seed 42）
ref = pd.read_parquet("data/cache/predictions.parquet")
ref["date"] = pd.to_datetime(ref["date"])


def backtest(pred_df, tag):
    PR = pred_df.set_index(["date", "symbol"])["prediction"]
    SC = scores_from_predictions(PR, build_tradable_mask(bars_all, pred_df["date"].unique()))
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"],
                             cm.get("stamp_tax_schedule") or None),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=dc.replace(policy_from_config(cfg)),
                         overlay=overlay_from_config(cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars_all, SC, benchmark_prices=BM)
    eq, tr = r["equity_curve"], r["trades"]
    yr = eq.groupby(eq.index.year).agg(["first", "last"])
    print("  %-10s 累计 %+8.2f%%  笔数 %4d  成本 %7.0f 元  平均目标仓位 %.1f%%"
          % (tag, pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum(),
             (r["execution"].get("policy_mean_gross_weight") or 0) * 100))
    print("             分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                                          for y, (f, l) in yr.iterrows()))
    return eq, tr


print()
print("### 1. 参照臂（现有 predictions.parquet，random_state=42）")
backtest(ref, "seed 42")

for seed in SEEDS:
    print()
    print("### seed %d：全 8 折重训" % seed)
    preds_all = []
    params = dict(mcfg.get("params") or {})
    params["random_state"] = seed
    for fi, (ws, we) in enumerate(windows):
        tr_mask = dataset["date"] < ws
        pr_mask = (dataset["date"] >= ws) & (dataset["date"] < we)
        if tr_mask.sum() < 5000 or pr_mask.sum() == 0:
            continue
        X, y = dataset.loc[tr_mask, feature_cols], dataset.loc[tr_mask, label_col]
        mid = int(len(X) * 0.85)
        trainer = LightGBMTrainer(model_type=mcfg["type"], params=params)
        res = trainer.train(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])
        fold = dataset.loc[pr_mask, ["date", "symbol"]].copy()
        fold["prediction"] = trainer.model.predict(dataset.loc[pr_mask, feature_cols])
        fold["forward_return"] = dataset.loc[pr_mask, label_col].values
        preds_all.append(fold)
        ic = fold.groupby("date").apply(
            lambda g: g["prediction"].corr(g["forward_return"], method="spearman"),
            include_groups=False).mean()
        print("    Fold %d 样本 %6d best_iteration=%3s 日均RankIC=%+.4f"
              % (fi + 1, int(pr_mask.sum()), res["best_iteration"], ic))
    pdf = pd.concat(preds_all, ignore_index=True)
    outdir = "data/cache/_probe_seed%d" % seed
    os.makedirs(outdir, exist_ok=True)
    pdf.to_parquet(os.path.join(outdir, "predictions.parquet"), index=False)
    print("  预测已存: %s/predictions.parquet (%d 条)" % (outdir, len(pdf)))
    backtest(pdf, "seed %d" % seed)
