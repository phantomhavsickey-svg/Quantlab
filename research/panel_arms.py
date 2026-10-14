# -*- coding: utf-8 -*-
"""指定面板 + 指定随机种子，跑整条 walk-forward 回测。

    python research/panel_arms.py data/cache/backup_20260930/factor_panel.parquet 42 7

用来回答：+200.92% 是那张旧面板的"稳定结果"，还是换一次随机数就没了？
口径照 main.py；日线缓存与标签用当前这份（旧缓存已被覆盖，这一点必须写明）。
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

PANEL = sys.argv[1]
SEEDS = [int(x) for x in sys.argv[2:]] or [42]
TAG = os.path.basename(os.path.dirname(PANEL))

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
mcfg, cb, cm = cfg["model"], cfg["backtest"], cfg["market"]
wf = mcfg["walk_forward"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])

panel = pd.read_parquet(PANEL)
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

print("### 0. %s（%s）" % (TAG, PANEL))
print("  面板 %d 行 %d 只，到 %s | 数据集 %d 条 | %d 个交易日"
      % (len(panel), panel["symbol"].nunique(), panel["date"].max().date(),
         len(dataset), dataset["date"].nunique()))

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
    print("  基准取不到(%s)" % e)
    BM = None


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
    print("  %-16s 累计 %+8.2f%%  笔数 %4d  成本 %7.0f 元  平均目标仓位 %.1f%%"
          % (tag, pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum(),
             (r["execution"].get("policy_mean_gross_weight") or 0) * 100))
    print("                   分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                                               for y, (f, l) in yr.iterrows()))
    return tr


for seed in SEEDS:
    preds_all = []
    params = dict(mcfg.get("params") or {})
    params["random_state"] = seed
    its = []
    for fi, (ws, we) in enumerate(windows):
        tr_mask = dataset["date"] < ws
        pr_mask = (dataset["date"] >= ws) & (dataset["date"] < we)
        if tr_mask.sum() < 5000 or pr_mask.sum() == 0:
            continue
        X, y = dataset.loc[tr_mask, feature_cols], dataset.loc[tr_mask, label_col]
        mid = int(len(X) * 0.85)
        trainer = LightGBMTrainer(model_type=mcfg["type"], params=params)
        res = trainer.train(X.iloc[:mid], y.iloc[:mid], X.iloc[mid:], y.iloc[mid:])
        its.append(res["best_iteration"])
        fold = dataset.loc[pr_mask, ["date", "symbol"]].copy()
        fold["prediction"] = trainer.model.predict(dataset.loc[pr_mask, feature_cols])
        fold["forward_return"] = dataset.loc[pr_mask, label_col].values
        preds_all.append(fold)
    pdf = pd.concat(preds_all, ignore_index=True)
    outdir = "data/cache/_probe_%s_seed%d" % (TAG, seed)
    os.makedirs(outdir, exist_ok=True)
    pdf.to_parquet(os.path.join(outdir, "predictions.parquet"), index=False)
    print()
    print("### %s + seed %d | best_iteration %s | 预测 %d 条 -> %s"
          % (TAG, seed, its, len(pdf), outdir))
    backtest(pdf, "%s seed%d" % (TAG, seed))
