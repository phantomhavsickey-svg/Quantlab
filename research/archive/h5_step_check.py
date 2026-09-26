# -*- coding: utf-8 -*-
"""补检：h5 的优势是不是"只重定标 buy/strong、没重定标 step_score_floor"造出来的。

step_score_floor 是补仓一档的最小分数步长（绝对分数口径）。基准档分数带
P84→P97 = 0.030→0.050 宽 0.020，h5 档只有 0.0089→0.0179 宽 0.009，
带内可分的阶梯数从 ~4 档变成 ~2 档。若 h5 的优势在这个参数一动就消失，
说明它靠的是策略几何而不是标签。

归档原因：读的是 data/cache/_c_backup/ 里那批对照预测，该目录只在开发机上存在、
不进 git，换机器跑不出来。结论已写进 README 与对话记录，留档只为翻账。
"""
import contextlib
import dataclasses as dc
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import numpy as np
import pandas as pd
import yaml

from data.cache import CacheManager
from data.downloader import DataDownloader
from models.predictor import scores_from_predictions
from backtest.engine import BacktestEngine
from backtest.cost import TransactionCostModel
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

B = "data/cache/_c_backup"
END = pd.Timestamp("2026-08-25")
Q_BUY, Q_STRONG = 0.84422, 0.96919
VAR = [("orig_h20", 20, B + "/predictions.parquet", 0.030),
       ("orig_h5", 5, B + "/pred_orig_h5.parquet", 0.0089),
       ("neut_h10", 10, B + "/pred_neut_h10.parquet", 0.0193)]

cfg0 = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg0["backtest"], cfg0["market"]
CAP = cb.get("initial_capital", 1_000_000)
pol0 = policy_from_config(cfg0)
ocfg0 = overlay_from_config(cfg0)

cache = CacheManager(cfg0["cache"]["directory"])
syms = pd.read_parquet(B + "/predictions.parquet")["symbol"].unique()
bars_full = {}
for s in syms:
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars_full[s] = df
bars = {s: df[df["日期"] <= END] for s, df in bars_full.items()}
bm = DataDownloader(cache, **(cfg0.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101",
    END.strftime("%Y%m%d")).set_index("日期")["收盘"]


def run(name, pol, ovl, sc):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"]),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ovl)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, sc, benchmark_prices=bm)
    rets = r["equity_curve"].pct_change().dropna()
    mon = (1 + rets).resample("ME").prod() - 1
    mp = rets.index.to_period("M")
    b3 = {pd.Period(x, "M") for x in mon.nlargest(3).index}
    b5 = {pd.Period(x, "M") for x in mon.nlargest(5).index}

    def sh(x):
        return float(x.mean() / x.std() * np.sqrt(252))
    ex = r["execution"]
    print("  {:34s} Sharpe {:5.2f} | 剔2024 {:5.2f} | 剔最好3月 {:5.2f} | "
          "剔最好5月 {:5.2f} | 只数 {:4.1f} 仓位 {:5.1%} 累计 {:+7.1%}".format(
              name, sh(rets), sh(rets[rets.index.year != 2024]),
              sh(rets[~mp.isin(b3)]), sh(rets[~mp.isin(b5)]),
              ex.get("policy_mean_names"), ex.get("policy_mean_gross_weight"),
              float(r["equity_curve"].iloc[-1] / CAP - 1)))


print("=" * 112)
print("step_score_floor 敏感性：0.005 = 现行绝对值；按比例缩放 = 0.005 × (该档 buy / 0.030)")
for tag, h, path, base_buy in VAR:
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= END]
    pr = p.set_index(["date", "symbol"])["prediction"]
    sc = scores_from_predictions(pr, build_tradable_mask(bars, p["date"].unique()))
    buy, strong = float(pr.quantile(Q_BUY)), float(pr.quantile(Q_STRONG))
    ovl = dc.replace(ocfg0, ic_horizon_days=h)
    print("\n[%s] buy %+.5f strong %+.5f 带宽 %.5f" % (tag, buy, strong, strong - buy))
    for lab, floor in (("floor 0.005（网格用的）", 0.005),
                       ("floor 按比例缩放", round(0.005 * buy / 0.030, 5)),
                       ("floor 0.0025", 0.0025),
                       ("floor 0.010", 0.010)):
        run("%s | %s" % (tag, lab),
            dc.replace(pol0, buy_score=round(buy, 5),
                       strong_score=round(strong, 5), step_score_floor=floor),
            ovl, sc)
