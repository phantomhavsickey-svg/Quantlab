# -*- coding: utf-8 -*-
"""两臂"交集"组合回测：每个调仓日只买两边分数都过建仓线的股。

组合分 = min(两臂分数)。min(a,b) >= 建仓线 等价于 a、b 都 >= 建仓线，所以
买入候选正好是两臂过线名单的交集；卖出同理——任一边掉到清仓线以下，组合分也在线下。

    python research/seed_intersect.py                          # 脚本里写死的 6 对
    python research/seed_intersect.py old 42 7                 # 一对
    python research/seed_intersect.py old 42 3 new 1 0         # 多对

回测支架与 panel_arms.py / seed_arms.py 完全一致（bars 池 = 该面板自己的 symbol 集、
费率/暴露层/政策全走 config.yaml），单臂数字可直接对上那两份日志。
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
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

PANELS = {"old": ("data/cache/backup_20260930/factor_panel.parquet", "backup_20260930"),
          "new": ("data/cache/factor_panel.parquet", "cache")}
DEFAULT_PAIRS = [("old", 42, 7), ("old", 42, 3), ("old", 0, 5),
                 ("new", 42, 7), ("new", 1, 0), ("new", 42, 13)]

args = sys.argv[1:]
if not args:
    PAIRS = DEFAULT_PAIRS
elif len(args) % 3 == 0:
    PAIRS = [(args[i], int(args[i + 1]), int(args[i + 2])) for i in range(0, len(args), 3)]
else:
    raise SystemExit(__doc__)

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = float(cb["initial_capital"])
pm = PerformanceMetrics()
cache = CacheManager(cfg["cache"]["directory"])


def find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            r = find_key(v, key)
            if r is not None:
                return r
    return None


LINE = float(find_key(cfg, "buy_score"))


def pred_path(tag, seed):
    cands = []
    if tag == "old":
        cands.append("data/cache/_probe_backup_20260930_seed%d" % seed)
    else:
        cands.append("data/cache/_probe_seed%d" % seed)
        cands.append("data/cache/_probe_cache_seed%d" % seed)
    for c in cands:
        p = os.path.join(c, "predictions.parquet")
        if os.path.exists(p):
            return p
    if tag == "new" and seed == 42:
        return "data/cache/predictions.parquet"
    raise FileNotFoundError("seed %d (%s) 的预测找不到" % (seed, tag))


_loaded = {}


def load_all(tag):
    if tag in _loaded:
        return _loaded[tag]
    panel_file, short = PANELS[tag]
    panel = pd.read_parquet(panel_file, columns=["symbol"])
    print("### 载入 %s 面板（%d 只）与日线 bars ..." % (tag, panel["symbol"].nunique()), flush=True)
    bars = {}
    for s in sorted(set(panel["symbol"])):
        df = cache.get_daily(s)
        if df is not None:
            df["日期"] = pd.to_datetime(df["日期"])
            bars[s] = df.sort_values("日期")
    try:
        BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
            cb.get("benchmark", "000852"), "20210101",
            max(d["日期"].max() for d in bars.values()).strftime("%Y%m%d")).set_index("日期")["收盘"]
    except Exception as e:
        print("  基准取不到(%s)" % e, flush=True)
        BM = None
    _loaded[tag] = (bars, BM, short)
    return _loaded[tag]


def load_pred(path):
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", "symbol", "prediction"]]


def backtest(pred_df, bars, BM, tag):
    PR = pred_df.set_index(["date", "symbol"])["prediction"]
    SC = scores_from_predictions(PR, build_tradable_mask(bars, pred_df["date"].unique()))
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
        r = eng.run(bars, SC, benchmark_prices=BM)
    eq, tr = r["equity_curve"], r["trades"]
    yr = eq.groupby(eq.index.year).agg(["first", "last"])
    print("  %-14s 累计 %+8.2f%%  笔数 %4d  成本 %7.0f 元  平均目标仓位 %.1f%%"
          % (tag, pm.cumulative_return(eq) * 100, len(tr), tr["cost"].sum(),
             (r["execution"].get("policy_mean_gross_weight") or 0) * 100), flush=True)
    print("                 分年 " + "  ".join("%d %+.1f%%" % (y, (l / f - 1) * 100)
                                               for y, (f, l) in yr.iterrows()), flush=True)
    return eq, tr


_arm_cache = {}


def arm(tag, seed):
    key = (tag, seed)
    if key not in _arm_cache:
        bars, BM, _ = load_all(tag)
        _arm_cache[key] = backtest(load_pred(pred_path(tag, seed)), bars, BM,
                                   "%s seed%d" % (tag, seed))
    return _arm_cache[key]


print("### 建仓线 %.5f（诊断用途：数过线只数）" % LINE, flush=True)
for tag, sa, sb in PAIRS:
    bars, BM, short = load_all(tag)
    da, db = load_pred(pred_path(tag, sa)), load_pred(pred_path(tag, sb))
    m = da.merge(db, on=["date", "symbol"], suffixes=("_a", "_b"))
    m["prediction"] = np.minimum(m["prediction_a"], m["prediction_b"])
    comb = m[["date", "symbol", "prediction"]]
    outdir = "data/cache/_probe_intersect_%s_s%d_s%d" % (short, sa, sb)
    os.makedirs(outdir, exist_ok=True)
    comb.to_parquet(os.path.join(outdir, "predictions.parquet"), index=False)
    last = m.groupby(m["date"].dt.to_period("M"))["date"].max()
    rows = []
    for d0 in last:
        x = m[m["date"] == d0]
        rows.append((len(x),
                     int((x["prediction_a"] >= LINE).sum()),
                     int((x["prediction_b"] >= LINE).sum()),
                     int((x["prediction"] >= LINE).sum())))
    r = pd.DataFrame(rows, columns=["n", "ca", "cb", "ci"])
    print()
    print("### %s 交集档 seed%d ∩ seed%d | 预测 %d 条 -> %s" % (tag, sa, sb, len(comb), outdir), flush=True)
    print("  调仓日 %d 个 | 平均可配 %.0f 只 | 过建仓线平均：seed%d %.1f 只、seed%d %.1f 只、交集 %.1f 只 | 交集=0 的调仓日 %d 个"
          % (len(r), r["n"].mean(), sa, r["ca"].mean(), sb, r["cb"].mean(), r["ci"].mean(),
             int((r["ci"] == 0).sum())), flush=True)
    arm(tag, sa)
    arm(tag, sb)
    backtest(comb, bars, BM, "交集(%d,%d)" % (sa, sb))
    print(flush=True)
