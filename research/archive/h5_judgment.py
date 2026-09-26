# -*- coding: utf-8 -*-
"""h5 判决两问（已归档）。

问题 1 尾部集中度：去掉最好的几个月，top-k 超额和组合 Sharpe 还剩多少？
问题 2 机制归因：5 日标签模型是不是短期动量代理？
  2a 分数的截面相关，以及把动量/换手/规模回归掉之后残差的 IC；
  2b 把"过去 5 日涨幅"这类裸因子直接灌进同一套带位策略 + 暴露层，看能打多少 Sharpe。

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
VAR = [
    ("orig_h20", 20, B + "/predictions.parquet"),
    ("orig_h5", 5, B + "/pred_orig_h5.parquet"),
    ("neut_h20", 20, B + "/pred_neut_h20.parquet"),
    ("neut_h10", 10, B + "/pred_neut_h10.parquet"),
]

cfg0 = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg0["backtest"], cfg0["market"]
CAP = cb.get("initial_capital", 1_000_000)

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

C = pd.DataFrame({s: df.set_index("日期")["收盘"] for s, df in bars_full.items()}).sort_index()
V = pd.DataFrame({s: df.set_index("日期")["成交量"] for s, df in bars_full.items()}).sort_index()
T = pd.DataFrame({s: df.set_index("日期")["换手率"] for s, df in bars_full.items()}).sort_index()
FWD20 = (C.shift(-20) / C - 1).stack().rename("f20").rename_axis(["date", "symbol"])

# 模型看得见的是 T-1 的信息，动量/换手/规模轴统一错一行
CAPW = (C * V / T)
with np.errstate(divide="ignore", invalid="ignore"):
    LOGCAP = np.log(CAPW.where(CAPW > 0)).shift(1)
MOM = {h: (C / C.shift(h) - 1).shift(1) for h in (5, 10, 20)}
DTURN = (T.rolling(5).mean() / T.rolling(20).mean() - 1).shift(1)


def stack(w, name):
    return w.stack().rename(name).rename_axis(["date", "symbol"])


FEAT = pd.concat([stack(MOM[5], "mom5"), stack(MOM[10], "mom10"),
                  stack(MOM[20], "mom20"), stack(DTURN, "dturn"),
                  stack(LOGCAP, "logcap")], axis=1)


def load(path, h):
    """返回 (分数面板, 原始预测宽序列)。"""
    p = pd.read_parquet(path)
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= END]
    pr = p.set_index(["date", "symbol"])["prediction"]
    sc = scores_from_predictions(pr, build_tradable_mask(bars, p["date"].unique()))
    return sc, pr


def tail_series(sc, ks=(3, 5, 12, 30, 100)):
    """每月最后评估日取分数前 k 只，算它们随后 20 日相对全池的超额。"""
    d = sc[["score"]].join(FWD20, how="inner")
    dt = pd.DatetimeIndex(d.index.get_level_values("date"))
    evs = pd.Series(dt).groupby(pd.PeriodIndex(dt, freq="M")).max()
    e = d[np.isin(dt, pd.DatetimeIndex(sorted(set(evs.values))))]
    uni = e.groupby("date")["f20"].mean()
    top = e.sort_values("score", ascending=False).groupby("date")
    return {k: top.head(k).groupby("date")["f20"].mean() - uni for k in ks}


def robust(s, name):
    n = len(s)
    tot = s.sum()
    desc = s.sort_values(ascending=False)
    d3, d5 = desc.iloc[3:], desc.iloc[5:]
    t = s.mean() / s.std() * np.sqrt(n)
    print("  {:10s} n={:3d} 均值 {:+7.3%} 中位 {:+7.3%} 胜率 {:4.0%} t={:5.2f}"
          " | 最好3月占总 {:5.0%} 去最好3月 {:+7.3%} 去最好5月 {:+7.3%}".format(
              name, n, s.mean(), s.median(), (s > 0).mean(), t,
              desc.iloc[:3].sum() / tot, d3.mean(), d5.mean()))


def stat(rets, label):
    cum = (1 + rets).prod() - 1
    sh = rets.mean() / rets.std() * np.sqrt(252)
    nav = (1 + rets).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    return "{} 累计 {:+7.1%} Sharpe {:5.2f} 回撤 {:+7.2%}".format(label, cum, sh, dd)


def run(name, pol, ovl, sc, freq="monthly"):
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency=freq,
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
    r24 = rets[rets.index.year != 2024]
    b3 = set(mon.nlargest(3).index)
    b5 = set(mon.nlargest(5).index)
    p3 = {pd.Period(x, "M") for x in b3}
    p5 = {pd.Period(x, "M") for x in b5}
    ex = r["execution"]
    print("  {:20s} {} | {} | {} | {}".format(
        name, stat(rets, "全样本"), stat(r24, "剔2024"),
        stat(rets[~mp.isin(p3)], "剔最好3月"),
        stat(rets[~mp.isin(p5)], "剔最好5月")))
    print("  {:20s} 只数 {:.1f} 仓位 {:.1%} 成本 {:.2%} 最好3月 {}".format(
        "", ex.get("policy_mean_names"), ex.get("policy_mean_gross_weight"),
        r["trades"]["cost"].sum() / CAP,
        " ".join("{:+.1%}".format(mon[x]) for x in sorted(b3))))
    return r


def cs_stats(sc, name):
    """分数的截面相关，以及把动量/换手/规模回归掉之后残差的 IC。"""
    d = sc[["score"]].join(FWD20, how="inner").join(FEAT, how="inner").dropna()
    cols = ["score", "mom5", "mom10", "mom20", "dturn", "logcap", "f20"]
    X = ["mom5", "mom20", "dturn", "logcap"]
    rk = d.groupby(level="date").rank(pct=True)
    full, resid = [], []
    corrs = {c: [] for c in cols[1:]}
    for _, x in rk.groupby(level="date"):
        s0, f0 = x["score"].values, x["f20"].values
        full.append(np.corrcoef(s0, f0)[0, 1])
        for c in cols[1:]:
            corrs[c].append(np.corrcoef(s0, x[c].values)[0, 1])
        A = np.column_stack([np.ones(len(x))] + [x[c].values for c in X])
        beta = np.linalg.lstsq(A, s0, rcond=None)[0]
        resid.append(np.corrcoef(s0 - A @ beta, f0)[0, 1])
    fi, ri = float(np.nanmean(full)), float(np.nanmean(resid))
    print("  {:16s} IC@20 {:+.4f} → 剔掉动量/换手/规模后残差 IC {:+.4f} "
          "(保留 {:.0%})".format(name, fi, ri, ri / fi))
    print("  {:16s} 与 score 的截面相关: ".format("") + "  ".join(
        "{} {:+.3f}".format(c, np.nanmean(corrs[c])) for c in cols[1:]))
    return fi, ri


print("=" * 112)
print("问题 1a 名单顶端的 20 日超额，以及去掉最好的几个月还剩多少")
STORE = {}
for tag, h, path in VAR:
    sc, pr = load(path, h)
    STORE[tag] = (sc, pr)
    print("\n[%s] 标签=%d日" % (tag, h))
    for k, s in tail_series(sc).items():
        robust(s, "top%d" % k)

print("\n" + "=" * 112)
print("问题 2a 分数是不是动量 / 换手 / 规模的代理")
for tag, h, path in VAR:
    cs_stats(STORE[tag][0], tag)

print("\n" + "=" * 112)
print("问题 2b 裸因子直接当分数，同一套带位策略 + 暴露层（阈值同样按分位对齐）")
pol0 = policy_from_config(cfg0)
ocfg0 = overlay_from_config(cfg0)
ref_idx = STORE["orig_h5"][1].index
for nm, wide, hh in (("过去5日涨幅", MOM[5], 5),
                     ("过去20日涨幅", MOM[20], 20),
                     ("过去5日跌幅(反转)", -MOM[5], 5)):
    s = stack(wide, "prediction").reindex(ref_idx).dropna()
    sc = scores_from_predictions(s, build_tradable_mask(
        bars, s.index.get_level_values("date").unique()))
    buy, strong = float(s.quantile(Q_BUY)), float(s.quantile(Q_STRONG))
    print("\n[%s] 分数 P50 %+.4f → buy %+.4f strong %+.4f"
          % (nm, float(s.median()), buy, strong))
    cs_stats(sc, nm)
    run(nm, dc.replace(pol0, buy_score=round(buy, 5),
                       strong_score=round(strong, 5)),
        dc.replace(ocfg0, ic_horizon_days=hh), sc)

print("\n" + "=" * 112)
print("问题 1b 组合层同样的稳健性检验（月度评估，剔除贡献最大的月份）")
for tag, h, path in VAR:
    sc, pr = STORE[tag]
    buy, strong = float(pr.quantile(Q_BUY)), float(pr.quantile(Q_STRONG))
    run(tag, dc.replace(pol0, buy_score=round(buy, 5),
                        strong_score=round(strong, 5)),
        dc.replace(ocfg0, ic_horizon_days=h), sc)

