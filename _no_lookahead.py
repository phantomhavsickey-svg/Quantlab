# -*- coding: utf-8 -*-
"""前视审计：切点之后的数据能不能改变切点之前的结果。

三问：(1) 引擎撮合与盯市；(2) 技术因子；(3) walk-forward 训练标签跨窗口边界。
对照做法 = 把 T 之后的价格砍掉 / 打到五折，看 T 之前的净值与成交流水是否逐格不变。
临时脚本。
"""
import contextlib
import io
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from data.cache import CacheManager
from data.downloader import DataDownloader
from factors.technical import TechnicalFactors
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

T = pd.Timestamp("2025-06-30")
PCOLS = ["开盘", "收盘", "最高", "最低"]

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
CAP = cb.get("initial_capital", 1_000_000)
pol = policy_from_config(cfg)
ocfg = overlay_from_config(cfg)

cache = CacheManager(cfg["cache"]["directory"])
pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
PR = pred.set_index(["date", "symbol"])["prediction"]

BM = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
    cb.get("benchmark", "000852"), "20210101", "20260825").set_index("日期")["收盘"]

bars_full = {}
for s in pred["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars_full[s] = df.sort_values("日期")


def run(bars, tag):
    sc = scores_from_predictions(PR, build_tradable_mask(bars, pred["date"].unique()))
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency="monthly",
                         max_positions=cb["max_positions"],
                         cost_model=TransactionCostModel(
                             cm["commission_rate"], cm["min_commission"],
                             cm["stamp_tax_rate"], cm["slippage_rate"]),
                         lot_size=int(cm.get("lot_size", 100)),
                         policy=pol, overlay=ocfg)
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.run(bars, sc, benchmark_prices=BM)
    print("  %-16s 成交 %3d 笔 | 净值 %4d 天 | 期末净值 %.6f×初始"
          % (tag, len(r["trades"]), len(r["marks"]),
             float(r["marks"]["total_value"].iloc[-1] / CAP)))
    return r


def flow_key(tr, hi):
    x = tr[tr["date"] < hi]
    return set(zip(x["date"].dt.strftime("%Y-%m-%d"), x["symbol"], x["side"],
                   x["shares"].astype(int), x["price"].round(6)))


def cmp(other, name):
    ma, mb = A["marks"], other["marks"]
    idx = ma.index[ma.index < T].intersection(mb.index[mb.index < T])
    dnav = float(np.max(np.abs(ma.loc[idx, "total_value"].to_numpy()
                               - mb.loc[idx, "total_value"].to_numpy())))
    dflow = len(flow_key(A["trades"], T) ^ flow_key(other["trades"], T))
    tail = "末端不同(检验有功率)" if abs(
        float(ma["total_value"].iloc[-1]) - float(mb["total_value"].iloc[-1])) > 1 \
        else "⚠ 末端相同,扰动没生效"
    print("  %s: %s 之前净值最大差 %.2e 元 | 成交流水差 %d 笔 | %s"
          % (name, T.date(), dnav, dflow, tail))


print("=" * 78)
print("§1 引擎：%s 之后的价格砍掉/减半，之前的一切必须逐格不变" % T.date())
A = run(bars_full, "A 全样本")
half = {s: df.assign(**{c: np.where(df["日期"] > T, df[c] * 0.5, df[c])
                        for c in PCOLS if c in df.columns})
        for s, df in bars_full.items()}
cmp(run(half, "B T 后 ×0.5"), "A vs B")
cmp(run({s: df[df["日期"] <= T] for s, df in bars_full.items()},
        "C T 后砍掉"), "A vs C")

print("=" * 78)
print("§2 技术因子：只用 %s 之前的 K 线重算，之前的因子值必须不变" % T.date())
periods = {"momentum_periods": cfg["factors"]["technical"]["momentum_periods"],
           "volatility_periods": cfg["factors"]["technical"]["volatility_periods"],
           "volume_ratio_periods": cfg["factors"]["technical"]["volume_ratio_periods"]}
random.seed(7)
elig = [s for s, df in bars_full.items() if int((df["日期"] < T).sum()) >= 120]
sample = random.sample(elig, 60)
cells = bad = 0
for s in sample:
    df = bars_full[s]
    pre = (df["日期"] < T).to_numpy()
    x = TechnicalFactors.compute_all(df, periods).to_numpy(dtype=float)[pre]
    y = TechnicalFactors.compute_all(df[df["日期"] < T], periods)\
        .to_numpy(dtype=float)      # 砍尾后本身就只剩切点之前
    assert x.shape == y.shape, (s, x.shape, y.shape)
    bad += int(np.sum(~((x == y) | (np.isnan(x) & np.isnan(y)))))
    cells += x.size
print("  抽样 %d/%d 只(切点前 ≥120 根 K 线) × %s 单元格:不一致 %d 格"
      % (len(sample), len(elig), f"{cells:,}", bad))

print("=" * 78)
print("§3 walk-forward:训练行 date < 窗口起点,但标签要走到 date+h → 边界重叠")
ds = np.array(sorted(pd.read_parquet("data/cache/factor_panel.parquet")["date"].unique()))

sig = pd.DatetimeIndex(sorted(pred["date"].unique()))
sig = sig.to_series().groupby(pd.Grouper(freq="ME")).max().dropna()
for h in (5, 20):
    inside, nwin = [], 0
    ws = pd.Timestamp(ds[0]) + pd.DateOffset(months=24)     # 与 cmd_train 同一口径
    last = pd.Timestamp(ds[-1]) - pd.Timedelta(days=1)      # 标签未走完的尾部不预测
    while ws <= last:
        p = int(np.searchsorted(ds, np.datetime64(ws)))
        zone_end = pd.Timestamp(ds[min(p + h - 1, len(ds) - 1)])
        inside += list(sig[(sig >= ws) & (sig <= zone_end)])
        nwin += 1
        ws = ws + pd.DateOffset(months=6)
    print("  horizon=%2d: %d 个窗口 × 边界 %2d 个交易日 → 44 个月度信号日里落在"
          "重叠区的 %d 个" % (h, nwin, h, len(set(inside))))

print("=" * 78)
print("§4 已实现盈亏口径：与净值变动对账（差 = 期末浮动盈亏，应当只是小尾巴）")
tr = A["trades"]
sells, buys = tr[tr["side"] == "sell"], tr[tr["side"] == "buy"]
realized = float(sells["realized_pnl"].sum())
last = A["marks"].iloc[-1]
dnav = float(last["total_value"]) - CAP
unreal = dnav - realized
print("  Σ已实现 %+.0f 元 | 净值变动 %+.0f 元 | 倒挤浮动 %+.0f 元 | 期末市值 %.0f 元"
      % (realized, dnav, unreal, float(last["market_value"])))
print("  每笔卖出已实现盈亏: 中位 %+.0f 元 | 最好 %+.0f | 最差 %+.0f | 卖出手续费合计 %.0f"
      % (sells["realized_pnl"].median(), sells["realized_pnl"].max(),
         sells["realized_pnl"].min(), float(sells["cost"].sum())))
print("  对照旧口径(卖额−卖出费,不扣买入成本): Σ=%.0f 元 = %.1f×净值变动"
      % (float(sells["net_proceeds"].sum()),
         float(sells["net_proceeds"].sum()) / dnav))


