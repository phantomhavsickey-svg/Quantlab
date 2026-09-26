# -*- coding: utf-8 -*-
"""幸存者偏差量级探测：把"样本期内退市的股票"请回来看一眼收益。

    python research/survivorship_probe.py

现在这台机器的股票池是**今天还在中证1000里的 1000 只**（`data/downloader.py` 用的
是中证指数公司当前成分股接口），所以整段回测里没有一只股票会退市、会消失。真实历史
里 2023~2026 有一批小票跌到退市——它们不在数据里，策略就没在它们身上亏过钱。

本脚本不重训模型、不改进生产代码，只回答一个问题：**这批缺席的股票，如果放在池子里，
收益比留下来的股票低多少？** 拿到的差值是偏差的**下界**，因为更大的一块——"当年在中证1000
里、后来被调出指数（通常因为跌了）"——免费接口拿不到名单，只有交易所的退市/终止名单能拿。

行情写到 `data/cache_delisted/`，不碰研究用的 `data/cache/`。
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import yaml

from data.cache import CacheManager
from data.downloader import DataDownloader

WIN_START = pd.Timestamp("2023-01-05")   # predictions.parquet 的第一天
WIN_END = pd.Timestamp("2026-09-15")     # predictions.parquet 的最后一天
CACHE_DEAD = os.path.join("data", "cache_delisted")
POOL_CACHE = os.path.join("data", "cache")

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))


def dead_names() -> pd.DataFrame:
    """两个交易所的退市名单，只要窗口起点之后才停牌的。"""
    import akshare as ak
    rows = []
    sh = ak.stock_info_sh_delist()
    sh["停牌日"] = pd.to_datetime(sh["暂停上市日期"], errors="coerce")
    for _, r in sh.iterrows():
        rows.append({"symbol": str(r["公司代码"]).zfill(6), "name": r["公司简称"],
                     "listed": pd.to_datetime(r["上市日期"], errors="coerce"),
                     "dead": r["停牌日"]})
    sz = ak.stock_info_sz_delist(symbol="终止上市公司")
    sz["终止上市日期"] = pd.to_datetime(sz["终止上市日期"], errors="coerce")
    for _, r in sz.iterrows():
        rows.append({"symbol": str(r["证券代码"]).zfill(6), "name": r["证券简称"],
                     "listed": pd.to_datetime(r["上市日期"], errors="coerce"),
                     "dead": r["终止上市日期"]})
    df = pd.DataFrame(rows).drop_duplicates(subset="symbol", keep="first")
    keep = df[(df["dead"] > WIN_START) & (df["listed"] < WIN_START)]
    return keep.sort_values("dead").reset_index(drop=True)


def span_return(df: pd.DataFrame) -> tuple:
    """窗口内等权买入持有收益：起点取窗口首日(或该股在窗口内的第一根 bar)。"""
    d = df.copy()
    d["日期"] = pd.to_datetime(d["日期"])
    d = d[(d["日期"] >= WIN_START) & (d["日期"] <= WIN_END)].dropna(subset=["收盘"])
    if len(d) < 20:                       # 不足 20 根 bar 不算，避免最后一两天的噪声
        return None
    r0 = float(d["收盘"].iloc[0])
    r1 = float(d["收盘"].iloc[-1])
    if not r0 or not r1:
        return None
    return (r1 / r0 - 1.0, len(d), d["日期"].iloc[-1])


def main():
    t0 = time.time()
    want = dead_names()
    print(f"### 交易所退市名单里，回测起点(2023-01-05)之前已上市、之后才退市的：{len(want)} 只")
    print(f"    退市时间分布：{want['dead'].min().date()} ~ {want['dead'].max().date()}")

    dl = DataDownloader(CacheManager(CACHE_DEAD), **(cfg.get("download") or {}))
    got, empty = [], []
    for i, r in enumerate(want.itertuples(), 1):
        df = dl.download_daily_ohlcv(r.symbol, "20230101", "20260920", "qfq")
        if df is None or df.empty:
            empty.append(r.symbol)
        else:
            s = span_return(df)
            if s is None:
                empty.append(r.symbol)
            else:
                got.append({"symbol": r.symbol, "name": r.name, "dead": r.dead,
                            "ret": s[0], "bars": s[1], "last": s[2]})
        if i % 40 == 0:
            print(f"    ...{i}/{len(want)}，拿到 {len(got)} 只，耗时 {time.time()-t0:.0f}s")

    g = pd.DataFrame(got)
    print(f"\n### 1. 缺席的那批（能取到行情的 {len(g)} 只，取不到 {len(empty)} 只）")
    if len(g):
        print(f"  窗口内买入持有收益：均值 {g.ret.mean():+.1%}  中位 {g.ret.median():+.1%}"
              f"  为正的比例 {(g.ret > 0).mean():.0%}")
        print(f"  按退市年份分组（中位收益 / 只数）：")
        for y, sub in g.groupby(g.dead.dt.year):
            print(f"    {y}  {sub.ret.median():+.1%} / {len(sub)} 只")

    live_pool = CacheManager(POOL_CACHE)
    syms = sorted(pd.read_parquet("data/cache/factor_panel.parquet")["symbol"].unique())
    cols, amt = {}, {}
    for s in syms:
        df = live_pool.get_daily(s)
        if df is not None and len(df):
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            cols[s] = df.set_index("日期")["收盘"]
            amt[s] = df.set_index("日期")["成交额"]
    pool = pd.DataFrame(cols).sort_index().loc[WIN_START:WIN_END]
    pool_amt = pd.DataFrame(amt).sort_index().loc[WIN_START:WIN_END]
    live = pool.ffill()
    pr = (live.iloc[-1] / live.iloc[0] - 1).dropna()
    pool_turn = pool_amt.median().dropna()
    print(f"\n### 2. 留下来的一千只（同一窗口、同一算法）")
    print(f"  等权买入持有收益：均值 {pr.mean():+.1%}  中位 {pr.median():+.1%}"
          f"  为正的比例 {(pr > 0).mean():.0%}")
    print(f"  日成交额中位数：组内中位 {pool_turn.median()/1e8:.2f} 亿"
          f"  P25 {pool_turn.quantile(.25)/1e8:.2f} 亿")
    if len(g):
        yrs = (WIN_END - WIN_START).days / 365.25
        a_live = (1 + pr.mean()) ** (1 / yrs) - 1
        print(f"\n### 3. 差值（幸存者偏差的下界）")
        print(f"  留下来的中位 {pr.median():+.1%} − 退市中位 {g.ret.median():+.1%} "
              f"= {pr.median() - g.ret.median():+.1%}"
              f"（均值口径 {pr.mean() - g.ret.mean():+.1%}）")
        n = len(g)
        tot = len(pr) + n
        mix = (pr.sum() + g.ret.sum()) / tot
        a_mix = (1 + mix) ** (1 / yrs) - 1
        print(f"  若把这 {n} 只放回池子做等权：等权均值从 {pr.mean():+.1%} 变成 {mix:+.1%}"
              f"（窗口 {yrs:.2f} 年 → 折年化 {a_live:+.1%} → {a_mix:+.1%}，"
              f"差 {(a_live - a_mix) * 100:.1f}pp/年）")

        # 这批名字本来能不能被本策略买到：中位单笔下单 9.7 万元（README「交易成本口径」）
        dead_amt = {}
        dead_pool = CacheManager(CACHE_DEAD)
        for r in g.itertuples():
            df = dead_pool.get_daily(r.symbol)
            if df is None or not len(df):
                continue
            d = df.copy()
            d["日期"] = pd.to_datetime(d["日期"])
            x = d.set_index("日期")["成交额"].loc[WIN_START:WIN_END].dropna()
            if len(x) >= 20:
                dead_amt[r.symbol] = x.median()
        da = pd.Series(dead_amt)
        q25 = pool_turn.quantile(.25)
        print(f"\n### 4. 流动性对照（能不能真的买到）")
        print(f"  退市组日成交额中位数 {da.median()/1e8:.2f} 亿 vs 在册池 {pool_turn.median()/1e8:.2f} 亿"
              f"（{len(da)} 只有数据）")
        print(f"  退市组里低于在册池 P25({q25/1e8:.2f} 亿)的占 {(da < q25).mean():.0%}"
              f"，低于中位单笔下单额 9.7 万元的 {(da < 97000).mean():.0%}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
