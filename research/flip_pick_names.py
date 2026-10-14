# -*- coding: utf-8 -*-
"""三臂卖出已实现前若干名 + 两臂共买名字,带中文名与板块。

    python research/flip_pick_names.py

名单来自 research/flip_arms.py 落盘的成交记录(data/cache/_probe_flip/*_trades.parquet)。
中文名走 data/downloader.py 的指数成分股接口(000852 中证1000),缺的用全市场列表补。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
logger.remove()

import pandas as pd
import yaml

from data.cache import CacheManager
from data.downloader import DataDownloader

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cache = CacheManager(cfg["cache"]["directory"])
ARMS = {"0 现行": "0", "A 模型取反": "A", "B 因子取反": "B"}


def board(code):
    if code.startswith(("688", "689")):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith("60"):
        return "沪主板"
    if code.startswith("00"):
        return "深主板"
    return "其他"


rows = {}
for tag, a in ARMS.items():
    tr = pd.read_parquet("data/cache/_probe_flip/%s_trades.parquet" % a)
    tr["symbol"] = tr["symbol"].astype(str).str.zfill(6)
    buy = tr[tr["side"] == "buy"].groupby("symbol").agg(
        n_buy=("shares", "size"), amt=("amount", "sum"))
    sell = tr[tr["side"] == "sell"].groupby("symbol")["realized_pnl"].sum()
    df = buy.join(sell, how="left").fillna({"realized_pnl": 0.0})
    df["board"] = [board(s) for s in df.index]
    rows[tag] = df.sort_values("realized_pnl", ascending=False)

need = sorted(set().union(*[set(r.head(12).index) for r in rows.values()])
              | set(rows["0 现行"].index) & set(rows["B 因子取反"].index))
names = {}
try:
    cons = DataDownloader(cache, **(cfg.get("download") or {})).download_index_constituents("000852")
    names.update(dict(zip(cons["symbol"].astype(str).str.zfill(6), cons["name"])))
except Exception as e:
    print("成分股名称接口失败: %s" % e)
if set(need) - set(names):
    try:
        import akshare as ak
        full = ak.stock_info_a_code_name()
        names.update(dict(zip(full["code"].astype(str).str.zfill(6), full["name"])))
    except Exception as e:
        print("全市场名称接口失败: %s" % e)

print("| 臂 | 代码 | 名称 | 板块 | 买入笔数 | 买入金额(元) | 已实现卖利(元) |")
print("|---|---|---|---|---|---|---|")
for tag, top in [("0 现行", 10), ("B 因子取反", 10), ("A 模型取反", 5)]:
    for s, r in rows[tag].head(top).iterrows():
        print("| %s | %s | %s | %s | %d | %s | %s |"
              % (tag, s, names.get(s, "名称未取到"), r["board"], r["n_buy"],
                 format(int(r["amt"]), ","), format(int(r["realized_pnl"]), ",")))

shared = sorted(set(rows["0 现行"].index) & set(rows["B 因子取反"].index))
print("| 0 与 B 共买 | | 共 %d 只 | | | | |" % len(shared))
for s in shared:
    r0, rb = rows["0 现行"].loc[s], rows["B 因子取反"].loc[s]
    print("| 共买-%s | %s | %s | %s | 0:%d 笔/B:%d 笔 | 0:%s / B:%s | 0:%s / B:%s |"
          % (r0["board"], s, names.get(s, "名称未取到"), r0["board"], r0["n_buy"], rb["n_buy"],
             format(int(r0["amt"]), ","), format(int(rb["amt"]), ","),
             format(int(r0["realized_pnl"]), ","), format(int(rb["realized_pnl"]), ",")))
