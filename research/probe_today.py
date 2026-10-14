# -*- coding: utf-8 -*-
"""只读探测：某个区间里，数据源到底给没给出 bar（含"今天收盘了没有"）。

    python research/probe_today.py [START END [抽样只数]]
    例：python research/probe_today.py 20261008 20261009 20

不写缓存、不写面板、不写预测：DataDownloader 的第一个参数传 None，
所以 CacheManager 不存在，_fetch_tx_direct 只返回 DataFrame。
帧的列名是英文 date/open/close（data/downloader.py:100），
指数那条走 download_index_daily，列名是中文，两边分开处理。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import yaml
from data.cache import CacheManager
from data.downloader import DataDownloader

argv = sys.argv[1:]
START = argv[0] if len(argv) > 0 else "20261008"
END = argv[1] if len(argv) > 1 else pd.Timestamp.today().strftime("%Y%m%d")
NS = int(argv[2]) if len(argv) > 2 else 20

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
dl = DataDownloader(None, **(cfg.get("download") or {}))
cm = CacheManager(cfg["cache"]["directory"])

end_ts = pd.Timestamp(END)
syms = cm.list_cached_symbols()
step = max(1, len(syms) // NS)
sample = syms[::step][:NS]

print("探测区间 %s ~ %s，抽样 %d 只（缓存共 %d 只，每隔 %d 只取 1 只）"
      % (START, END, len(sample), len(syms), step))
print("只读，不写任何文件")
print()
hit = 0
for sym in sample:
    d = dl._fetch_tx_direct(sym, START, END, "qfq")
    if d is None or len(d) == 0:
        print("  %s 返回 0 行" % sym)
        continue
    idx = pd.to_datetime(d["date"])
    last = idx.max()
    has = last == end_ts
    hit += int(has)
    print("  %s 返回 %d 行，最后一根 %s，收盘 %.2f %s"
          % (sym, len(d), last.date(), float(d["close"].iloc[-1]),
             "<- 有 %s 这天" % END if has else ""))
print()
print("抽样 %d 只里，数据源已给出 %s 这根 bar 的有 %d 只（%.0f%%）"
      % (len(sample), END, hit, hit / max(1, len(sample)) * 100))

b = dl.download_index_daily("000852", START, END)
if b is not None and len(b):
    bidx = pd.to_datetime(b["日期"] if "日期" in b.columns else b["date"])
    print("指数 000852 返回 %d 行，最后一根 %s" % (len(b), bidx.max().date()))
