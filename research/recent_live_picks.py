"""最近若干个交易日的实盘口径分数名单（只读，不写生产文件）。

与 live/run_daily.py 走同一份模型文件和同一套因子预处理，区别只是这里
把截面从"最新一天"扩展到最近 N 天，用来回答"这几天模型选的是哪些股"。

用法: python research/recent_live_picks.py [天数, 默认 5]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from live.daily_signal import DailySignalGenerator
from utils.market_rules import build_tradable_mask

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
POL = CFG["position_policy"]
BUY, STRONG, SELL = (float(POL["buy_score"]), float(POL["strong_score"]),
                     float(POL["sell_score"]))
ADD = float(POL.get("add_score_line", 0.0) or 0.0)
NAMES = int(POL["max_names"])
LOOKBACK = int(CFG["live"]["lookback_days"])
ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 5

gen = DailySignalGenerator(CFG)
print("模型文件: %s | 特征 %d 个" % (
    os.path.join(CFG["model"].get("save_dir", "models/saved"),
                 sorted(f for f in os.listdir(
                     CFG["model"].get("save_dir", "models/saved"))
                        if f.startswith("lgb_"))[-1]),
    len(gen.trainer.feature_names)))

daily = gen._load_daily("2026-09-30", LOOKBACK)
bar_last = max(df["日期"].max() for df in daily.values())
print("日线缓存最后一天: %s | 取数股票数: %d" % (
    pd.Timestamp(bar_last).date(), len(daily)))
processed = gen.build_processed_panel("2026-09-30", LOOKBACK, daily)
fnames = list(gen.trainer.feature_names)
dates = sorted(processed["date"].unique())[-ndays:]

print("三条线: 清仓 %.5f | 建仓 %.5f | 强档/补仓 %.5f" % (SELL, BUY,
      ADD if ADD > 0 else STRONG))
print()
for d in dates:
    dt = pd.Timestamp(d)
    X = processed[processed["date"] == d].set_index(["date", "symbol"])[fnames]
    if X.empty:
        continue
    s = pd.Series(gen.trainer.model.predict(X.fillna(np.nan)), index=X.index,
                  name="score")
    tr = build_tradable_mask(daily, [d])
    kept = s[s.index.isin(tr.index)] if len(tr) else s
    over = kept[kept >= BUY].sort_values(ascending=False)
    print("  %s  有分 %d 只(可交易 %d) | ≥建仓线 %2d 只 | ≥强档线 %2d 只 | 最高 %.5f"
          % (dt.date(), len(s), len(kept), len(over),
             int((kept >= STRONG).sum()),
             float(kept.max()) if len(kept) else float("nan")))
    for rank, (idx, sc) in enumerate(over.head(max(NAMES, 12)).items(), start=1):
        sym = idx[1] if isinstance(idx, tuple) else idx
        print("      #%02d %s %.5f %s" % (rank, sym, sc,
                                          "满仓档" if sc >= STRONG else ""))
    if len(over) > max(NAMES, 12):
        print("      （过线 %d 只，只列前 %d）" % (len(over), max(NAMES, 12)))
