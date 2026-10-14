"""实盘口径分数 与 predictions.parquet 的逐日对账（只读）。

recent_live_picks.py 用 models/saved 里最新那份模型现算分数；predictions.parquet
是 Walk-Forward 每一折在"训练截止之后"的那段样本上的预测（用的模型各不相同）。
两者在同一天不该天然相等，这里量化差多少，判断名单能不能作为实盘口径读。

用法: python research/live_vs_preds.py [对账日, 默认取两者共有的最后一天]
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
BUY = float(POL["buy_score"])
LOOKBACK = int(CFG["live"]["lookback_days"])
ASOF = sys.argv[1] if len(sys.argv) > 1 else None

gen = DailySignalGenerator(CFG)
preds = pd.read_parquet("data/cache/predictions.parquet")
preds["date"] = pd.to_datetime(preds["date"])

daily = gen._load_daily(ASOF or "2026-09-30", LOOKBACK)
processed = gen.build_processed_panel(ASOF or "2026-09-30", LOOKBACK, daily)
fnames = list(gen.trainer.feature_names)

live_dates = set(pd.Timestamp(d).date() for d in processed["date"].unique())
pred_dates = set(preds["date"].dt.date.unique())
common = sorted(live_dates & pred_dates)
print("实盘可算日期区间: %s .. %s (%d 天)" % (
    min(live_dates), max(live_dates), len(live_dates)))
print("predictions 区间: %s .. %s (%d 天)" % (
    min(pred_dates), max(pred_dates), len(pred_dates)))
print("两者共有: %d 天，最后 5 天 = %s" % (len(common), [str(x) for x in common[-5:]]))
print()

target = pd.Timestamp(ASOF).date() if ASOF else (common[-1] if common else None)
if target is None:
    print("没有可对账的日期")
    sys.exit(0)

d = pd.Timestamp(target)
X = processed[processed["date"] == d].set_index(["date", "symbol"])[fnames]
live = pd.Series(gen.trainer.model.predict(X.fillna(np.nan)), index=X.index, name="live")
tr = build_tradable_mask(daily, [d])
if len(tr):
    live = live[live.index.isin(tr.index)]

pr = preds[preds["date"] == d].set_index("symbol")["prediction"]
lv = live.reset_index()
joint = lv.set_index("symbol")["live"].to_frame().join(pr.rename("preds"), how="inner")
print("对账日 %s | 交集 %d 只（实盘口径 %d，predictions %d）" % (
    target, len(joint), len(live), len(pr)))
if len(joint) < 3:
    sys.exit(0)

sp = joint["live"].rank(method="average").corr(joint["preds"].rank(method="average"))
pe = float(np.corrcoef(joint["live"], joint["preds"])[0, 1])
print("  Spearman(等级相关) = %.4f | Pearson(线性相关) = %.4f" % (sp, pe))
print("  实盘口径 中位数 %.5f / 最高 %.5f | predictions 中位数 %.5f / 最高 %.5f" % (
    joint["live"].median(), joint["live"].max(),
    joint["preds"].median(), joint["preds"].max()))
n_live = int((live >= BUY).sum())
n_pred = int((pr >= BUY).sum())
top_n = int(POL["max_names"])
tl = set(live.sort_values(ascending=False).head(top_n).reset_index()["symbol"])
tp = set(pr.sort_values(ascending=False).head(top_n).index)
print("  ≥建仓线只数: 实盘口径 %d vs predictions %d" % (n_live, n_pred))
print("  前 %d 只重合: %d" % (top_n, len(tl & tp)))
print("  仅实盘口径有: %s" % sorted(tl - tp))
print("  仅 predictions 有: %s" % sorted(tp - tl))
