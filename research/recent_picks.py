"""最近若干个信号日的过线名单（只读，不写任何生产文件）。

用法: python research/recent_picks.py [天数, 默认 5]
口径: 分数来自 data/cache/predictions.parquet（Walk-Forward 样本外预测），
      建仓线/强线读 config.yaml 的 policy.buy_score / policy.strong_score。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
POL = CFG["position_policy"]
BUY = float(POL["buy_score"])
STRONG = float(POL["strong_score"])
SELL = float(POL["sell_score"])
ADD = float(POL.get("add_score_line", 0.0) or 0.0)
MAXPOS = int(POL["max_names"])

days = int(sys.argv[1]) if len(sys.argv) > 1 else 5

p = pd.read_parquet("data/cache/predictions.parquet")
p["date"] = pd.to_datetime(p["date"])
print("分数表 %s 行 | 列 %s" % (len(p), list(p.columns)))
print("分数起止 %s ~ %s | 建仓线 %.5f 强线 %.5f 清仓线 %.5f 补仓线 %s" % (
    p["date"].min().date(), p["date"].max().date(), BUY, STRONG, SELL,
    ("开 %.5f" % ADD) if ADD > 0 else "关"))
print()

dates = sorted(p["date"].unique())[-days:]
print("### 最近 %d 个信号日：过建仓线的名字（按分数降序）" % days)
for d in dates:
    g = p[p["date"] == d].sort_values("prediction", ascending=False)
    over = g[g["prediction"] >= BUY]
    print("  %s  当日有分 %d 只 | ≥建仓线 %2d 只 | ≥强线 %2d 只 | 最高 %.5f" % (
        pd.Timestamp(d).date(), len(g), len(over),
        int((g["prediction"] >= STRONG).sum()), float(g["prediction"].max())))
    for rank, (_, r) in enumerate(over.head(MAXPOS).iterrows(), start=1):
        flag = "满仓档" if r["prediction"] >= STRONG else ""
        print("      #%02d %s %.5f %s" % (rank, r["symbol"], r["prediction"], flag))
    if len(over) > MAXPOS:
        print("      （过线 %d 只 > 名额 %d，只列前 %d）" % (len(over), MAXPOS, MAXPOS))
