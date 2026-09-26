# -*- coding: utf-8 -*-
"""预测分数**按月**的分布：月中位分、负分占比、低于现行清仓线 / 建仓线的占比。

    python research/monthly_scores.py      # 约 10 秒；只读缓存，不改生产代码

它回答的是 `docs/entry-位置管理.md` §9 第 1 条那个问题：把清仓线从 0 抬到 P84 之后，
"整个池子一起掉线、一次性清仓"这种情形到底有多远。答案要量出来才有：全样本逐月
低于清仓线的占比在 65%~96%，**从来没有哪个月是 100%**，所以"级别下移"这条路在
现行样本里走不到；真正在现场发生的是另一种漂移（排序能力归零，见 §9）。

口径：样本 = `data/cache/predictions.parquet` 的全部 854,763 条预测（896 个交易日 ×
1000 只），按自然月分组；占比 = 该月内 `(分数 < 线)` 的**观测条数**占比（名字-日为单位，
不是只数）。池内等权与分年基准不在这里，那是 `research/annual_baseline.py` 的活。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import pandas as pd
import yaml

from utils.position_policy import policy_from_config

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
pol = policy_from_config(cfg)
SELL, BUY, STRONG = pol.sell_score, pol.buy_score, pol.strong_score

pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
print("### 预测分数按月分布（样本 %s 条 / %d 个交易日 / %d 只）"
      % (f"{len(pred):,}", pred["date"].nunique(), pred["symbol"].nunique()))
print("    三条线：清仓 %.5f | 建仓 %.5f | 强档 %.5f" % (SELL, BUY, STRONG))

g = pred.groupby(pred["date"].dt.to_period("M"))["prediction"]
tab = pd.DataFrame({"median": g.median(),
                    "neg": g.apply(lambda x: float((x < 0).mean())),
                    "below_sell": g.apply(lambda x: float((x < SELL).mean())),
                    "below_buy": g.apply(lambda x: float((x < BUY).mean())),
                    "below_strong": g.apply(lambda x: float((x < STRONG).mean()))})
for y in sorted(set(tab.index.year)):
    t = tab[tab.index.year == y]
    print("  %d  月中位分 %+.2f%%~%+.2f%%  负分 %.1f%%~%.1f%%  低于清仓线 %.1f%%~%.1f%%"
          "  低于建仓线 %.1f%%~%.1f%%  低于强线 %.1f%%~%.1f%%" % (
              y, t["median"].min() * 100, t["median"].max() * 100,
              t["neg"].min() * 100, t["neg"].max() * 100,
              t["below_sell"].min() * 100, t["below_sell"].max() * 100,
              t["below_buy"].min() * 100, t["below_buy"].max() * 100,
              t["below_strong"].min() * 100, t["below_strong"].max() * 100))
print("  全样本  月中位分中位 %+.3f%% | 负分占比均值 %.1f%% | 低于清仓线均值 %.1f%%"
      " | 低于建仓线均值 %.1f%% | 低于强线均值 %.1f%%" % (
          tab["median"].median() * 100, tab["neg"].mean() * 100,
          tab["below_sell"].mean() * 100, tab["below_buy"].mean() * 100,
          tab["below_strong"].mean() * 100))
print("  逐月低于清仓线占比：最低 %.1f%%（%s） 最高 %.1f%%（%s）｜共 %d 个月"
      % (tab["below_sell"].min() * 100, tab["below_sell"].idxmin(),
         tab["below_sell"].max() * 100, tab["below_sell"].idxmax(), len(tab)))
print("  有没有哪个月整池 100%% 掉在清仓线下方：%s"
      % ("有" if (tab["below_sell"] >= 0.999).any() else "没有"))
for lab in (2024, 2026):
    print("  %d 各月（中位分 / 负分 / 低于清仓线）: %s" % (lab, " ".join(
        "%s %+.2f%%/%.1f%%/%.1f%%" % (str(p), tab["median"][p] * 100,
                                      tab["neg"][p] * 100,
                                      tab["below_sell"][p] * 100)
        for p in tab.index[tab.index.year == lab])))
print("done")
