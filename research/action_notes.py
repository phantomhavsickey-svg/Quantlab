# -*- coding: utf-8 -*-
"""现行默认档跑一遍全量回测，统计仓位规则打出的「未动作」备注 —— docs 里那几组
"这个开关一生真的响过几次"的数字出自这里。

    python research/action_notes.py        # 约 20 秒；只读缓存，不改生产代码

为什么需要它：`utils/position_policy.py` 里好几条分支（减仓价锁、降杠杆的最小交易额、
"减一档即清仓"）看起来很重要，但如果 45 轮里一次都没真触发，文档就不该把它写成生效机制。
规则改口径之后这些计数会整组变（清仓线抬到 P84 那一版：减仓价到期解除从 55 次掉到 8 次、
真正拦住重建从 1 次掉到 0 次），所以每次动 `position_policy` 都要重跑这一条。

口径：读 `engine` 的 `logger.debug("未动作 …")`，逐条正则计数 + 把数字掩成 N 后聚类，
所以 Top-12 那一段是"原因种类"的分布，不是动作次数（动作次数看
`research/sell_line.py` 的 `动作 {…}` 那行或 `main.py backtest` 的打印）。
"""
import collections
import contextlib
import io
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # config.yaml 与 data/cache 一律按仓库根目录解析

import pandas as pd
import yaml
from loguru import logger

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from data.cache import CacheManager
from models.predictor import scores_from_predictions
from utils.exposure import overlay_from_config
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config

LOG = io.StringIO()                         # 备注直接收在内存里：Windows 上
logger.remove()                             # 回测结束不等于文件句柄释放
logger.add(LOG, level="DEBUG", format="{message}")

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb, cm = cfg["backtest"], cfg["market"]
cache = CacheManager(cfg["cache"]["directory"])
pred = pd.read_parquet("data/cache/predictions.parquet")
pred["date"] = pd.to_datetime(pred["date"])
PR = pred.set_index(["date", "symbol"])["prediction"]
bars = {}
for s in pred["symbol"].unique():
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        bars[s] = df.sort_values("日期")
SC = scores_from_predictions(PR, build_tradable_mask(bars, pred["date"].unique()))
pol = policy_from_config(cfg)

eng = BacktestEngine(initial_capital=cb.get("initial_capital", 1_000_000),
                     rebalance_frequency="monthly",
                     max_positions=cb["max_positions"],
                     cost_model=TransactionCostModel(
                         cm["commission_rate"], cm["min_commission"],
                         cm["stamp_tax_rate"], cm["slippage_rate"],
                         cm.get("stamp_tax_schedule")),
                     lot_size=int(cm.get("lot_size", 100)),
                     policy=pol, overlay=overlay_from_config(cfg))
with contextlib.redirect_stdout(io.StringIO()):
    r = eng.run(bars, SC)

t = LOG.getvalue()
acts = (r["execution"].get("policy_actions") or {})
print("### 现行默认档（建仓线 %.5f / 清仓线 %.5f）一次全量回测的「未动作」备注"
      % (pol.buy_score, pol.sell_score))
print("  备注总条数 %d | 已成交动作 %s" % (t.count("未动作 "), dict(acts)))
pat = {
    "减仓价约束到期解除": r"超过 \d+ 天有效期,解除",
    "禁止重建（现价高于减仓价）": r"禁止重建",
    "禁止补仓（现价高于减仓价）": r"禁止补仓",
    "缺成交价": r"缺成交价",
    "当日无预测分数": r"当日无预测分数",
    "降杠杆差额 < 最小交易额,不动": r"最小交易额",
    "减一档即改清仓": r"改为清仓",
}
for k, rx in pat.items():
    print("  %-26s %d" % (k, len(re.findall(rx, t))))
print()
print("  全部备注类型（数字掩成 N 后聚类，Top-12）：")
c = collections.Counter()
for line in t.splitlines():
    m = re.match(r"未动作 (\S+) (\S+): (.*)", line)
    if m:
        why = re.sub(r"[\d.]+", "N", m.group(3))
        c[why[:46]] += 1
for why, n in c.most_common(12):
    print("    %4d  %s" % (n, why))
print("done")
