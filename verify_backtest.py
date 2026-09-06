# -*- coding: utf-8 -*-
"""临时验证脚本：重算回测真实成本、基准超额与分年度收益。"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import yaml
from data.cache import CacheManager
from models.predictor import Predictor
from backtest.engine import BacktestEngine
from backtest.cost import TransactionCostModel

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cb = cfg["backtest"]
cm = cfg["market"]

preds_df = pd.read_parquet("data/cache/predictions.parquet")
preds_df["date"] = pd.to_datetime(preds_df["date"])
predictions = preds_df.set_index(["date", "symbol"])["prediction"]

cache = CacheManager(cfg["cache"]["directory"])
symbols = preds_df["symbol"].unique().tolist()
data_dict = {}
for s in symbols:
    df = cache.get_daily(s)
    if df is not None:
        df["日期"] = pd.to_datetime(df["日期"])
        data_dict[s] = df

pred = Predictor(trainer=None, top_k=cb["max_positions"],
                 position_sizing=cb["position_sizing"])
signals = pred.generate_signals_from_series(predictions)

cost = TransactionCostModel(
    commission_rate=cm["commission_rate"],
    min_commission=cm["min_commission"],
    stamp_tax_rate=cm["stamp_tax_rate"],
    slippage_rate=cm["slippage_rate"],
)

# ---- 基准：1000 只成分股等权指数（直接由已下载数据构造） ----
closes = pd.DataFrame(
    {s: df.set_index("日期")["收盘"] for s, df in data_dict.items()})
# 每日等权收益 = 全部成分股日收益的横截面均值（T+1 前复权，无未来函数）
bm = closes.pct_change().mean(axis=1).add(1).cumprod()
bm.name = "close"
lo = predictions.index.get_level_values("date").min()
hi = predictions.index.get_level_values("date").max()
bm = bm[(bm.index >= lo) & (bm.index <= hi)]

engine = BacktestEngine(
    initial_capital=cb.get("initial_capital", 1_000_000),
    rebalance_frequency=cb["rebalance_frequency"],
    max_positions=cb["max_positions"],
    cost_model=cost,
)
result = engine.run(data_dict, signals, benchmark_prices=bm)
m = result["metrics"]
trades = result["trades"]
eq = result["equity_curve"]

lines = []
lines.append("回测区间: %s ~ %s" % (eq.index.min().date(), eq.index.max().date()))
lines.append("交易日数: %d" % len(eq))
lines.append("")
lines.append("=== 策略指标 ===")
for k in ["cumulative_return", "annual_return", "annual_volatility",
          "sharpe_ratio", "calmar_ratio", "max_drawdown", "win_rate"]:
    if k in m:
        lines.append("%s: %.4f" % (k, m[k]))
dd = m.get("max_drawdown_peak_date", None)
dt = m.get("max_drawdown_trough_date", None)
if dd is not None:
    lines.append("最大回撤区间: %s -> %s" % (str(dd)[:10], str(dt)[:10]))
lines.append("")
lines.append("=== 交易与成本 ===")
lines.append("总交易次数: %d" % m.get("total_trades", 0))
lines.append("真实总成本(佣金+印花税+滑点): %.0f 元 (初始资金的 %.2f%%)" % (
    trades["cost"].sum(), trades["cost"].sum() / 1_000_000 * 100))
lines.append("平均单笔成本率: %.4f%%" % (
    (trades["cost"] / trades["amount"]).mean() * 100))
lines.append("")
if "benchmark_cumulative_return" in m:
    lines.append("=== 基准对比 (中证1000) ===")
    lines.append("基准累计收益: %.4f" % m["benchmark_cumulative_return"])
    lines.append("基准年化收益: %.4f" % m["benchmark_annual_return"])
    lines.append("超额收益(累计): %.4f" % m["excess_return"])
    lines.append("信息比率: %.4f" % m["information_ratio"])
else:
    lines.append("=== 基准对比: 下载失败，无法计算 ===")
lines.append("")
lines.append("=== 分年度收益 (策略 vs 基准) ===")
strat_yearly = eq.groupby(eq.index.year).apply(
    lambda g: g.iloc[-1] / g.iloc[0] - 1)
if bm is not None:
    bm_yearly = bm.groupby(bm.index.year).apply(
        lambda g: g.iloc[-1] / g.iloc[0] - 1)
    for y in strat_yearly.index:
        bmv = bm_yearly.get(y, np.nan)
        lines.append("%d: 策略 %+.2f%% | 基准 %+.2f%% | 超额 %+.2f%%" % (
            y, strat_yearly[y] * 100, bmv * 100,
            (strat_yearly[y] - bmv) * 100))
else:
    for y in strat_yearly.index:
        lines.append("%d: 策略 %+.2f%%" % (y, strat_yearly[y] * 100))

with open("reports/verify_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("done")
