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
# 与 main.py backtest 同口径:信号日无成交/已停牌的不占 Top-K 名额
from utils.market_rules import build_tradable_mask
from utils.position_policy import policy_from_config
from utils.exposure import overlay_from_config
tradable = build_tradable_mask(
    data_dict, predictions.index.get_level_values("date").unique())
policy = policy_from_config(cfg)          # 与 main.py 一样:策略开就跑策略口径
overlay = overlay_from_config(cfg) if policy is not None else None
if policy is None:
    signals = pred.generate_signals_from_series(predictions, tradable=tradable)
else:
    from models.predictor import scores_from_predictions
    signals = scores_from_predictions(predictions, tradable=tradable)

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
# 不按 predictions 区间裁基准：净值曲线走到缓存末日(比 predictions 晚约一个月)，
# 裁短会把基准最后一段收益直接抹掉。引擎自己会对齐到净值区间。
lo, hi = closes.index.min(), closes.index.max()

# ---- 真实指数：等权组合用的是"今天的"1000 只成分，含幸存者偏差 ----
# 指数接口优先腾讯直连（东财在本机 IP 上常被限流）
bm_real = None
try:
    from data.downloader import DataDownloader
    idx = DataDownloader(cache, **(cfg.get("download") or {})).download_index_daily(
        cb.get("benchmark", "000852"), lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d"))
    bm_real = idx.set_index("日期")["收盘"]
except Exception as e:
    print("真实指数基准不可用，只用等权组合:", e)

engine = BacktestEngine(
    initial_capital=cb.get("initial_capital", 1_000_000),
    rebalance_frequency=cb["rebalance_frequency"],
    max_positions=cb["max_positions"],
    cost_model=cost,
    lot_size=int(cm.get("lot_size", 100)),
    policy=policy,
    overlay=overlay,
)
result = engine.run(data_dict, signals,
                    benchmark_prices=bm_real if bm_real is not None else bm)
m = result["metrics"]
trades = result["trades"]
eq = result["equity_curve"]

lines = []
lines.append("引擎口径: 2026-09-21 执行层修复后(差额建仓/无前视盯市/可交易性约束)")
if policy is None:
    lines.append("仓位口径: Top-K 等权(position_policy 关闭)")
else:
    ex = result["execution"]
    lines.append("仓位口径: 分数带位策略(建仓线 %.4f / 清仓线 %.4f / %.0f%%→%.0f%%"
                 " 建仓 / 单票上限 %.0f%% / 新仓上限 %d 只)" % (
                     policy.buy_score, policy.sell_score,
                     policy.base_weight * 100, policy.max_entry_weight * 100,
                     policy.max_position_weight * 100, policy.max_names))
    lines.append("策略动作: %s | 平均目标仓位 %.1f%% | 平均持仓 %.1f 只 | 期末 %d 只" % (
        ex["policy_actions"], ex["policy_mean_gross_weight"] * 100,
        ex["policy_mean_names"], ex["policy_final_names"]))
    if overlay is not None and "exposure_mean_cap" in ex:
        lines.append("暴露层: B 目标波动 %.0f%%/回看 %d 日/缩放地板 %.2f"
                     " + A RankIC 门控(窗 %d 日、标签 %d 日、≥%d 观测、触发 ×%.2f)"
                     % (overlay.vol_target_ann * 100, overlay.vol_lookback_days,
                        overlay.scale_floor, overlay.ic_window_days,
                        overlay.ic_horizon_days, overlay.ic_min_obs,
                        overlay.ic_cap_mult))
        er = ex["exposure"]
        lines.append("      平均仓位上限 %.1f%% (最低 %.1f%%) | IC 门控触发 %d/%d"
                     " 次评估 | 全池已实现波动均值 %.1f%%%s" % (
                         ex["exposure_mean_cap"] * 100, ex["exposure_min_cap"] * 100,
                         ex["n_gated_evals"], len(er),
                         er["realized_vol"].mean() * 100,
                         (" | 窗口 RankIC 均值 %+.4f / 最低 %+.4f"
                          % (er["rank_ic"].mean(), er["rank_ic"].min())
                          if er["rank_ic"].notna().any() else "")))
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
if "benchmark_cumulative_return" in m and result.get("benchmark_curve") is not None:
    lines.append("=== 基准对比 (%s) ===" % (
        "中证1000 指数" if bm_real is not None else "1000 成分等权组合"))
    lines.append("基准累计收益: %.4f" % m["benchmark_cumulative_return"])
    lines.append("基准年化收益: %.4f" % m["benchmark_annual_return"])
    lines.append("超额收益(累计): %.4f" % m["excess_return"])
    lines.append("信息比率: %.4f" % m["information_ratio"])
    # 与净值曲线同区间才可比：predictions 比净值曲线早一个月开始
    ew = bm[(bm.index >= eq.index.min()) & (bm.index <= eq.index.max())]
    lines.append("对照·等权组合累计收益: %.4f (今日成分，含幸存者偏差)" % (
        ew.iloc[-1] / ew.iloc[0] - 1))
else:
    lines.append("=== 基准对比: 下载失败，无法计算 ===")
lines.append("")
lines.append("=== 分年度收益 (策略 vs 各基准) ===")
strat_yearly = eq.groupby(eq.index.year).apply(
    lambda g: g.iloc[-1] / g.iloc[0] - 1)
bms = [("等权组合", bm)] if bm_real is None else \
      [("中证1000", result.get("benchmark_curve")), ("等权组合", bm)]
yearly = []
for name, c in bms:
    c = c[(c.index >= eq.index.min()) & (c.index <= eq.index.max())]
    yearly.append((name, c.groupby(c.index.year).apply(
        lambda g: g.iloc[-1] / g.iloc[0] - 1)))
for y in strat_yearly.index:
    seg = " | ".join("%s %+.2f%%" % (n, gy.get(y, np.nan) * 100)
                     for n, gy in yearly)
    lines.append("%d: 策略 %+.2f%% | %s" % (y, strat_yearly[y] * 100, seg))

# ---- 与 evidence/verify_results.txt(旧引擎、同一子区间)可比的切片 ----
S, E = pd.Timestamp("2023-02-01"), pd.Timestamp("2025-12-31")
lines.append("")
lines.append("=== 可比子区间 %s ~ %s (旧存档 evidence/ 的口径) ===" % (S.date(), E.date()))


def _seg(curve):
    c = curve[(curve.index >= S) & (curve.index <= E)]
    r = c.pct_change().dropna()
    cum = c.iloc[-1] / c.iloc[0] - 1
    dd = (c / c.cummax() - 1).min()
    return ("累计 %+.2f%% | 年化 %+.2f%% | 年化波动 %.2f%% | 最大回撤 %.2f%%"
            % (cum * 100, ((1 + cum) ** (252 / len(c)) - 1) * 100,
               r.std() * np.sqrt(252) * 100, dd * 100))


lines.append("策略: " + _seg(eq))
lines.append("等权组合: " + _seg(bm))
if bm_real is not None:
    lines.append("中证1000 指数: " + _seg(bm_real))

with open("reports/verify_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("done")
