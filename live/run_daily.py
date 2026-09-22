#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日批处理 — 数据增量更新 → 信号生成 → 调仓执行。

用法:
    python live/run_daily.py                  # 更新数据 + 输出目标组合/调仓指令（不执行）
    python live/run_daily.py --simulate       # 并用模拟券商撮合（纸面持仓，状态持久化）
    python live/run_daily.py --qmt            # 生成真实柜台指令文件（不自动下单）
    python live/run_daily.py --qmt --confirm  # 通过 QMT 真实下单（请先完成程序化交易报备）

建议用 Windows 任务计划程序每天 15:30 运行（见 live/scheduler_setup.ps1）。
默认不带 --qmt/--confirm，绝对安全，只产出文件。

两条调仓链路:
    Top-K 等权（position_policy.enabled=false）
        只在月末交易日动作，目标是 backtest.max_positions 只等权股票。
    分数带位策略（position_policy.enabled=true）
        **每个交易日都评估**：补仓/减仓由"分数相对上一档移动够一个步长"触发，
        攒到月末才看一次会让触发日与成交日差上几周。每只股票的状态
        （建仓分数、参考分数、步长 Δ、减仓价）持久化在
        live/state/policy_state.json，只在**实际股数发生变化**后推进；
        涨跌停/停牌/资金不足挡下的单子下一轮自动重做。
        与回测共用 utils/position_policy.py，同一条算式。
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
from loguru import logger

from utils.logger import setup_logger
from utils.calendar import is_trading_day, get_month_end_trading_days
from utils.market_rules import at_limit_down, at_limit_up
from utils.position_policy import apply_fills, load_states, policy_from_config, \
    save_states
from data.cache import CacheManager
from data.downloader import DataDownloader
from paper_trade.broker import SimulatedBroker
from live.daily_signal import DailySignalGenerator
from live.qmt_broker import QMTBroker


def _fill_price_map(filled) -> dict:
    """模拟撮合回报 → {symbol: 实际成交价}(减仓价约束用它,不用下单时的参考价)。"""
    out = {}
    for o in filled or []:
        px = float(getattr(o, "filled_price", 0.0) or 0.0)
        if px > 0:
            out[str(getattr(o, "symbol", ""))] = px
    return out


def _commit_policy(states, pol, before, after, fill_price, path, today):
    """按**实际股数变化**提交策略状态并落盘(pol 为 None 时什么也不做)。"""
    if pol is None:
        return
    commits = apply_fills(states, pol.intents, before, after, fill_price,
                          asof=today)
    for s, msg in commits.items():
        logger.info(f"策略状态 {s}: {msg}")
    save_states(path, states)
    logger.info(f"策略状态已落盘: {path}({len(states)} 只有状态)")


def main():
    parser = argparse.ArgumentParser(description="QuantLab 每日批处理")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（默认取 config.backtest.initial_capital）")
    parser.add_argument("--simulate", action="store_true",
                        help="用模拟券商撮合执行（纸面持仓持久化）")
    parser.add_argument("--qmt", action="store_true",
                        help="通过 QMT 柜台执行（必须同时加 --confirm 才真正下单）")
    parser.add_argument("--confirm", action="store_true",
                        help="确认真实下单（仅配合 --qmt 使用）")
    parser.add_argument("--lookback-days", type=int, default=None,
                        help="信号计算回看交易日数（默认取 config.live.lookback_days）")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg_live = config.get("live", {})
    cfg_market = config["market"]
    cfg_backtest = config["backtest"]
    capital = args.capital or cfg_backtest.get("initial_capital", 1_000_000)
    lookback = args.lookback_days or cfg_live.get("lookback_days", 250)

    setup_logger(log_level="INFO", log_file="logs/live.log",
                 rotation="10 MB", retention="30 days")

    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    logger.info(f"===== 每日批处理 {today_str} =====")

    if not is_trading_day(today):
        logger.info("今日非交易日，跳过")
        return

    # ---- 1. 增量更新数据 ----
    cache = CacheManager(config["cache"]["directory"])
    downloader = DataDownloader(cache, **(config.get("download") or {}))
    symbols = cache.list_cached_symbols()
    if not symbols:
        logger.error("缓存中没有股票数据，请先运行 pipeline")
        return
    start_ak = (today - timedelta(days=lookback * 2 + 30)).strftime("%Y%m%d")
    end_ak = today.strftime("%Y%m%d")
    logger.info(f"增量更新 {len(symbols)} 只股票日线至 {today_str} ...")
    downloader.download_batch_daily(symbols, start_ak, end_ak, adjust="qfq")

    # ---- 2. 信号生成 ----
    policy = policy_from_config(config)
    state_dir = cfg_live.get("state_dir", "live/state")
    os.makedirs(state_dir, exist_ok=True)
    policy_state_file = os.path.join(state_dir, "policy_state.json")
    states = load_states(policy_state_file) if policy else {}
    if policy is not None:
        logger.info(
            f"仓位策略已启用: 建仓线 {policy.buy_score:.4f} / "
            f"清仓线 {policy.sell_score:.4f} / "
            f"{policy.base_weight:.0%}→{policy.max_entry_weight:.0%} 建仓,"
            f" 单票上限 {policy.max_position_weight:.0%},"
            f" 新仓上限 {policy.max_names} 只,"
            f" 已有状态 {len(states)} 只")

    gen = DailySignalGenerator(config)
    if policy is not None:
        # 策略模式要全截面分数,不要名单:掉出 Top-K ≠ 跌破清仓线
        scores = gen.signal_scores(today_str, lookback_days=lookback)
        target = pd.Series(dtype=float)
        if scores.empty:
            logger.warning("分数为空，仅完成数据更新")
            return
    else:
        scores = pd.Series(dtype=float)
        target = gen.generate(today_str, lookback_days=lookback)
        if target is None or len(target) == 0:
            logger.warning("信号为空，仅完成数据更新")
            return

    order_dir = cfg_live.get("order_dir", "live/output")
    os.makedirs(order_dir, exist_ok=True)
    tag = today_str.replace("-", "")
    if policy is None:
        gen.export_target(target, os.path.join(order_dir, f"target_{tag}.csv"))

    # ---- 3. 调仓日判断 ----
    # 策略模式下每个交易日都评估(补/减是分数触发的,等到月末就失去意义);
    # Top-K 等权仍然只在月末交易日动组合。
    month_end = [d.strftime("%Y-%m-%d") for d in
                 get_month_end_trading_days(today_str[:8] + "01", today_str)]
    if policy is None and today_str not in month_end:
        logger.info("今日非月末调仓日，不执行交易")
        return

    # ---- 4. 参考价与今日行情快照 ----
    want = set(symbols) | set(scores.index if policy is not None
                              else target.index)
    ref_price = {}
    snapshot = {}
    for sym in want:
        df = cache.get_daily(sym)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"])
        last = df.iloc[-1]
        ref_price[sym] = float(last["收盘"])
        snapshot[sym] = {
            "open": float(last["开盘"]),
            "high": float(last["最高"]),
            "low": float(last["最低"]),
            "close": float(last["收盘"]),
            "volume": float(last["成交量"]),
            "at_limit_up": at_limit_up(sym, last.get("涨跌幅")),
            "at_limit_down": at_limit_down(sym, last.get("涨跌幅")),
        }

    # ---- 5. 执行 ----
    if args.simulate:
        broker = SimulatedBroker(
            initial_cash=capital,
            commission_rate=cfg_market["commission_rate"],
            min_commission=cfg_market["min_commission"],
            stamp_tax_rate=cfg_market["stamp_tax_rate"],
            slippage_rate=cfg_market["slippage_rate"],
            lot_size=cfg_market["lot_size"],
        )
        state_file = os.path.join(state_dir, "simulate_state.json")
        broker.load_state(state_file)

        # 持仓对象直接传给指令构造:带 available_shares,T+1 锁仓才算得准
        before = {str(s): int(p.shares) for s, p in broker.positions.items()}
        if policy is not None:
            orders, pol = gen.make_policy_orders(
                scores, broker.positions, broker.cash, ref_price, states,
                policy, asof=today)
            target = pd.Series({s: w for s, w in pol.weights.items() if w > 0})
            gen.export_target(target, os.path.join(order_dir, f"target_{tag}.csv"))
        else:
            positions = {s: p.shares for s, p in broker.positions.items()}
            orders, pol = gen.make_orders(target, positions, broker.cash,
                                          ref_price), None
        gen.export_orders(orders, os.path.join(order_dir, f"orders_{tag}.csv"))

        for o in orders:
            broker.place_market_order(o["symbol"], o["side"], o["quantity"])
        filled = broker.process_daily(today, snapshot)
        logger.info(f"模拟撮合完成: {len(filled)} 笔成交")

        broker.save_state(state_file)
        after = {str(s): int(p.shares) for s, p in broker.positions.items()}
        _commit_policy(states, pol, before, after, _fill_price_map(filled),
                       policy_state_file, today)
        total_value = broker.get_total_value()
        logger.info(f"模拟盘资产: {total_value:,.0f} | "
                    f"持仓 {len(broker.positions)} 只 | "
                    f"盈亏 {total_value - capital:+,.0f}")
        with open(os.path.join(order_dir, f"summary_{tag}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"日期: {today_str}\n"
                    f"模式: simulate\n"
                    f"仓位策略: {'开' if policy else '关'}\n"
                    f"目标持仓: {len(target)} 只\n"
                    f"指令笔数: {len(orders)}\n"
                    f"模拟盘总资产: {total_value:,.0f}\n"
                    f"累计盈亏: {total_value - capital:+,.0f}\n")

    elif args.qmt:
        qmt_broker = QMTBroker(cfg_live.get("qmt", {}), capital)
        positions = {p["symbol"]: p["shares"]
                     for p in qmt_broker.get_positions_summary()}
        cash = qmt_broker.get_cash()
        before = {str(s): int(q) for s, q in positions.items()}
        if policy is not None:
            orders, pol = gen.make_policy_orders(
                scores, positions, cash, ref_price, states, policy, asof=today)
            target = pd.Series({s: w for s, w in pol.weights.items() if w > 0})
            gen.export_target(target, os.path.join(order_dir, f"target_{tag}.csv"))
        else:
            orders, pol = gen.make_orders(target, positions, cash,
                                          ref_price), None
        gen.export_orders(orders, os.path.join(order_dir, f"orders_{tag}.csv"))
        if args.confirm:
            logger.warning("开始真实下单（--confirm）")
            for o in orders:
                qmt_broker.place_market_order(o["symbol"], o["side"],
                                              o["quantity"], o["ref_price"])
            logger.warning("真实下单完成，请尽快核对成交回报")
            # QMTBroker 没有 query_stock_trades,按下单后的持仓快照提交状态:
            # 已成交的部分会被 rebalance_plan 认成"仓位已到位"而不再重复下单,
            # 未成交(挂单中/被拒)的部分下一轮重新提议 —— 与回测同一套"只按实际
            # 股数变化推进"的口径。当天连跑两次会对未成交单重复报单,请勿重复执行。
            after_now = {str(p["symbol"]): int(p["shares"])
                         for p in qmt_broker.get_positions_summary()}
            _commit_policy(states, pol, before, after_now, {},
                           policy_state_file, today)
        else:
            logger.warning("QMT 模式未加 --confirm，未下单；"
                           f"指令文件已输出: {order_dir}/orders_{tag}.csv")
    else:
        if policy is not None:
            orders, pol = gen.make_policy_orders(
                scores, {}, capital, ref_price, states, policy, asof=today)
            target = pd.Series({s: w for s, w in pol.weights.items() if w > 0})
            gen.export_target(target, os.path.join(order_dir, f"target_{tag}.csv"))
        else:
            orders = gen.make_orders(target, {}, capital, ref_price)
        gen.export_orders(orders, os.path.join(order_dir, f"orders_{tag}.csv"))
        logger.info(f"未指定 --simulate/--qmt，仅输出指令文件: "
                    f"{order_dir}/orders_{tag}.csv（买入数量按初始资金估算）")

    logger.info("===== 每日批处理完成 =====")


if __name__ == "__main__":
    main()
