#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日批处理 — 数据增量更新 → 信号生成 → 月末调仓执行。

用法:
    python live/run_daily.py                  # 更新数据 + 输出目标组合/调仓指令（不执行）
    python live/run_daily.py --simulate       # 并用模拟券商撮合（纸面持仓，状态持久化）
    python live/run_daily.py --qmt            # 生成真实柜台指令文件（不自动下单）
    python live/run_daily.py --qmt --confirm  # 通过 QMT 真实下单（请先完成程序化交易报备）

建议用 Windows 任务计划程序每天 15:30 运行（见 live/scheduler_setup.ps1）。
默认不带 --qmt/--confirm，绝对安全，只产出文件。
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
from data.cache import CacheManager
from data.downloader import DataDownloader
from paper_trade.broker import SimulatedBroker
from live.daily_signal import DailySignalGenerator
from live.qmt_broker import QMTBroker


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
    downloader = DataDownloader(cache)
    symbols = cache.list_cached_symbols()
    if not symbols:
        logger.error("缓存中没有股票数据，请先运行 pipeline")
        return
    start_ak = (today - timedelta(days=lookback * 2 + 30)).strftime("%Y%m%d")
    end_ak = today.strftime("%Y%m%d")
    logger.info(f"增量更新 {len(symbols)} 只股票日线至 {today_str} ...")
    downloader.download_batch_daily(symbols, start_ak, end_ak, adjust="qfq")

    # ---- 2. 信号生成 ----
    gen = DailySignalGenerator(config)
    target = gen.generate(today_str, lookback_days=lookback)
    if target is None or len(target) == 0:
        logger.warning("信号为空，仅完成数据更新")
        return

    order_dir = cfg_live.get("order_dir", "live/output")
    os.makedirs(order_dir, exist_ok=True)
    tag = today_str.replace("-", "")
    gen.export_target(target, os.path.join(order_dir, f"target_{tag}.csv"))

    # ---- 3. 调仓日判断（月末交易日） ----
    month_end = [d.strftime("%Y-%m-%d") for d in
                 get_month_end_trading_days(today_str[:8] + "01", today_str)]
    if today_str not in month_end:
        logger.info("今日非月末调仓日，不执行交易")
        return

    # ---- 4. 参考价与今日行情快照 ----
    ref_price = {}
    snapshot = {}
    for sym in list(set(symbols) | set(target.index)):
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
            "at_limit_up": float(last.get("涨跌幅", 0) or 0) >= 9.5,
            "at_limit_down": float(last.get("涨跌幅", 0) or 0) <= -9.5,
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
        state_dir = cfg_live.get("state_dir", "live/state")
        os.makedirs(state_dir, exist_ok=True)
        state_file = os.path.join(state_dir, "simulate_state.json")
        broker.load_state(state_file)

        positions = {s: p.shares for s, p in broker.positions.items()}
        orders = gen.make_orders(target, positions, broker.cash, ref_price)
        gen.export_orders(orders, os.path.join(order_dir, f"orders_{tag}.csv"))

        for o in orders:
            broker.place_market_order(o["symbol"], o["side"], o["quantity"])
        filled = broker.process_daily(today, snapshot)
        logger.info(f"模拟撮合完成: {len(filled)} 笔成交")

        broker.save_state(state_file)
        total_value = broker.get_total_value()
        logger.info(f"模拟盘资产: {total_value:,.0f} | "
                    f"持仓 {len(broker.positions)} 只 | "
                    f"盈亏 {total_value - capital:+,.0f}")
        with open(os.path.join(order_dir, f"summary_{tag}.txt"),
                  "w", encoding="utf-8") as f:
            f.write(f"日期: {today_str}\n"
                    f"模式: simulate\n"
                    f"目标持仓: {len(target)} 只\n"
                    f"指令笔数: {len(orders)}\n"
                    f"模拟盘总资产: {total_value:,.0f}\n"
                    f"累计盈亏: {total_value - capital:+,.0f}\n")

    elif args.qmt:
        qmt_broker = QMTBroker(cfg_live.get("qmt", {}), capital)
        positions = {p["symbol"]: p["shares"]
                     for p in qmt_broker.get_positions_summary()}
        cash = qmt_broker.get_cash()
        orders = gen.make_orders(target, positions, cash, ref_price)
        gen.export_orders(orders, os.path.join(order_dir, f"orders_{tag}.csv"))
        if args.confirm:
            logger.warning("开始真实下单（--confirm）")
            for o in orders:
                qmt_broker.place_market_order(o["symbol"], o["side"],
                                              o["quantity"], o["ref_price"])
            logger.warning("真实下单完成，请尽快核对成交回报")
        else:
            logger.warning("QMT 模式未加 --confirm，未下单；"
                           f"指令文件已输出: {order_dir}/orders_{tag}.csv")
    else:
        gen.export_orders(
            gen.make_orders(target, {}, capital, ref_price),
            os.path.join(order_dir, f"orders_{tag}.csv"))
        logger.info(f"未指定 --simulate/--qmt，仅输出指令文件: "
                    f"{order_dir}/orders_{tag}.csv（买入数量按初始资金估算）")

    logger.info("===== 每日批处理完成 =====")


if __name__ == "__main__":
    main()
