#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时模拟交易 — 接入新浪实时行情 + 模型信号 + 模拟券商执行。

注意：本文件是盘中行情演示版。日常信号/调仓请使用 live/run_daily.py
（批处理链路：因子预处理与回测完全一致，支持状态持久化与 QMT 柜台）。

用法:
    python live/live_trading.py --capital 1000000

工作原理:
    1. 每 5 秒拉取一次实时行情
    2. 每天 9:35 (开盘后5分钟) 执行一次信号生成 + 调仓
    3. 调仓后持续盯市，计算盈亏
    4. 按 Ctrl+C 停止

注: 这是一个演示版，展示完整链路。实际部署需要:
    - 定时任务 (schedule/cron) 替代 sleep 轮询
    - 行情 WebSocket 替代 HTTP 轮询 (降低延迟)
    - vn.py/CTP 网关替代模拟撮合 (接真实柜台)
"""

import os
import sys
import time
import argparse
import signal
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from live import SinaQuoteFeed, QuoteMonitor
from data.cache import CacheManager
from models.trainer import LightGBMTrainer
from models.predictor import Predictor
from paper_trade.broker import SimulatedBroker, OrderSide
from paper_trade.portfolio import PortfolioTracker
from paper_trade.journal import TradeJournal
from factors.technical import TechnicalFactors
from utils.logger import setup_logger


class LiveTradingEngine:
    """实时模拟交易引擎。

    连接实时行情 → 模型生成信号 → 模拟下单 → 盯市。
    """

    def __init__(self, capital: float, config: dict):
        self.capital = capital
        self.config = config
        self.cfg_market = config["market"]
        self.cfg_backtest = config["backtest"]

        # ---- 加载模型 ----
        self.trainer = self._load_model()
        self.predictor = Predictor(
            self.trainer,
            top_k=self.cfg_backtest["max_positions"],
            position_sizing=self.cfg_backtest["position_sizing"],
        )

        # ---- 初始化组件 ----
        self.broker = SimulatedBroker(
            initial_cash=capital,
            commission_rate=self.cfg_market["commission_rate"],
            min_commission=self.cfg_market["min_commission"],
            stamp_tax_rate=self.cfg_market["stamp_tax_rate"],
            slippage_rate=self.cfg_market["slippage_rate"],
        )
        self.portfolio = PortfolioTracker(initial_capital=capital)
        self.journal = TradeJournal()
        self.feed = SinaQuoteFeed()

        # ---- 股票池 ----
        cache = CacheManager(config["cache"]["directory"])
        self.symbols = cache.list_cached_symbols()
        logger.info(f"实时交易股票池: {len(self.symbols)} 只")

        # ---- 状态 ----
        self.last_rebalance_date: date | None = None
        self.running = False
        self.quotes_cache: dict = {}

    def _load_model(self) -> LightGBMTrainer:
        """加载最新模型。"""
        saved_dir = "models/saved"
        models = sorted([f for f in os.listdir(saved_dir)
                         if f.endswith(".txt")])
        if not models:
            raise FileNotFoundError("未找到已训练模型")
        path = os.path.join(saved_dir, models[-1])
        trainer = LightGBMTrainer(model_type=self.config["model"]["type"])
        trainer.load(path)
        logger.info(f"模型已加载: {path}")
        return trainer

    # ==================== 信号生成 ====================

    def generate_signals(self) -> pd.DataFrame:
        """基于最新行情 + 缓存历史数据生成今日交易信号。

        流程:
            1. 从缓存加载每个股票的近期日线
            2. 计算技术因子
            3. 模型预测 → 排名 → 选 Top-K
            4. 返回目标权重
        """
        cache = CacheManager(self.config["cache"]["directory"])
        tech_periods = self.config["factors"]["technical"]

        all_factors = []
        for sym in self.symbols:
            df = cache.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            df["日期"] = pd.to_datetime(df["日期"])

            # 把实时价格追加到最后一行
            if sym in self.quotes_cache:
                latest = self.quotes_cache[sym]
                last_row = df.iloc[-1:].copy()
                last_row["日期"] = pd.Timestamp(date.today())
                last_row["收盘"] = latest.price
                last_row["开盘"] = latest.open
                last_row["最高"] = latest.high
                last_row["最低"] = latest.low
                # 实时接口无可靠的当日换手率，置 NaN 交由模型处理
                # （训练数据的换手率来自日线接口，口径一致）
                last_row["换手率"] = float("nan")
                df = pd.concat([df, last_row], ignore_index=True)

            factors = TechnicalFactors.compute_all(df, tech_periods)
            factors["symbol"] = sym
            factors["date"] = df["日期"].values[-1]  # 取最后一天的因子
            all_factors.append(factors.iloc[-1:])

        if not all_factors:
            logger.warning("无数据可用于信号生成")
            return pd.DataFrame()

        factor_df = pd.concat(all_factors, ignore_index=True)
        factor_cols = [c for c in factor_df.columns
                       if c not in ["date", "symbol"]]
        X = factor_df.set_index(["date", "symbol"])[factor_cols]

        if X.empty:
            return pd.DataFrame()

        signals = self.predictor.generate_signals(X)
        logger.info(f"信号更新: {len(signals[signals['weight']>0])} 只目标持仓")
        return signals

    # ==================== 调仓执行 ====================

    def rebalance(self):
        """执行一次调仓：生成信号 → 卖旧 → 买新。"""
        today = date.today()
        if today.weekday() >= 5:  # 周末不调仓
            return

        logger.info(f"[{today}] 开始调仓...")

        # 1. 生成信号
        signals = self.generate_signals()
        if signals.empty:
            return

        # 2. 获取当天的目标信号
        try:
            today_ts = pd.Timestamp(today)
            if today_ts in signals.index.get_level_values("date"):
                day_signals = signals.xs(today_ts, level="date")
            else:
                # 取最近的信号日期
                signal_dates = sorted(signals.index.get_level_values("date").unique())
                nearest = signal_dates[-1]
                day_signals = signals.xs(nearest, level="date")
        except Exception:
            return

        target = day_signals[day_signals["weight"] > 0]
        target_symbols = set(target.index)

        if not target_symbols:
            logger.info("今日无选中股票，跳过调仓")
            return

        # 3. 卖出不在目标中的持仓
        for sym, pos in list(self.broker.positions.items()):
            if sym not in target_symbols and pos.available_shares > 0:
                self.broker.place_market_order(sym, "sell", pos.available_shares)
                logger.info(f"  卖出: {sym} x{pos.available_shares}")

        # 4. 买入目标
        buy_capital = self.broker.cash / max(len(target_symbols), 1)
        for sym in target_symbols:
            if sym in self.quotes_cache:
                price = self.quotes_cache[sym].price
                shares = self.broker.lot_size * (
                    int(buy_capital / price) // self.broker.lot_size
                )
                if shares >= self.broker.lot_size:
                    self.broker.place_market_order(sym, "buy", shares)
                    logger.info(f"  买入: {sym} x{shares} @ {price:.2f}")

        self.last_rebalance_date = today
        logger.info(f"调仓完成: 目标 {len(target_symbols)} 只, "
                     f"持仓 {len(self.broker.positions)} 只, "
                     f"现金 {self.broker.cash:,.0f}")

    # ==================== 行情回调 ====================

    def on_quotes(self, quotes: dict):
        """收到实时行情时调用。"""
        self.quotes_cache = quotes
        now = datetime.now()

        # 每30秒打印状态
        if now.second % 30 == 0:
            self._print_status()

    def _print_status(self):
        """打印当前状态。"""
        total_value = self.broker.get_total_value()
        cash = self.broker.cash
        market_value = total_value - cash
        pnl = total_value - self.capital
        pnl_pct = pnl / self.capital * 100

        # 找最大涨跌幅的持仓
        biggest_mover = None
        biggest_change = 0
        for sym, pos in self.broker.positions.items():
            if sym in self.quotes_cache:
                q = self.quotes_cache[sym]
                if abs(q.change_pct) > abs(biggest_change):
                    biggest_change = q.change_pct
                    biggest_mover = q

        mover_str = ""
        if biggest_mover:
            mover_str = (f" | 异动: {biggest_mover.name} "
                         f"{biggest_mover.change_pct:+.2f}%")

        print(f"\r[{(datetime.now().strftime('%H:%M:%S'))}] "
              f"净值={total_value/capital:.4f} | "
              f"持仓={len(self.broker.positions)}只 | "
              f"浮动={pnl_pct:+.2f}%{mover_str}"
              f"      ", end="", flush=True)

    # ==================== 主循环 ====================

    def run(self, rebalance_time: str = "09:35"):
        """启动实时交易循环。

        Args:
            rebalance_time: 每日调仓时间 (HH:MM)，默认开盘后5分钟
        """
        self.running = True
        logger.info(f"=" * 55)
        logger.info(f"  实时模拟交易已启动")
        logger.info(f"  资金: {self.capital:,.0f} | 股票池: {len(self.symbols)} 只")
        logger.info(f"  行情源: 新浪财经 | 调仓时间: 每日 {rebalance_time}")
        logger.info(f"  按 Ctrl+C 停止")
        logger.info(f"=" * 55)

        # 首次立即调仓
        self.rebalance()

        rebalance_hour = int(rebalance_time.split(":")[0])
        rebalance_min = int(rebalance_time.split(":")[1])

        try:
            while self.running:
                now = datetime.now()

                # 非交易时间跳过
                if now.weekday() >= 5:
                    time.sleep(60)
                    continue

                # 到达调仓时间且今天还没调过
                if (now.hour == rebalance_hour and
                        now.minute == rebalance_min and
                        self.last_rebalance_date != date.today()):
                    self.rebalance()

                # 拉行情
                try:
                    quotes = self.feed.fetch(self.symbols)
                    self.on_quotes(quotes)

                    # 盯市 + 撮合
                    market_snapshot = {
                        sym: {
                            "open": q.open,
                            "high": q.high,
                            "low": q.low,
                            "close": q.price,
                            "volume": q.volume,
                            "at_limit_up": q.change_pct >= 9.5,
                            "at_limit_down": q.change_pct <= -9.5,
                        }
                        for sym, q in quotes.items()
                    }
                    filled = self.broker.process_daily(date.today(),
                                                        market_snapshot)
                    for order in filled:
                        self.journal.log_trade(order)

                    self.portfolio.update(date.today(), self.broker)

                except Exception as e:
                    logger.error(f"行情处理异常: {e}")

                time.sleep(5)  # 5秒轮询

        except KeyboardInterrupt:
            logger.info("收到停止信号")
        finally:
            self.stop()

    def stop(self):
        """停止交易，打印结果。"""
        self.running = False

        total_value = self.broker.get_total_value()
        pnl = total_value - self.capital

        print(f"\n\n{'='*55}")
        print(f"  实时模拟交易结束")
        print(f"  最终资金: {total_value:,.0f} 元")
        print(f"  累计盈亏: {pnl:+,.0f} 元 ({pnl/self.capital*100:+.2f}%)")
        print(f"  当前持仓: {len(self.broker.positions)} 只")
        print(f"{'='*55}\n")

        # 持仓明细
        positions = self.broker.get_positions_summary()
        if positions:
            pos_df = pd.DataFrame(positions).sort_values("pnl_pct", ascending=False)
            print("持仓明细 (Top-10):")
            print(pos_df.head(10)[["symbol", "shares", "pnl_pct"]].to_string(index=False))

        # 导出
        self.journal.export_csv("live_trading_result")
        logger.info("交易日志已导出")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="QuantLab 实时模拟交易")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="初始资金 (默认: 100万)")
    parser.add_argument("--rebalance-time", default="09:35",
                        help="每日调仓时间 (默认: 09:35)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true",
                        help="演示模式: 开盘时间外也能跑（用最近行情测试）")
    args = parser.parse_args()

    # 加载配置
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 日志
    setup_logger()

    # 启动引擎
    engine = LiveTradingEngine(capital=args.capital, config=config)

    if args.demo:
        logger.info("演示模式: 拉一次行情 + 生成信号 + 打印结果")
        feed = SinaQuoteFeed()
        cache = CacheManager(config["cache"]["directory"])
        symbols = cache.list_cached_symbols()[:10]  # 只取10只演示

        print(f"\n拉取 {len(symbols)} 只股票实时行情...")
        quotes = feed.fetch(symbols)
        for sym in sorted(quotes.keys()):
            q = quotes[sym]
            print(f"  {q}")

        print(f"\n当前时间: {datetime.now()}")
        if datetime.now().weekday() >= 5:
            print("⚠️  现在是周末/A股休市，行情数据为上一交易日收盘价")
    else:
        engine.run(rebalance_time=args.rebalance_time)


if __name__ == "__main__":
    main()
