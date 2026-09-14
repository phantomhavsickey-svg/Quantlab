#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日自动交易流程 — 收盘后自动更新数据 → 算因子 → 出信号 → 模拟下单。

用法:
    python daily_runner.py                    # 手动运行一次
    python daily_runner.py --auto             # 自动模式（每天15:30运行）
    python daily_runner.py --status           # 查看当前持仓和信号

工作原理:
    每个交易日:
      15:30  更新当日日线数据（AKShare/腾讯接口）
      15:35  重新计算所有因子
      15:40  模型预测 → 排名 → 选股 → 生成明日信号
      15:45  打印明日操作建议（买什么、卖什么）
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yaml
from loguru import logger

from data.cache import CacheManager
from data.downloader import DataDownloader
from data.cleaner import DataCleaner
from factors.technical import TechnicalFactors
from factors.processor import FactorProcessor
from models.trainer import LightGBMTrainer
from models.predictor import Predictor
from paper_trade.broker import SimulatedBroker
from paper_trade.portfolio import PortfolioTracker
from paper_trade.journal import TradeJournal
from portfolio_manager import PortfolioManager
from live import SinaQuoteFeed
from utils.logger import setup_logger


class DailyRunner:
    """每日自动交易执行器。

    每天收盘后自动:
        1. 更新数据
        2. 算因子
        3. 出信号
        4. 模拟下单
        5. 记录持仓
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.cache = CacheManager(self.config["cache"]["directory"])
        self.downloader = DataDownloader(self.cache)
        self.feed = SinaQuoteFeed()

        # 加载模型
        self.trainer = self._load_latest_model()
        self.predictor = Predictor(
            self.trainer,
            top_k=self.config["backtest"]["max_positions"],
            position_sizing=self.config["backtest"]["position_sizing"],
        )

        # 持仓管理器（真实状态持久化）
        self.pm = PortfolioManager(self.config)

        # 模拟券商（用于跟踪持仓）
        self.broker = SimulatedBroker(
            initial_cash=1_000_000,
            commission_rate=self.config["market"]["commission_rate"],
            min_commission=self.config["market"]["min_commission"],
            stamp_tax_rate=self.config["market"]["stamp_tax_rate"],
            slippage_rate=self.config["market"]["slippage_rate"],
        )
        self.portfolio = PortfolioTracker(initial_capital=1_000_000)
        self.journal = TradeJournal()

        # 当前信号
        self.current_signals: pd.DataFrame | None = None
        self.symbols = self.cache.list_cached_symbols()

    def _load_latest_model(self) -> LightGBMTrainer:
        saved_dir = "models/saved"
        models = sorted([f for f in os.listdir(saved_dir)
                         if f.endswith(".txt")])
        if not models:
            raise FileNotFoundError("未找到已训练模型，请先训练")
        path = os.path.join(saved_dir, models[-1])
        trainer = LightGBMTrainer(model_type=self.config["model"]["type"])
        trainer.load(path)
        logger.info(f"模型: {path}")
        return trainer

    # ==================== 第1步: 更新数据 ====================

    def update_data(self, target_date: str | None = None):
        """下载最新日线数据，追加到缓存。

        Args:
            target_date: 目标日期 YYYY-MM-DD，默认今天
        """
        if target_date is None:
            target_date = date.today().strftime("%Y-%m-%d")

        # 检查是否交易日
        if date.today().weekday() >= 5:
            logger.info("今天不是交易日，跳过数据更新")
            return

        start_ak = target_date.replace("-", "")
        end_ak = start_ak

        logger.info(f"[Step 1/5] 更新数据: {target_date}")

        success = 0
        for sym in self.symbols:
            # 检查缓存是否已有今日数据
            _, last_date = self.cache.get_daily_date_range(sym)
            if last_date is not None:
                last_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
                if last_str >= target_date:
                    continue  # 已有

            try:
                df = self.downloader.download_daily_ohlcv(
                    sym, start_ak, end_ak, adjust="qfq")
                if df is not None and not df.empty:
                    self.cache.update_daily(sym, df)
                    success += 1
            except Exception:
                pass

        logger.info(f"  数据更新: {success}/{len(self.symbols)} 只有新数据")

    # ==================== 第2步: 计算因子 ====================

    def compute_factors(self) -> pd.DataFrame:
        """用最新数据重算因子面板。"""
        logger.info("[Step 2/5] 计算因子...")

        tech_periods = {
            "momentum_periods": self.config["factors"]["technical"]["momentum_periods"],
            "volatility_periods": self.config["factors"]["technical"]["volatility_periods"],
            "volume_ratio_periods": self.config["factors"]["technical"]["volume_ratio_periods"],
        }

        all_factors = []
        for sym in self.symbols:
            df = self.cache.get_daily(sym)
            if df is None or len(df) < 60:
                continue

            df["日期"] = pd.to_datetime(df["日期"])
            factors = TechnicalFactors.compute_all(df, tech_periods)
            factors["date"] = df["日期"].values
            factors["symbol"] = sym
            all_factors.append(factors)

        panel = pd.concat(all_factors, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])

        # 因子处理
        processor = FactorProcessor()
        cfg_factors = self.config["factors"]
        processed = processor.process(panel, cfg_factors)

        # 关键：保存最新因子面板，供 portfolio_manager.rebalance() 使用
        os.makedirs("data/cache", exist_ok=True)
        processed.to_parquet("data/cache/factor_panel.parquet", index=False)
        logger.info(f"  因子面板: {len(processed)} 条, "
                     f"{len(processed.columns)-2} 个因子 "
                     f"[已保存，最新日期: "
                     f"{processed['date'].max().strftime('%Y-%m-%d')}]")
        return processed

    # ==================== 第3步: 生成信号 ====================

    def generate_signals(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        """用最新因子值生成交易信号。"""
        logger.info("[Step 3/5] 生成信号...")

        factor_cols = [c for c in factor_panel.columns
                       if c not in ["date", "symbol"]]
        feature_matrix = factor_panel.set_index(["date", "symbol"])[factor_cols]

        # 补齐缺失列（用0填充模型没见过的新因子）
        for col in self.trainer.feature_names:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0.0
        feature_matrix = feature_matrix[self.trainer.feature_names]

        signals = self.predictor.generate_signals(feature_matrix)

        # 取最新日期的信号
        latest_date = sorted(signals.index.get_level_values("date").unique())[-1]
        latest_signals = signals.xs(latest_date, level="date")

        top_picks = latest_signals[latest_signals["weight"] > 0]
        logger.info(f"  最新信号: {len(top_picks)} 只目标持仓 "
                     f"({pd.Timestamp(latest_date).strftime('%Y-%m-%d')})")

        self.current_signals = signals
        return signals

    # ==================== 第4步: 生成操作建议 ====================

    def generate_orders(self, signals: pd.DataFrame):
        """对比当前持仓和目标信号，生成买卖清单。"""
        logger.info("[Step 4/5] 生成操作建议...")

        # 最新信号
        latest_date = sorted(signals.index.get_level_values("date").unique())[-1]
        latest = signals.xs(latest_date, level="date")
        target_symbols = set(latest[latest["weight"] > 0].index)

        # 当前持仓
        current_holdings = set(self.broker.positions.keys())

        # 需卖出
        to_sell = current_holdings - target_symbols
        # 需买入
        to_buy = target_symbols - current_holdings
        # 继续持有
        to_hold = current_holdings & target_symbols

        return {
            "date": latest_date,
            "sell": sorted(to_sell),
            "buy": sorted(to_buy),
            "hold": sorted(to_hold),
            "target_count": len(target_symbols),
            "current_count": len(current_holdings),
        }

    # ==================== 第5步: 打印报告 ====================

    def print_daily_report(self, orders: dict):
        """打印每日操作报告。"""
        print(f"\n{'='*60}")
        print(f"  QuantLab 每日信号报告")
        print(f"  日期: {pd.Timestamp(orders['date']).strftime('%Y-%m-%d')}")
        print(f"  目标持仓: {orders['target_count']} 只 | "
              f"当前持仓: {orders['current_count']} 只")
        print(f"{'='*60}")

        if orders["sell"]:
            print(f"\n  [卖出] ({len(orders['sell'])} 只)")
            for sym in orders["sell"][:10]:
                name = self._get_stock_name(sym)
                pos = self.broker.positions.get(sym)
                shares = pos.shares if pos else "?"
                print(f"    {sym} {name:<8s} 清仓 x{shares}")
            if len(orders["sell"]) > 10:
                print(f"    ... 还有 {len(orders['sell'])-10} 只")

        if orders["buy"]:
            print(f"\n  [买入] ({len(orders['buy'])} 只)")
            for sym in orders["buy"][:10]:
                name = self._get_stock_name(sym)
                q = self._get_quote(sym)
                price_str = f" @ {q.price:.2f}" if q else ""
                print(f"    {sym} {name:<8s}{price_str}")
            if len(orders["buy"]) > 10:
                print(f"    ... 还有 {len(orders['buy'])-10} 只")

        if orders["hold"]:
            print(f"\n  [持有] ({len(orders['hold'])} 只)")
            for sym in orders["hold"][:5]:
                name = self._get_stock_name(sym)
                q = self._get_quote(sym)
                pnl = ""
                if q and sym in self.broker.positions:
                    pos = self.broker.positions[sym]
                    pnl_pct = (q.price / pos.avg_cost - 1) * 100
                    pnl = f" 成本{pos.avg_cost:.2f} 浮{'+' if pnl_pct>0 else ''}{pnl_pct:.1f}%"
                print(f"    {sym} {name:<8s}{pnl}")
            if len(orders["hold"]) > 5:
                print(f"    ... 还有 {len(orders['hold'])-5} 只")

        print(f"\n{'='*60}\n")

    def _get_stock_name(self, symbol: str) -> str:
        df = self.cache.get_daily(symbol)
        if df is not None:
            return ""
        return ""

    def _get_quote(self, symbol: str):
        try:
            return self.feed.fetch_single(symbol)
        except Exception:
            return None

    # ==================== 自动模式 ====================

    def run_auto(self):
        """自动模式：每天 15:30 自动执行。"""
        logger.info("自动模式启动 — 每个交易日 15:30 执行")

        while True:
            now = datetime.now()

            # 周末跳过
            if now.weekday() >= 5:
                logger.info(f"周末，下次检查: 周一")
                # 睡到周一
                days_to_monday = 7 - now.weekday()
                next_check = now.replace(hour=15, minute=30) + timedelta(days=days_to_monday)
                sleep_sec = (next_check - now).total_seconds()
                time.sleep(min(sleep_sec, 3600))
                continue

            # 等到 15:30
            target_time = now.replace(hour=15, minute=30, second=0)
            if now < target_time:
                wait = (target_time - now).total_seconds()
                logger.info(f"等待到 15:30 执行... ({wait/60:.0f}分钟)")
                time.sleep(min(wait, 600))  # 最多等10分钟就检查一次
                continue

            # 今天已经执行过？
            today_str = now.strftime("%Y-%m-%d")
            logger.info(f"===== {today_str} 每日流程开始 =====")

            try:
                self.update_data(today_str)
                factor_panel = self.compute_factors()
                signals = self.generate_signals(factor_panel)
                orders = self.generate_orders(signals)
                self.print_daily_report(orders)

                # 自动调仓（按最新信号，模拟成交）
                logger.info("自动调仓...")
                report = self.pm.rebalance()
                if report and report["actions"]:
                    logger.info(f"调仓完成: {len(report['actions'])} 笔操作")
                else:
                    logger.info("持仓已符合目标，无需调仓")

                # 记录今日盈亏
                self.pm.record_daily_pnl()

                # 重新训练模型（每周末）
                if now.weekday() == 4:  # 周五
                    logger.info("周末：重新训练模型...")
                    os.system(f'"{sys.executable}" main.py train '
                              f'--train-end {today_str} '
                              f'--test-start {(now + timedelta(days=30)).strftime("%Y-%m-%d")}')

            except Exception as e:
                logger.error(f"每日流程异常: {e}")

            # 等到明天
            time.sleep(3600)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="QuantLab 每日自动交易")
    parser.add_argument("--auto", action="store_true",
                        help="自动模式（每天15:30执行）")
    parser.add_argument("--status", action="store_true",
                        help="查看当前状态")
    parser.add_argument("--train", action="store_true",
                        help="执行前先重新训练模型")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="初始资金")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    setup_logger()
    runner = DailyRunner(args.config)

    if args.status:
        # 显示当前状态
        print(f"\n{'='*45}")
        print(f"  QuantLab 当前状态")
        print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  股票池: {len(runner.symbols)} 只")
        print(f"  持仓: {len(runner.broker.positions)} 只")
        total = runner.broker.get_total_value()
        print(f"  净值: {total/args.capital:.4f}")
        if runner.current_signals is not None:
            latest_date = sorted(
                runner.current_signals.index.get_level_values("date").unique()
            )[-1]
            print(f"  最新信号: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")
        print(f"{'='*45}\n")
        return

    if args.auto:
        runner.run_auto()
    else:
        # 手动执行一次
        today = date.today().strftime("%Y-%m-%d")
        print(f"\n  QuantLab 每日交易流程 [{today}]")
        print(f"  {'='*50}")

        if args.train:
            logger.info("重新训练模型...")
            os.system(f'"{sys.executable}" main.py train '
                      f'--train-end 2024-12-31 --test-start 2025-01-01')
            runner.trainer = runner._load_latest_model()

        runner.update_data(today)
        factor_panel = runner.compute_factors()  # 会自动保存最新因子面板
        signals = runner.generate_signals(factor_panel)
        orders = runner.generate_orders(signals)
        runner.print_daily_report(orders)

        # 自动调仓（用刚保存的最新信号）
        logger.info("执行自动调仓...")
        report = runner.pm.rebalance()
        if report and report["actions"]:
            logger.info(f"调仓完成: {len(report['actions'])} 笔操作")
        else:
            logger.info("持仓已符合目标，无需调仓")

        # 记录今日盈亏
        runner.pm.record_daily_pnl()
        runner.pm.print_status()

        print("  每日流程完成：数据已更新、信号已生成、调仓已执行。\n")


if __name__ == "__main__":
    main()
