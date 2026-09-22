#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
持仓管理器 — 保存持仓、自动调仓、每日盈亏记录。

用法:
    python portfolio_manager.py status        # 查看持仓 + 今日盈亏
    python portfolio_manager.py rebalance     # 按最新模型信号调仓
    python portfolio_manager.py pnl           # 查看每日盈亏历史
    python portfolio_manager.py pnl --days 30 # 最近30天盈亏
    python portfolio_manager.py pnl --plot    # 生成盈亏曲线HTML
    python portfolio_manager.py reset         # 重置（重新从上次模拟盘导入）

数据文件:
    portfolio_state.json    # 当前持仓状态 (现金+股票)
    logs/daily_pnl.csv      # 每日盈亏历史
"""

import os
import sys
import json
import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from data.cache import CacheManager
from models.trainer import LightGBMTrainer
from models.predictor import Predictor
from live import SinaQuoteFeed
from backtest.cost import TransactionCostModel
from utils.logger import setup_logger

STATE_FILE = "portfolio_state.json"
PNL_FILE = "logs/daily_pnl.csv"
LOT_SIZE = 100


def _norm_symbol(s) -> str:
    """规范化股票代码为6位字符串。"""
    s = str(s)
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)


class PortfolioManager:
    """持仓管理器。

    职责:
        1. 持久化持仓状态 (portfolio_state.json)
        2. 按模型信号自动调仓（含交易成本、T+1锁仓、整手）
        3. 每日盯市，记录盈亏 (logs/daily_pnl.csv)
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
        self.config = config

        self.cost = TransactionCostModel(
            commission_rate=config["market"]["commission_rate"],
            min_commission=config["market"]["min_commission"],
            stamp_tax_rate=config["market"]["stamp_tax_rate"],
            slippage_rate=config["market"]["slippage_rate"],
        )
        self.feed = SinaQuoteFeed()
        self.cache = CacheManager(config["cache"]["directory"])

        self.state = self._load_state()

    # ==================== 状态持久化 ====================

    def _load_state(self) -> dict:
        """加载持仓状态；不存在则从上次模拟盘导入。"""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(f"持仓状态已加载: 现金={state['cash']:,.0f}, "
                         f"持仓={len(state['positions'])}只")
            return state

        logger.info("未找到持仓状态，从上次模拟盘导入...")
        return self._import_from_paper_trade()

    def _import_from_paper_trade(self) -> dict:
        """从上次模拟盘结果重建初始持仓。"""
        trades_file = "logs/paper_trade_result_trades.csv"
        daily_file = "logs/paper_trade_result_daily.csv"

        positions = {}
        if os.path.exists(trades_file):
            trades = pd.read_csv(trades_file)
            # 买入记录用于计算加权平均成本
            buy_records = {}
            for _, t in trades.iterrows():
                sym = _norm_symbol(t["symbol"])
                qty = int(t["quantity"])
                if t["side"] == "buy":
                    positions[sym] = positions.get(sym, 0) + qty
                    rec = buy_records.setdefault(sym, {"shares": 0, "spent": 0.0})
                    rec["shares"] += qty
                    rec["spent"] += float(t["amount"]) + float(t["total_cost"])
                else:
                    positions[sym] = positions.get(sym, 0) - qty
            positions = {k: v for k, v in positions.items() if v > 0}
            logger.info(f"从交易记录导入 {len(positions)} 只持仓")

        # 现金：上次模拟盘最后一天的现金
        cash = 0.0
        baseline_value = 0.0
        if os.path.exists(daily_file):
            daily = pd.read_csv(daily_file)
            cash = float(daily["cash"].iloc[-1])
            baseline_value = float(daily["total_value"].iloc[-1])
            logger.info(f"上次模拟盘: 现金={cash:,.0f} 元, "
                         f"总资产={baseline_value:,.0f} 元")

        state = {
            "initial_capital": 1_000_000,
            "baseline_value": baseline_value,  # 导入时的总资产（盈亏基准）
            "cash": cash,
            "positions": {
                sym: {
                    "shares": qty,
                    "avg_cost": round(buy_records[sym]["spent"] /
                                       buy_records[sym]["shares"], 3)
                    if sym in buy_records and buy_records[sym]["shares"] > 0
                    else 0.0,
                    "locked": 0,
                }
                for sym, qty in positions.items()
            },
            "last_rebalance": None,
            "last_pnl_date": None,
        }
        self._save_state(state)
        return state

    def _save_state(self, state: dict | None = None):
        """保存持仓状态。"""
        if state is None:
            state = self.state
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.debug("持仓状态已保存")

    # ==================== 价格获取 ====================

    def _get_prices(self, symbols: list[str]) -> dict:
        """获取股票价格：优先实时价，其次缓存最新收盘价。

        Returns:
            {symbol: {"price": float, "source": "live"/"cached"}}
        """
        prices = {}
        # 实时
        if symbols:
            try:
                quotes = self.feed.fetch(symbols)
                for sym, q in quotes.items():
                    if q.price > 0:
                        prices[sym] = {"price": q.price, "source": "live"}
            except Exception as e:
                logger.warning(f"实时行情获取失败: {e}")

        # 缓存兜底
        for sym in symbols:
            if sym not in prices:
                df = self.cache.get_daily(sym)
                if df is not None and not df.empty:
                    last_close = float(df["收盘"].iloc[-1])
                    prices[sym] = {"price": last_close, "source": "cached"}

        return prices

    # ==================== 持仓查询 ====================

    def get_positions_with_pnl(self) -> pd.DataFrame:
        """持仓明细 + 实时盈亏。"""
        if not self.state["positions"]:
            return pd.DataFrame()

        symbols = list(self.state["positions"].keys())
        prices = self._get_prices(symbols)

        rows = []
        for sym, pos in self.state["positions"].items():
            p = prices.get(sym)
            if not p:
                continue
            price = p["price"]
            shares = pos["shares"]
            market_value = price * shares
            avg_cost = pos.get("avg_cost", 0) or 0

            pnl = (price - avg_cost) * shares if avg_cost > 0 else 0
            pnl_pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0

            rows.append({
                "symbol": sym,
                "shares": shares,
                "avg_cost": round(avg_cost, 3),
                "price": price,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "source": p["source"],
            })

        df = pd.DataFrame(rows).sort_values("market_value", ascending=False)
        return df

    def get_summary(self) -> dict:
        """账户摘要。"""
        df = self.get_positions_with_pnl()
        market_value = df["market_value"].sum() if not df.empty else 0
        total_value = self.state["cash"] + market_value
        initial = self.state["initial_capital"]

        return {
            "cash": self.state["cash"],
            "market_value": market_value,
            "total_value": total_value,
            "initial_capital": initial,
            "cumulative_pnl": total_value - initial,
            "cumulative_return": (total_value / initial - 1) if initial else 0,
            "n_positions": len(df),
            "positions": df,
        }

    # ==================== 自动调仓 ====================

    def rebalance(self, top_k: int | None = None, dry_run: bool = False) -> dict:
        """按最新模型信号调仓。

        流程:
            1. 生成最新信号（模型预测排名选股）
            2. 卖出不在目标中的持仓
            3. 对目标股票按目标市值等权买入
            4. 记录交易，更新状态

        Args:
            top_k: 目标持仓数（默认取配置）
            dry_run: 只打印不执行

        Returns:
            调仓报告 dict
        """
        config = self.config
        # 分数带位策略开启时不能走这条链路:下面的算式是
        # `current_value / len(target_symbols)` 的等权摊派,既不看建仓线也不看
        # 单票上限 —— 静默按它下单会把 5% 基准变成 1/N,策略等于没生效。
        from utils.position_policy import policy_from_config
        if policy_from_config(config) is not None:
            raise RuntimeError(
                "position_policy.enabled=true,portfolio_manager 的调仓是等权摊派,"
                "不实现分数带位规则。策略链路请走 python live/run_daily.py"
                "(--simulate / --qmt),策略表现请用 python main.py backtest 评估。")
        if top_k is None:
            top_k = config["backtest"]["max_positions"]

        # 1. 生成信号
        logger.info("生成最新信号...")
        trainer = self._load_model()
        predictor = Predictor(trainer, top_k=top_k,
                              position_sizing="equal_weight")

        factor_panel = pd.read_parquet("data/cache/factor_panel.parquet")
        factor_panel["date"] = pd.to_datetime(factor_panel["date"])
        factor_cols = [c for c in factor_panel.columns
                       if c not in ["date", "symbol"]]
        X = factor_panel.set_index(["date", "symbol"])[factor_cols]

        signals = predictor.generate_signals(X)
        latest_date = sorted(signals.index.get_level_values("date").unique())[-1]
        latest = signals.xs(latest_date, level="date")
        target_symbols = set(latest[latest["weight"] > 0].index)

        logger.info(f"信号日期: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}, "
                     f"目标: {len(target_symbols)} 只")

        # 2. 价格
        all_symbols = list(set(target_symbols) |
                           set(self.state["positions"].keys()))
        prices = self._get_prices(all_symbols)
        if not prices:
            logger.error("无法获取价格，调仓取消")
            return {}

        # 3. 计算目标市值（等权）
        current_value = self.state["cash"] + sum(
            self.state["positions"].get(s, {}).get("shares", 0) *
            prices.get(s, {}).get("price", 0)
            for s in self.state["positions"]
        )
        target_value_per_stock = current_value / len(target_symbols)

        trades = []   # 成交记录
        action_list = []  # 操作清单

        # --- 卖出：非目标持仓 + 目标中超配的 ---
        for sym, pos in list(self.state["positions"].items()):
            p = prices.get(sym)
            if not p:
                continue
            price = p["price"]
            shares = pos["shares"]
            locked = pos.get("locked", 0)
            available = shares - locked
            if available <= 0:
                continue

            if sym not in target_symbols:
                # 清仓
                if not dry_run:
                    self._execute_sell(sym, available, price, trades)
                action_list.append(("SELL", sym, available, price, "清仓(不在目标)"))
            else:
                # 超配部分卖出
                current_mv = shares * price
                if current_mv > target_value_per_stock * 1.05:
                    excess_shares = self.cost.round_lot(
                        int((current_mv - target_value_per_stock) / price))
                    excess_shares = min(excess_shares, available)
                    if excess_shares >= LOT_SIZE:
                        if not dry_run:
                            self._execute_sell(sym, excess_shares, price, trades)
                        action_list.append(("SELL", sym, excess_shares, price,
                                            "减仓至等权"))

        # --- 买入：目标中的 ---
        for sym in sorted(target_symbols):
            p = prices.get(sym)
            if not p:
                continue
            price = p["price"]
            current_shares = self.state["positions"].get(sym, {}).get("shares", 0)
            current_mv = current_shares * price

            if current_mv < target_value_per_stock * 0.95:
                buy_value = target_value_per_stock - current_mv
                shares_to_buy = self.cost.round_lot(int(buy_value / price))
                if shares_to_buy >= LOT_SIZE:
                    if not dry_run:
                        self._execute_buy(sym, shares_to_buy, price, trades)
                    action_list.append(("BUY", sym, shares_to_buy, price,
                                        "建仓/加仓至等权"))

        # 4. 保存状态
        if not dry_run:
            self.state["last_rebalance"] = date.today().strftime("%Y-%m-%d")
            self._save_state()

        report = {
            "date": latest_date,
            "target_count": len(target_symbols),
            "actions": action_list,
            "trades": trades,
            "dry_run": dry_run,
        }
        return report

    def _execute_sell(self, sym: str, shares: int, price: float, trades: list):
        """执行卖出（扣印花税+佣金+滑点）。"""
        amount = price * shares
        cost = self.cost.total_cost(amount, "sell")
        proceeds = amount - cost

        self.state["cash"] += proceeds
        pos = self.state["positions"][sym]
        pos["shares"] -= shares
        pos["locked"] = max(pos.get("locked", 0) - shares, 0)
        if pos["shares"] <= 0:
            del self.state["positions"][sym]

        trades.append({
            "date": date.today().strftime("%Y-%m-%d"),
            "symbol": sym, "side": "sell", "shares": shares,
            "price": price, "amount": amount, "cost": cost,
        })

    def _execute_buy(self, sym: str, shares: int, price: float, trades: list):
        """执行买入（扣佣金+滑点，T+1锁仓）。"""
        amount = price * shares
        cost = self.cost.total_cost(amount, "buy")
        total = amount + cost

        # 资金不足则减少股数
        if total > self.state["cash"]:
            shares = self.cost.round_lot(
                int((self.state["cash"] * 0.99 - self.cost.min_commission) / price))
            if shares < LOT_SIZE:
                return
            amount = price * shares
            cost = self.cost.total_cost(amount, "buy")
            total = amount + cost

        self.state["cash"] -= total

        pos = self.state["positions"].get(sym, {
            "shares": 0, "avg_cost": 0.0, "locked": 0})
        total_shares = pos["shares"] + shares
        if total_shares > 0:
            pos["avg_cost"] = ((pos["avg_cost"] * pos["shares"]) +
                               (price * shares)) / total_shares
        pos["shares"] = total_shares
        pos["locked"] = pos.get("locked", 0) + shares  # T+1
        self.state["positions"][sym] = pos

        trades.append({
            "date": date.today().strftime("%Y-%m-%d"),
            "symbol": sym, "side": "buy", "shares": shares,
            "price": price, "amount": amount, "cost": cost,
        })

    def _load_model(self) -> LightGBMTrainer:
        saved_dir = "models/saved"
        models = sorted([f for f in os.listdir(saved_dir)
                         if f.endswith(".txt")])
        if not models:
            raise FileNotFoundError("未找到已训练模型")
        trainer = LightGBMTrainer(model_type=self.config["model"]["type"])
        trainer.load(os.path.join(saved_dir, models[-1]))
        return trainer

    # ==================== 每日盈亏 ====================

    def record_daily_pnl(self) -> dict:
        """记录今日盈亏（用最新价格盯市）。

        Returns:
            今日盈亏 dict
        """
        summary = self.get_summary()

        today = date.today().strftime("%Y-%m-%d")
        prev_value = None

        # 读取历史
        if os.path.exists(PNL_FILE):
            df = pd.read_csv(PNL_FILE)
            prev_value = float(df["total_value"].iloc[-1])
        else:
            # 首次记录：以上次模拟盘终值为基准
            prev_value = self.state.get("baseline_value",
                                        self.state["initial_capital"])

        daily_pnl = summary["total_value"] - prev_value
        daily_return = daily_pnl / prev_value if prev_value else 0

        row = {
            "date": today,
            "total_value": round(summary["total_value"], 2),
            "cash": round(summary["cash"], 2),
            "market_value": round(summary["market_value"], 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return": round(daily_return, 6),
            "cumulative_pnl": round(summary["cumulative_pnl"], 2),
            "cumulative_return": round(summary["cumulative_return"], 6),
            "n_positions": summary["n_positions"],
        }

        # 去重（同一天多次记录只保留最后一次）
        if os.path.exists(PNL_FILE):
            df = pd.read_csv(PNL_FILE)
            df = df[df["date"] != today]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
            os.makedirs("logs", exist_ok=True)
        df.to_csv(PNL_FILE, index=False, encoding="utf-8-sig")

        logger.info(f"今日盈亏: {daily_pnl:+,.0f} 元 "
                     f"({daily_return*100:+.2f}%), "
                     f"累计: {summary['cumulative_pnl']:+,.0f} 元")
        return row

    def get_pnl_history(self, days: int | None = None) -> pd.DataFrame:
        """读取每日盈亏历史。"""
        if not os.path.exists(PNL_FILE):
            return pd.DataFrame()
        df = pd.read_csv(PNL_FILE)
        df["date"] = pd.to_datetime(df["date"])
        if days:
            df = df.tail(days)
        return df

    # ==================== 展示 ====================

    def print_status(self):
        """打印持仓状态。"""
        summary = self.get_summary()
        print(f"\n{'='*58}")
        print(f"  QuantLab 持仓状态  [{date.today()}]")
        print(f"{'='*58}")
        print(f"  总资产:   {summary['total_value']:>12,.0f} 元")
        print(f"  现金:     {summary['cash']:>12,.0f} 元")
        print(f"  持仓市值: {summary['market_value']:>12,.0f} 元")
        print(f"  累计盈亏: {summary['cumulative_pnl']:>+11,.0f} 元 "
              f"({summary['cumulative_return']*100:+.2f}%)")
        print(f"  持仓数量: {summary['n_positions']} 只")
        print(f"{'-'*58}")

        df = summary["positions"]
        if df.empty:
            print("  (空仓)")
        else:
            print(f"  {'代码':<8}{'股数':>8}{'成本':>10}{'现价':>10}"
                  f"{'市值':>12}{'盈亏%':>9}")
            for _, r in df.head(15).iterrows():
                print(f"  {r['symbol']:<8}{r['shares']:>8}{r['avg_cost']:>10.2f}"
                      f"{r['price']:>10.2f}{r['market_value']:>12,.0f}"
                      f"{r['pnl_pct']:>+8.1f}%")
            if len(df) > 15:
                print(f"  ... 还有 {len(df)-15} 只")

        total_pnl = df["pnl"].sum() if not df.empty else 0
        print(f"{'-'*58}")
        print(f"  持仓浮盈合计: {total_pnl:+,.0f} 元")
        print(f"{'='*58}\n")

    def print_pnl(self, days: int = 10):
        """打印每日盈亏历史。"""
        df = self.get_pnl_history(days)
        if df.empty:
            print("\n  暂无每日盈亏记录。先运行 record 或 rebalance。\n")
            return

        print(f"\n{'='*58}")
        print(f"  每日盈亏记录 (最近 {len(df)} 天)")
        print(f"{'='*58}")
        print(f"  {'日期':<12}{'总资产':>12}{'当日盈亏':>12}{'当日%':>9}"
              f"{'累计%':>9}")
        for _, r in df.iterrows():
            d = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            print(f"  {d:<12}{r['total_value']:>12,.0f}"
                  f"{r['daily_pnl']:>+11,.0f}"
                  f"{r['daily_return']*100:>+8.2f}%"
                  f"{r['cumulative_return']*100:>+8.2f}%")

        print(f"{'-'*58}")

        last = df.iloc[-1]
        print(f"  最新: 总资产 {last['total_value']:,.0f} 元 | "
              f"当日 {last['daily_pnl']:+,.0f} 元 | "
              f"累计 {last['cumulative_return']*100:+.2f}%")
        print(f"{'='*58}\n")

    def plot_pnl(self):
        """生成每日盈亏曲线 HTML。"""
        df = self.get_pnl_history()
        if df.empty or len(df) < 2:
            print("数据不足，无法绘图")
            return

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35],
                subplot_titles=("累计净值", "每日盈亏"),
            )

            fig.add_trace(go.Scatter(
                x=df["date"], y=df["total_value"],
                mode="lines", name="总资产",
                line=dict(color="#4ec9b0", width=2),
            ), row=1, col=1)

            colors = ["#f44747" if x < 0 else "#4ec9b0"
                      for x in df["daily_pnl"]]
            fig.add_trace(go.Bar(
                x=df["date"], y=df["daily_pnl"],
                name="每日盈亏", marker_color=colors,
            ), row=2, col=1)

            fig.update_layout(
                title="QuantLab 每日盈亏",
                template="plotly_dark", height=600,
                hovermode="x unified",
            )
            os.makedirs("reports", exist_ok=True)
            path = f"reports/pnl_{date.today().strftime('%Y%m%d')}.html"
            fig.write_html(path)
            print(f"盈亏曲线已生成: {path}")
            os.startfile(os.path.abspath(path))
        except Exception as e:
            logger.error(f"绘图失败: {e}")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="QuantLab 持仓管理器")
    sub = parser.add_subparsers(dest="command", help="子命令")

    p_status = sub.add_parser("status", help="查看持仓+盈亏")
    p_status.add_argument("--record", action="store_true",
                          help="先记录今日盈亏再显示")

    p_reb = sub.add_parser("rebalance", help="按最新信号调仓")
    p_reb.add_argument("--top-k", type=int, default=None,
                       help="目标持仓数")
    p_reb.add_argument("--dry-run", action="store_true",
                       help="只预览不执行")

    p_pnl = sub.add_parser("pnl", help="查看每日盈亏")
    p_pnl.add_argument("--days", type=int, default=10,
                       help="显示最近N天")
    p_pnl.add_argument("--plot", action="store_true",
                       help="生成盈亏曲线图")

    sub.add_parser("reset", help="重置持仓（从上次模拟盘导入）")

    args = parser.parse_args()

    setup_logger()
    manager = PortfolioManager()

    if args.command == "status":
        if args.record:
            manager.record_daily_pnl()
        manager.print_status()
    elif args.command == "rebalance":
        report = manager.rebalance(top_k=args.top_k, dry_run=args.dry_run)
        if report and report["actions"]:
            mode = "【预览】" if args.dry_run else "【执行】"
            print(f"\n{mode} 调仓清单 ({len(report['actions'])} 笔操作)")
            print(f"{'='*58}")
            for side, sym, shares, price, reason in report["actions"]:
                tag = "买" if side == "BUY" else "卖"
                print(f"  [{tag}] {sym} x{shares:>5} @ {price:>8.2f}  {reason}")
            print(f"{'='*58}\n")
            if not args.dry_run:
                manager.record_daily_pnl()
        else:
            print("\n  无调仓操作（持仓已符合目标）\n")
    elif args.command == "pnl":
        if args.plot:
            manager.plot_pnl()
        manager.print_pnl(args.days)
    elif args.command == "reset":
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print("持仓状态已重置")
        manager = PortfolioManager()
        manager.print_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
