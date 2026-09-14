"""
交易日志 — 记录每笔交易，生成统计报表，导出 CSV。
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import date, datetime
from loguru import logger


class TradeJournal:
    """交易日志本。

    记录:
        - 每笔订单提交
        - 每笔成交
        - 每日持仓快照
        - 资金流水
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.orders_log: list[dict] = []
        self.trades_log: list[dict] = []
        self.daily_snapshots: list[dict] = []

    # ==================== 记录 ====================

    def log_order(self, order):
        """记录订单提交。"""
        self.orders_log.append({
            "time": datetime.now().isoformat(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else order.side,
            "quantity": order.quantity,
            "order_type": order.order_type.value if hasattr(order.order_type, "value") else order.order_type,
            "limit_price": order.limit_price,
            "status": order.status.value if hasattr(order.status, "value") else order.status,
        })

    def log_trade(self, order):
        """记录成交。"""
        self.trades_log.append({
            "date": str(order.filled_date),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else order.side,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "amount": order.filled_price * order.filled_quantity,
            "commission": order.commission,
            "stamp_tax": order.stamp_tax,
            "slippage": order.slippage,
            "total_cost": order.commission + order.stamp_tax + order.slippage,
            "net_proceeds": (order.filled_price * order.filled_quantity -
                             order.commission - order.stamp_tax -
                             order.slippage) * (1 if order.side.value == "sell" else -1),
        })

    def log_daily_snapshot(self, dt: date, broker, portfolio):
        """记录每日快照。"""
        total_value = broker.get_total_value()
        self.daily_snapshots.append({
            "date": str(dt),
            "cash": broker.cash,
            "market_value": broker.get_market_value(),
            "total_value": total_value,
            "n_positions": len(broker.positions),
            "pnl": total_value - broker.initial_cash,
            "drawdown": portfolio.get_current_drawdown(),
        })

    # ==================== 统计 ====================

    def compute_statistics(self) -> dict:
        """计算交易统计。

        Returns:
            dict with: win_rate, avg_return, best_trade, worst_trade,
                      profit_factor, avg_holding_days, total_trades
        """
        if not self.trades_log:
            return {"total_trades": 0}

        sells = [t for t in self.trades_log if t["side"] == "sell"]

        if not sells:
            return {"total_trades": len(self.trades_log)}

        net_proceeds = [s["net_proceeds"] for s in sells]

        return {
            "total_trades": len(self.trades_log),
            "completed_round_trips": len(sells),
            "win_rate": sum(1 for x in net_proceeds if x > 0) / len(net_proceeds),
            "avg_pnl_per_trade": np.mean(net_proceeds),
            "total_pnl": sum(net_proceeds),
            "best_trade": max(net_proceeds),
            "worst_trade": min(net_proceeds),
            "profit_factor": (
                sum(x for x in net_proceeds if x > 0) /
                abs(sum(x for x in net_proceeds if x < 0))
            ) if sum(x for x in net_proceeds if x < 0) != 0 else float("inf"),
            "total_commission": sum(t["commission"] for t in self.trades_log),
            "total_stamp_tax": sum(t["stamp_tax"] for t in self.trades_log),
            "total_slippage": sum(t["slippage"] for t in self.trades_log),
            "total_cost": sum(t["total_cost"] for t in self.trades_log),
        }

    # ==================== 导出 ====================

    def export_csv(self, filename_prefix: str | None = None):
        """导出所有日志为 CSV。

        Args:
            filename_prefix: 文件名前缀
        """
        if filename_prefix is None:
            filename_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 成交记录
        if self.trades_log:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_trades.csv")
            pd.DataFrame(self.trades_log).to_csv(path, index=False,
                                                  encoding="utf-8-sig")
            logger.info(f"成交记录已导出: {path}")

        # 每日快照
        if self.daily_snapshots:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_daily.csv")
            pd.DataFrame(self.daily_snapshots).to_csv(path, index=False,
                                                       encoding="utf-8-sig")
            logger.info(f"每日快照已导出: {path}")

        # 订单记录
        if self.orders_log:
            path = os.path.join(self.output_dir,
                                f"{filename_prefix}_orders.csv")
            pd.DataFrame(self.orders_log).to_csv(path, index=False,
                                                  encoding="utf-8-sig")
            logger.info(f"订单记录已导出: {path}")

    def generate_report(self) -> str:
        """生成文本版交易报告。"""
        stats = self.compute_statistics()

        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("  模拟盘交易报告")
        lines.append("=" * 50)

        if stats["total_trades"] == 0:
            lines.append("  暂无交易记录")
            return "\n".join(lines)

        lines.append(f"  总交易次数:  {stats['total_trades']}")
        lines.append(f"  完整来回:    {stats.get('completed_round_trips', 0)}")
        lines.append(f"  胜率:        {stats.get('win_rate', 0)*100:.1f}%")
        lines.append(f"  总盈亏:      {stats.get('total_pnl', 0):,.0f} 元")
        lines.append(f"  平均盈亏:    {stats.get('avg_pnl_per_trade', 0):,.0f} 元/笔")
        lines.append(f"  最佳交易:    {stats.get('best_trade', 0):,.0f} 元")
        lines.append(f"  最差交易:    {stats.get('worst_trade', 0):,.0f} 元")
        lines.append(f"  利润因子:    {stats.get('profit_factor', 0):.2f}")
        lines.append(f"  总佣金:      {stats.get('total_commission', 0):,.0f} 元")
        lines.append(f"  总印花税:    {stats.get('total_stamp_tax', 0):,.0f} 元")
        lines.append(f"  总交易成本:  {stats.get('total_cost', 0):,.0f} 元")
        lines.append("=" * 50)

        return "\n".join(lines)
