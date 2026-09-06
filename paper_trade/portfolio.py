"""
持仓跟踪 — 盯市估值、风险监控、每日收益记录。
"""

import pandas as pd
import numpy as np
from datetime import date
from loguru import logger


class PortfolioTracker:
    """持仓跟踪器。

    功能:
        - 每日 Mark-to-Market 盯市
        - 收益率序列记录
        - 风险指标实时监控（持仓集中度、回撤等）
        - 风控告警
    """

    def __init__(self, initial_capital: float = 1_000_000,
                 risk_limits: dict | None = None):
        """
        Args:
            initial_capital: 初始资金
            risk_limits: 风控限制
                - max_position_pct: 单股票最大仓位（默认20%）
                - max_drawdown_pct: 触发止损的最大回撤（默认20%）
                - max_sector_pct: 单行业最大仓位（默认40%）
        """
        self.initial_capital = initial_capital

        # 默认风控参数
        self.risk_limits = {
            "max_position_pct": 0.20,
            "max_drawdown_pct": 0.20,
            "max_sector_pct": 0.40,
        }
        if risk_limits:
            self.risk_limits.update(risk_limits)

        # 每日净值记录
        self.daily_values: dict[date, float] = {}
        self.daily_returns: dict[date, float] = {}
        self.peak_value = initial_capital

    # ==================== 每日更新 ====================

    def update(self, dt: date, broker):
        """记录当日资产净值。

        Args:
            dt: 日期
            broker: SimulatedBroker 实例
        """
        total_value = broker.get_total_value()
        self.daily_values[dt] = total_value

        # 计算日收益率
        prev_dates = sorted(self.daily_values.keys())
        if len(prev_dates) >= 2:
            prev_value = self.daily_values[prev_dates[-2]]
            if prev_value > 0:
                self.daily_returns[dt] = total_value / prev_value - 1
        else:
            self.daily_returns[dt] = 0.0

        # 更新峰值
        if total_value > self.peak_value:
            self.peak_value = total_value

    # ==================== 风险检查 ====================

    def check_risk_limits(self, broker) -> list[str]:
        """检查风控指标，返回告警列表。

        Args:
            broker: SimulatedBroker 实例

        Returns:
            告警消息列表
        """
        alerts = []
        total_value = broker.get_total_value()
        positions = broker.get_positions_summary()

        if total_value <= 0:
            alerts.append("⚠️ 总资产 <= 0！")

        # 1. 单股票集中度
        for pos in positions:
            position_pct = (pos["shares"] * pos["market_price"]) / total_value
            if position_pct > self.risk_limits["max_position_pct"]:
                alerts.append(
                    f"⚠️ {pos['symbol']} 仓位超限: {position_pct:.1%} "
                    f"(限制 {self.risk_limits['max_position_pct']:.0%})"
                )

        # 2. 回撤
        if total_value < self.peak_value:
            drawdown = 1 - total_value / self.peak_value
            if drawdown > self.risk_limits["max_drawdown_pct"]:
                alerts.append(
                    f"🚨 回撤超限: {drawdown:.1%} "
                    f"(限制 {self.risk_limits['max_drawdown_pct']:.0%})"
                )

        # 3. 持仓数量异常
        if len(positions) > 50:
            alerts.append(f"⚠️ 持仓数量过多: {len(positions)} 只")

        return alerts

    # ==================== 报表 ====================

    def get_returns_series(self) -> pd.Series:
        """返回日收益率 Series。"""
        if not self.daily_returns:
            return pd.Series(dtype=float)
        return pd.Series(self.daily_returns).sort_index()

    def get_equity_curve(self) -> pd.Series:
        """返回净值曲线。"""
        if not self.daily_values:
            return pd.Series(dtype=float)
        equity = pd.Series(self.daily_values).sort_index()
        return equity / self.initial_capital

    def get_current_drawdown(self) -> float:
        """当前回撤。"""
        if not self.daily_values:
            return 0.0
        current = max(self.daily_values.values())
        drawdown = 1 - current / self.peak_value
        return max(drawdown, 0.0)

    def summary(self) -> dict:
        """生成持仓摘要。"""
        equity = self.get_equity_curve()
        if len(equity) < 2:
            return {}

        returns = self.get_returns_series()
        return {
            "total_return": equity.iloc[-1] - 1,
            "annual_vol": float(returns.std() * np.sqrt(252)),
            "max_drawdown": self.get_current_drawdown(),
            "current_value": self.daily_values.get(
                max(self.daily_values.keys()), 0) if self.daily_values else 0,
        }
