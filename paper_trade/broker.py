"""
模拟券商 — 纸交易订单撮合引擎。
模拟真实的 A 股交易规则：T+1、整手买卖、涨跌停不可交易。
"""

import json
import os
from datetime import date, datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

from backtest.cost import TransactionCostModel


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单对象。"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int           # 目标股数
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_date: date | None = None
    filled_date: date | None = None
    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage: float = 0.0
    notes: str = ""


@dataclass
class Position:
    """持仓对象。"""
    symbol: str
    shares: int             # 当前持股数
    available_shares: int   # 可卖出股数（T+1规则：今天买的锁仓）
    avg_cost: float         # 平均成本价
    market_price: float     # 最新市价
    unrealized_pnl: float = 0.0  # 浮动盈亏
    locked_shares: int = 0       # 锁仓股数（T日买入，T+1才可卖）


class SimulatedBroker:
    """模拟券商 — 模拟 A 股交易执行。

    规则：
        1. 市价单 → 下一个交易日的开盘价成交
        2. 限价单 → 触价成交（high >= limit_price 买，low <= limit_price 卖）
        3. T+1：今天买入的股票明天才能卖出
        4. 最小交易单位：100股（一手）
        5. 涨跌停板不可交易
    """

    def __init__(self,
                 initial_cash: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5.0,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001,
                 lot_size: int = 100):
        """
        Args:
            initial_cash: 初始现金
            commission_rate: 佣金率
            min_commission: 最低佣金
            stamp_tax_rate: 印花税率（仅卖出）
            slippage_rate: 滑点率
            lot_size: 一手股数
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.lot_size = lot_size
        # 费率只有一份实现:回测的 backtest/cost.py。券商自己再写一遍 max()/0.0005
        # 迟早会和回测分叉(佣金 5 元下限就分叉过一次 —— 缩量预留按纯费率估会少留钱)。
        self.cost = TransactionCostModel(commission_rate, min_commission,
                                        stamp_tax_rate, slippage_rate)

        self.positions: dict[str, Position] = {}  # symbol → Position
        self.orders: list[Order] = []
        self.order_counter = 0
        self.trade_date: date | None = None

    # ==================== 下单 ====================

    def total_cost(self, amount: float, side: str = "buy") -> float:
        """单笔预估费用(元)。

        side 默认 "buy",因为下单前的资金预留(utils/sizing)只问买入那一侧;
        直接当 cost_fn 传给它就是正确用法。
        """
        return self.cost.total_cost(amount, side)

    def place_order(self, symbol: str, side: str, quantity: int,
                    order_type: str = "market",
                    limit_price: float | None = None) -> Order:
        """提交订单。

        Args:
            symbol: 6位股票代码
            side: "buy" / "sell"
            quantity: 目标股数（自动向下取整到整手）
            order_type: "market" / "limit"
            limit_price: 限价（限价单必填）

        Returns:
            Order 对象
        """
        self.order_counter += 1
        order_id = f"ORD{self.order_counter:06d}"

        # 整手取整
        quantity = (quantity // self.lot_size) * self.lot_size

        if quantity <= 0:
            order = Order(
                order_id=order_id, symbol=symbol,
                side=OrderSide(side), quantity=0,
                order_type=OrderType(order_type),
                limit_price=limit_price,
                status=OrderStatus.REJECTED,
                notes="数量不足1手，已拒单"
            )
            self.orders.append(order)
            return order

        # 卖空检查
        if side == "sell":
            pos = self.positions.get(symbol)
            if pos is None or pos.available_shares < quantity:
                avail = pos.available_shares if pos else 0
                order = Order(
                    order_id=order_id, symbol=symbol,
                    side=OrderSide.SELL, quantity=quantity,
                    order_type=OrderType(order_type),
                    limit_price=limit_price,
                    status=OrderStatus.REJECTED,
                    notes=f"可卖数量不足（需要{quantity}，可用{avail}），已拒单"
                )
                self.orders.append(order)
                return order

        order = Order(
            order_id=order_id, symbol=symbol,
            side=OrderSide(side), quantity=quantity,
            order_type=OrderType(order_type),
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            created_date=self.trade_date,
        )
        self.orders.append(order)
        logger.debug(f"订单已提交: {order_id} {side} {symbol} x{quantity}")
        return order

    def place_market_order(self, symbol: str, side: str,
                           quantity: int) -> Order:
        """提交市价单。"""
        return self.place_order(symbol, side, quantity, "market")

    def place_limit_order(self, symbol: str, side: str,
                          quantity: int, limit_price: float) -> Order:
        """提交限价单。"""
        return self.place_order(symbol, side, quantity, "limit", limit_price)

    # ==================== 撤单 ====================

    def cancel_order(self, order_id: str) -> bool:
        """撤销待成交订单。"""
        for order in self.orders:
            if order.order_id == order_id:
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
                    order.notes = "用户撤单"
                    logger.info(f"订单已撤销: {order_id}")
                    return True
                else:
                    logger.warning(f"订单 {order_id} 状态为 {order.status}，无法撤销")
                    return False
        logger.warning(f"未找到订单: {order_id}")
        return False

    # ==================== 每日撮合 ====================

    def process_daily(self, date: date, market_data: dict) -> list[Order]:
        """处理每日撮合（模拟收盘后执行）。

        流程:
            1. 将 T-1 日锁仓的股票解锁（T+1 到期）
            2. 逐个处理待成交订单
            3. 市价单以当日开盘价成交
            4. 限价单以触价逻辑判断是否成交

        Args:
            date: 当前交易日
            market_data: {symbol: {open, high, low, close, at_limit_up, at_limit_down}}

        Returns:
            当日成交的订单列表
        """
        self.trade_date = date
        filled_today = []

        # Step 1: 解锁 T+1 仓位
        for pos in self.positions.values():
            pos.available_shares += pos.locked_shares
            pos.locked_shares = 0

        # Step 2: 处理待成交订单
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue

            if order.symbol not in market_data:
                continue

            bar = market_data[order.symbol]

            # 涨跌停检查
            if order.side == OrderSide.BUY and bar.get("at_limit_up", False):
                continue  # 涨停买不到
            if order.side == OrderSide.SELL and bar.get("at_limit_down", False):
                continue  # 跌停卖不掉

            # 停牌检查
            if bar.get("volume", 0) <= 0:
                continue

            # --- 撮合逻辑 ---
            fill_price = None

            if order.order_type == OrderType.MARKET:
                # 市价单：以开盘价成交
                fill_price = bar["open"]
            elif order.order_type == OrderType.LIMIT:
                # 限价单：触价成交
                if order.side == OrderSide.BUY:
                    if bar["low"] <= order.limit_price:
                        fill_price = order.limit_price
                else:  # SELL
                    if bar["high"] >= order.limit_price:
                        fill_price = order.limit_price

            if fill_price is None or fill_price <= 0:
                continue

            # --- 执行成交 ---
            amount = fill_price * order.quantity
            commission = self.cost.commission(amount)
            stamp_tax = self.cost.stamp_tax(amount, order.side.value)
            slippage = self.cost.slippage(amount)
            total_cost = commission + stamp_tax + slippage

            if order.side == OrderSide.BUY:
                # 买入
                total_deduction = amount + total_cost
                if total_deduction > self.cash:
                    # 资金不足，部分成交（尽量少买1手）
                    reduced_qty = order.quantity - self.lot_size
                    if reduced_qty <= 0:
                        continue
                    order.quantity = reduced_qty
                    amount = fill_price * order.quantity
                    commission = self.cost.commission(amount)
                    slippage = self.cost.slippage(amount)
                    total_cost = commission + slippage
                    total_deduction = amount + total_cost
                    if total_deduction > self.cash:
                        continue

                self.cash -= total_deduction

                # 更新持仓
                if order.symbol not in self.positions:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        shares=0, available_shares=0,
                        avg_cost=0.0, market_price=fill_price,
                        locked_shares=0,
                    )

                pos = self.positions[order.symbol]
                total_shares = pos.shares + order.quantity
                # 平均成本价
                if total_shares > 0:
                    pos.avg_cost = ((pos.avg_cost * pos.shares) +
                                    (fill_price * order.quantity)) / total_shares
                pos.shares = total_shares
                pos.locked_shares += order.quantity  # T+1 锁定
                pos.market_price = fill_price

            else:  # SELL
                # 持仓可能已被之前的卖单清空
                if order.symbol not in self.positions:
                    continue

                self.cash += (amount - total_cost)

                pos = self.positions[order.symbol]
                pos.shares -= order.quantity
                pos.available_shares -= order.quantity
                if pos.shares <= 0:
                    del self.positions[order.symbol]

            # 更新订单
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.filled_date = date
            order.commission = commission
            order.stamp_tax = stamp_tax
            order.slippage = slippage
            order.status = OrderStatus.FILLED

            filled_today.append(order)

        # Step 3: 更新持仓市价
        for sym, pos in self.positions.items():
            if sym in market_data:
                pos.market_price = market_data[sym]["close"]
                pos.unrealized_pnl = (pos.market_price - pos.avg_cost) * pos.shares

        return filled_today

    # ==================== 查询 ====================

    def get_total_value(self) -> float:
        """总资产 = 现金 + 持仓市值。"""
        market_value = sum(
            p.shares * p.market_price for p in self.positions.values())
        return self.cash + market_value

    def get_market_value(self) -> float:
        """持仓总市值。"""
        return sum(p.shares * p.market_price for p in self.positions.values())

    def get_positions_summary(self) -> list[dict]:
        """持仓汇总。"""
        return [{
            "symbol": p.symbol,
            "shares": p.shares,
            "available": p.available_shares,
            "locked": p.locked_shares,
            "avg_cost": p.avg_cost,
            "market_price": p.market_price,
            "unrealized_pnl": p.unrealized_pnl,
            "pnl_pct": (p.market_price / p.avg_cost - 1) * 100
            if p.avg_cost > 0 else 0.0,
        } for p in self.positions.values()]

    def get_pending_orders(self) -> list[Order]:
        """待成交订单。"""
        return [o for o in self.orders if o.status == OrderStatus.PENDING]

    def get_today_pnl(self) -> float:
        """当日总盈亏。"""
        return self.get_total_value() - self.initial_cash

    # ==================== 状态持久化 ====================

    def save_state(self, path: str):
        """保存持仓与现金状态（每日批处理重启后恢复）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "trade_date": str(self.trade_date) if self.trade_date else None,
            "positions": {
                s: {
                    "shares": p.shares,
                    "available_shares": p.available_shares,
                    "locked_shares": p.locked_shares,
                    "avg_cost": p.avg_cost,
                    "market_price": p.market_price,
                }
                for s, p in self.positions.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"持仓状态已保存: {path}")

    def load_state(self, path: str):
        """从状态文件恢复持仓与现金（不存在则用初始资金）。"""
        if not os.path.exists(path):
            logger.info(f"状态文件不存在: {path}，使用初始资金")
            return
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.cash = float(state.get("cash", self.initial_cash))
        self.initial_cash = float(state.get("initial_cash", self.initial_cash))
        self.positions = {}
        for s, p in state.get("positions", {}).items():
            self.positions[s] = Position(
                symbol=s,
                shares=int(p["shares"]),
                available_shares=int(p.get("available_shares", p["shares"])),
                avg_cost=float(p.get("avg_cost", 0.0)),
                market_price=float(p.get("market_price", 0.0)),
                locked_shares=int(p.get("locked_shares", 0)),
            )
        logger.info(f"持仓状态已恢复: {len(self.positions)} 只持仓, "
                    f"现金 {self.cash:,.0f}")
