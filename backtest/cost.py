"""
交易成本模型 — A股真实的佣金、印花税、滑点。
"""

import pandas as pd
import numpy as np
from loguru import logger


class TransactionCostModel:
    """A股交易成本计算。

    成本构成：
        1. 佣金（买卖双向）：默认万三(0.03%)，最低5元
        2. 印花税（仅卖出）：万五(0.05%)，2023-08-28 起；更早是万10 → 用
           `stamp_tax_schedule` 按生效日分档，回测区间跨档时不给日期就是少收一半
        3. 滑点（买卖双向）：默认0.1%，按**成交额**计提现金，不是把成交价抬高。
           两者代数等价（rate×amount == rate×qty×price），只在整手取整上可能差
           不到一手（实测本仓库默认档 0/115 笔达到一整手，见 research/cost_audit.py）

    记账方式：买入 `cash -= amount + cost`、卖出 `cash += amount - cost`，
    成交价与盯市都用行情原始价，费用在成交日一次扣清，不摊到持有期。

    已知边界（不是 bug，是适用范围）：滑点是常数，不含冲击成本。当前默认档中位单笔
    9.1 万元，只占个股日成交额 0.068%（池内中位 1.33 亿元/日），常数 0.1% 合理偏保守；
    本金放大到千万级（中位单 ≈ 0.68% 日成交额）就必须让滑点随参与率走了。
    """

    def __init__(self,
                 commission_rate: float = 0.0003,
                 min_commission: float = 5.0,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001,
                 stamp_tax_schedule: dict | None = None):
        """
        Args:
            commission_rate: 佣金费率（默认万三）
            min_commission: 最低佣金（元）
            stamp_tax_rate: 印花税率（仅卖出，默认现行万五）
            slippage_rate: 滑点费率
            stamp_tax_schedule: {生效日: 税率}，如 {"1990-01-01": 0.001,
                                "2023-08-28": 0.0005}。取"生效日 ≤ 成交日"里最新的
                                那档；成交日早于全部生效日时用最早那档。不给就是不分档。
        """
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.stamp_schedule = sorted(
            ((pd.Timestamp(d), float(r)) for d, r in (stamp_tax_schedule or {}).items()),
            key=lambda x: x[0])

    def commission(self, trade_amount: float) -> float:
        """单边佣金。

        Args:
            trade_amount: 成交金额

        Returns:
            佣金（元）
        """
        fee = trade_amount * self.commission_rate
        return max(fee, self.min_commission)

    def stamp_rate_on(self, trade_date=None) -> float:
        """该成交日适用的印花税率（未分档或不给日期 → self.stamp_tax_rate）。"""
        if not self.stamp_schedule:
            return self.stamp_tax_rate
        if trade_date is None:
            return self.stamp_tax_rate
        t = pd.Timestamp(trade_date)
        if t < self.stamp_schedule[0][0]:
            return self.stamp_schedule[0][1]      # 早于第一档：按最早那档
        rate = self.stamp_tax_rate
        for d, r in self.stamp_schedule:
            if d <= t:
                rate = r
        return rate

    def stamp_tax(self, trade_amount: float, side: str, trade_date=None) -> float:
        """印花税（仅卖出）。

        Args:
            trade_amount: 成交金额
            side: "buy" / "sell"
            trade_date: 成交日，用于取分档税率（None = 不分档）

        Returns:
            印花税（元）
        """
        if side == "sell":
            return trade_amount * self.stamp_rate_on(trade_date)
        return 0.0

    def slippage(self, trade_amount: float) -> float:
        """滑点成本（按成交额计提，买卖双边同率）。

        Args:
            trade_amount: 成交金额

        Returns:
            滑点（元）
        """
        return trade_amount * self.slippage_rate

    def total_cost(self, trade_amount: float, side: str,
                   trade_date=None) -> float:
        """单笔交易总成本。

        Args:
            trade_amount: 成交金额
            side: "buy" / "sell"
            trade_date: 成交日（决定卖出用哪一档印花税）

        Returns:
            总成本（元）
        """
        return (self.commission(trade_amount) +
                self.stamp_tax(trade_amount, side, trade_date) +
                self.slippage(trade_amount))

    def apply_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """对交易记录批量计算成本。

        Args:
            trades_df: DataFrame with columns [amount, side]，有 date 列则按分档取印花税

        Returns:
            增加 'commission', 'stamp_tax', 'slippage', 'total_cost' 列
        """
        df = trades_df.copy()
        dates = df["date"] if "date" in df.columns else [None] * len(df)
        df["commission"] = df["amount"].apply(self.commission)
        df["stamp_tax"] = [self.stamp_tax(r["amount"], r["side"], d)
                           for (_, r), d in zip(df.iterrows(), dates)]
        df["slippage"] = df["amount"].apply(self.slippage)
        df["total_cost"] = (df["commission"] + df["stamp_tax"] +
                            df["slippage"])
        return df

    def effective_cost_rate(self, side: str = "buy", trade_date=None) -> float:
        """单边有效成本率（不含 5 元佣金下限，用于资金预留的近似）。"""
        base = self.commission_rate + self.slippage_rate
        if side == "sell":
            base += self.stamp_rate_on(trade_date)
        return base

    def round_lot(self, quantity: int, lot_size: int = 100) -> int:
        """将股数向下取整到整手数。

        Args:
            quantity: 目标股数
            lot_size: 一手股数（A股=100）

        Returns:
            整手股数
        """
        return (quantity // lot_size) * lot_size

    def __repr__(self) -> str:
        return (f"TransactionCost(佣金={self.commission_rate*10000:.0f}‱, "
                f"最低{self.min_commission}元, "
                f"印花税={self.stamp_tax_rate*10000:.0f}‱, "
                f"滑点={self.slippage_rate*100:.2f}%)")
