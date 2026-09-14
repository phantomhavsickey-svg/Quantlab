# -*- coding: utf-8 -*-
"""
QMT 真实柜台适配器 — 与 SimulatedBroker 同接口（方法签名对齐），
引擎层无需改动即可在模拟盘/实盘之间切换。

使用前提:
    1. 在支持 QMT 的券商开户并开通程序化交易权限
       （程序化交易须按监管要求完成报备）
    2. 安装券商提供的 miniQMT 客户端并保持登录
    3. config.yaml 的 live.qmt 配置:
         mini_qmt_path: 客户端用户数据目录（如 D:\\国金QMT\\userdata_mini）
         account_id: 资金账号
    4. 把客户端自带的 xtquant 目录加入 Python 环境
       （或 pip install xtquant，部分券商提供离线 whl）
"""

from loguru import logger

try:
    from xtquant import xttrader, xtconstant
    from xtquant.xttype import StockAccount
    HAS_XTQUANT = True
except ImportError:
    HAS_XTQUANT = False
    xttrader = xtconstant = None
    StockAccount = None


class QMTBroker:
    """QMT 柜台适配器。

    实现 SimulatedBroker 的同名方法：
        place_market_order / place_limit_order / cancel_order /
        get_positions_summary / get_cash / get_total_value
    """

    def __init__(self, qmt_config: dict, capital: float | None = None):
        if not HAS_XTQUANT:
            raise RuntimeError(
                "未安装 xtquant。请先安装券商 miniQMT 客户端，"
                "并把客户端自带的 xtquant 加入 Python 环境"
                "（或 pip install xtquant），"
                "然后在 config.yaml 的 live.qmt 中配置路径和账号。")
        path = qmt_config.get("mini_qmt_path", "")
        account_id = str(qmt_config.get("account_id", ""))
        if not path or not account_id:
            raise RuntimeError(
                "live.qmt 未配置 mini_qmt_path / account_id，"
                "拒绝启动真实柜台。")
        self.trader = xttrader.XtQuantTrader(path, 1)
        self.trader.start()
        connect_ok = self.trader.connect()
        if connect_ok != 0:
            raise RuntimeError(
                f"QMT 连接失败（错误码 {connect_ok}），请确认客户端已登录")
        self.account = StockAccount(account_id)
        self.cash = 0.0

    # ==================== 代码转换 ====================

    @staticmethod
    def _with_exchange(symbol: str) -> str:
        """6位代码 → 带交易所后缀: 600519 → 600519.SH。"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(("60", "68", "9")):
            return f"{symbol}.SH"
        if symbol.startswith(("4", "8")):
            return f"{symbol}.BJ"
        return f"{symbol}.SZ"

    # ==================== 查询 ====================

    def get_cash(self) -> float:
        """可用现金。"""
        asset = self.trader.query_stock_asset(self.account)
        if asset is None:
            logger.warning("查询资金失败，返回缓存值")
            return self.cash
        self.cash = float(asset.cash)
        return self.cash

    def get_positions_summary(self) -> list[dict]:
        """持仓汇总（与 SimulatedBroker 同构）。"""
        rows = []
        for pos in (self.trader.query_stock_positions(self.account) or []):
            rows.append({
                "symbol": pos.stock_code.split(".")[0],
                "shares": int(pos.volume),
                "available": int(pos.can_use_volume),
                "avg_cost": float(pos.open_price),
                "market_price": (float(pos.market_value / pos.volume)
                                 if pos.volume else 0.0),
            })
        return rows

    def get_total_value(self) -> float:
        """总资产。"""
        asset = self.trader.query_stock_asset(self.account)
        if asset is None:
            return self.cash + sum(
                p["market_price"] * p["shares"]
                for p in self.get_positions_summary())
        return float(asset.total_asset)

    # ==================== 下单 ====================

    def place_market_order(self, symbol: str, side: str, quantity: int,
                           ref_price: float | None = None) -> int:
        """下单（默认按参考价的限价单；未成交需人工处理或撤单重下）。

        Returns:
            订单编号（失败返回 -1）
        """
        stock_code = self._with_exchange(symbol)
        order_type = (xtconstant.STOCK_BUY if side == "buy"
                      else xtconstant.STOCK_SELL)
        price = float(ref_price or 0.0)
        order_id = self.trader.order_stock(
            self.account, stock_code, order_type, int(quantity),
            xtconstant.FIX_PRICE, price, "quantlab", "rebalance")
        if order_id < 0:
            logger.error(f"下单失败: {symbol} {side} x{quantity} @ {price}")
        else:
            logger.info(f"已下单: {symbol} {side} x{quantity} "
                        f"@ {price}（订单号 {order_id}）")
        return int(order_id)

    def place_limit_order(self, symbol: str, side: str, quantity: int,
                          limit_price: float) -> int:
        """限价单（同 place_market_order，显式价格）。"""
        return self.place_market_order(symbol, side, quantity, limit_price)

    def cancel_order(self, order_id: int) -> bool:
        """撤单。"""
        return self.trader.cancel_order_stock(self.account, int(order_id)) == 0
