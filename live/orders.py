"""
指令构造 — 目标权重 + 持仓 + 现金 + 参考价 → 买卖指令(dict)。

与具体券商无关:回测引擎和本模块共用 utils/sizing 的那份算式,
不会再出现"回测按目标市值建仓、实盘按现金摊派"的口径分叉。
"""

from loguru import logger

from utils.sizing import rebalance_plan, scale_buys_to_budget


def _shares_of(pos) -> int:
    """持仓既可以是股数,也可以是带 shares/available_shares 的持仓对象。"""
    if isinstance(pos, (int, float)):
        return int(pos)
    return int(getattr(pos, "shares", 0) or 0)


def _available_of(pos, shares: int) -> int:
    avail = getattr(pos, "available_shares", None)
    return shares if avail is None else int(avail or 0)


def _price_of(pos, sym, ref_price: dict) -> float:
    if sym in ref_price and ref_price[sym]:
        return float(ref_price[sym])
    return float(getattr(pos, "market_price", 0.0) or 0.0)


def make_orders(target_weights, positions: dict, cash: float,
                ref_price: dict, lot_size: int = 100,
                max_total_pct: float = 1.0,
                fee_rate_buy: float = 0.0) -> list[dict]:
    """从目标权重生成买卖指令(先卖后买,下目标市值与现有市值的差额)。

    Args:
        target_weights: {symbol: 权重} 或带 index 的 Series;权重 > 0 即应持有,
                        缺席即应清仓
        positions: {symbol: 股数或持仓对象}
        cash: 当前可用资金
        ref_price: {symbol: 参考价}(缺价的股票保留原状,不下单)
        lot_size: 一手股数
        max_total_pct: 投资总额上限
        fee_rate_buy: 买入单边费率(佣金+滑点);资金不足时按"含费"缩量,
                      与回测引擎同一条算式

    Returns:
        list[dict]: {"symbol","side","quantity","ref_price"},卖出在前、买入在后
    """
    if hasattr(target_weights, "index"):          # pd.Series {symbol: weight}
        target_weights = dict(target_weights.items())
    target = {s: float(w) for s, w in (target_weights or {}).items()
              if w is not None and float(w) > 0}
    if not target and not positions:
        return []

    held = {s: _shares_of(p) for s, p in positions.items()}
    prices = {s: float(p) for s, p in (ref_price or {}).items()
              if p is not None and float(p) > 0}
    mv = sum(q * _price_of(positions[s], s, ref_price)
             for s, q in held.items())
    total_value = cash + mv

    plan = rebalance_plan(target, held, prices, total_value,
                          lot_size=lot_size, max_total_pct=max_total_pct)

    orders: list[dict] = []
    proceeds = 0.0
    for sym in sorted(plan.sells):
        want = min(plan.sells[sym], held.get(sym, 0))
        qty = min(want, _available_of(positions[sym], held.get(sym, 0)))
        if qty < held.get(sym, 0):        # 只减不清仓 → 按整手卖,零股留着
            qty = (qty // lot_size) * lot_size
        if qty <= 0:
            if plan.sells[sym] > 0:
                logger.debug(f"卖出 {sym}: 可卖不足 1 手,跳过")
            continue
        proceeds += qty * prices[sym]
        orders.append({"symbol": sym, "side": "sell", "quantity": int(qty),
                       "ref_price": prices[sym]})

    scale_buys_to_budget(plan, cash + proceeds, prices, lot_size=lot_size,
                         fee_rate_buy=fee_rate_buy)
    for sym in sorted(plan.buys):
        qty = plan.buys[sym]
        if qty <= 0:
            continue
        orders.append({"symbol": sym, "side": "buy", "quantity": int(qty),
                       "ref_price": prices[sym]})

    logger.info(f"指令构造完成: {sum(1 for o in orders if o['side']=='sell')} 卖 "
                f"+ {sum(1 for o in orders if o['side']=='buy')} 买 "
                f"(目标 {len(target)} 只, 总资产 {total_value:,.0f})")
    return orders
