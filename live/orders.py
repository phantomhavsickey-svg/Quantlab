"""
指令构造 — 目标/分数 + 持仓 + 现金 + 参考价 → 买卖指令(dict)。

与具体券商无关:回测引擎和本模块共用 utils/sizing 的那份算式,
不会再出现"回测按目标建仓、实盘按现金摊派"的口径分叉。

两条路径共用同一段"Plan → 指令"的收尾逻辑:
    make_orders  等权 / 信号强度加权(直接给目标权重)
    plan_orders  分数带位策略(建仓线、补仓档、单票上限、减仓价)
策略模式下指令只负责下单,状态推进统一由
utils.position_policy.apply_fills 在成交回报之后做 —— 与回测引擎同一个提交点。
"""

from loguru import logger

from utils.position_policy import plan as policy_plan
from utils.position_policy import sync_intent_shares
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


def _position_maps(positions: dict, ref_price: dict):
    """(持仓字典, 股数, 有效参考价, 持仓市值),键统一成字符串。

    参考价缺失时用持仓对象自带的市价兜底估总资产 —— 那只股票本来也下不了单,
    但它的钱不能从预算里凭空消失。
    """
    positions = {str(s): p for s, p in (positions or {}).items()}
    held = {s: _shares_of(p) for s, p in positions.items()}
    prices = {str(s): float(p) for s, p in (ref_price or {}).items()
              if p is not None and float(p) > 0}
    mv = sum(q * _price_of(positions[s], s, ref_price)
             for s, q in held.items())
    return positions, held, prices, mv


def _to_requests(plan, positions, held, prices, cash, lot_size,
                 fee_rate_buy, intents=None):
    """Plan → 指令列表(先卖后买)。给了 intents 就把现金缩量回写进意图。"""
    orders: list[dict] = []

    # --- 先卖:清仓或减到目标(受 T+1 可卖数量约束;清仓允许零股) ---
    proceeds = 0.0
    for sym in sorted(plan.sells):
        pos = positions.get(sym)
        want = min(plan.sells[sym], held.get(sym, 0))
        qty = min(want, _available_of(pos, held.get(sym, 0)))
        if qty < held.get(sym, 0):        # 只减不清仓 → 按整手卖,零股留着
            qty = (qty // lot_size) * lot_size
        if qty <= 0:
            if plan.sells[sym] > 0:
                logger.debug(f"卖出 {sym}: 可卖数量不足 1 手,跳过")
            continue
        proceeds += qty * prices[sym]
        orders.append({"symbol": sym, "side": "sell", "quantity": int(qty),
                       "ref_price": prices[sym]})

    # --- 后买:建仓或加到目标(受可用资金约束) ---
    scale_buys_to_budget(plan, cash + proceeds, prices, lot_size=lot_size,
                         fee_rate_buy=fee_rate_buy)
    if intents:
        # 现金缩量是"这一档只能买这么多",按缩量后的股数推进状态。不回写的话,
        # 补仓每轮都被判成未足量成交、参考分数永不推进、仓位反复补到上限。
        sync_intent_shares(intents, plan)
    for sym in sorted(plan.buys):
        qty = plan.buys[sym]
        if qty <= 0:
            continue
        orders.append({"symbol": sym, "side": "buy", "quantity": int(qty),
                       "ref_price": prices[sym]})
    return orders


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

    positions, held, prices, mv = _position_maps(positions, ref_price)
    total_value = cash + mv
    plan = rebalance_plan(target, held, prices, total_value,
                          lot_size=lot_size, max_total_pct=max_total_pct)
    orders = _to_requests(plan, positions, held, prices, cash, lot_size,
                          fee_rate_buy)

    logger.info(f"指令构造完成: {sum(1 for o in orders if o['side'] == 'sell')} 卖 "
                f"+ {sum(1 for o in orders if o['side'] == 'buy')} 买 "
                f"(目标 {len(target)} 只, 总资产 {total_value:,.0f})")
    return orders


def plan_orders(scores, positions: dict, cash: float, ref_price: dict,
                states: dict, policy, *, lot_size: int = 100,
                asof=None) -> tuple[list[dict], object]:
    """分数带位策略版指令构造(与回测引擎走同一个 utils.position_policy.plan)。

    Args:
        scores: {symbol: 分数} 或 pd.Series,**全截面**(低于建仓线的候选也要在里面)
        positions: {symbol: 股数或持仓对象}
        cash: 当前可用资金
        ref_price: {symbol: 参考价}
        states: {symbol: NameState} 就地不变;成交回报后由 apply_fills 推进
        policy: utils.position_policy.PolicyConfig

    Returns:
        (orders, PolicyPlan) —— 撮合回报到手后调用
        apply_fills(states, pol.intents, before, after, fill_price, asof)
        再 save_states 落盘。
    """
    if hasattr(scores, "items"):
        scores = {str(s): float(v) for s, v in scores.items()
                  if v is not None and v == v}
    positions, held, prices, mv = _position_maps(positions, ref_price)
    total_value = cash + mv
    pol = policy_plan(scores or {}, held, prices, states, total_value,
                      policy, lot_size=lot_size, asof=asof)
    orders = _to_requests(pol.plan, positions, held, prices, cash, lot_size,
                          policy.fee_rate_buy, pol.intents)
    n_names = len([w for w in pol.weights.values() if w > 0])
    logger.info(
        f"策略指令: {sum(1 for o in orders if o['side'] == 'sell')} 卖 + "
        f"{sum(1 for o in orders if o['side'] == 'buy')} 买,动作 {pol.actions or '无'},"
        f" 目标 {n_names} 只 / 仓位 {pol.gross_weight:.1%},"
        f" 总资产 {total_value:,.0f}")
    return orders, pol
