"""执行层(选股→指令→成交→盯市)的口径测试。

与 quantlab2/tests/test_execution.py 是同一批断言:两边共用 utils/sizing 与
utils/market_rules,所以"回测怎么算量、实盘就怎么算量"要在两个仓库里都钉住。
每条测试只钉一处修复,数据全部合成且刻意把"停牌/缺行/涨跌停/低换手"摆在最
容易踩到的位置。不需要重训、不需要真实缓存。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.cost import TransactionCostModel
from backtest.engine import BacktestEngine
from live.orders import make_orders
from models.predictor import Predictor
from utils.market_rules import (at_limit_down, at_limit_up, can_fill,
                                get_limit_pct, build_tradable_mask)
from utils.position_policy import PolicyConfig
from utils.sizing import rebalance_plan, scale_buys_to_budget

CAP = 1_000_000
DATES = pd.DatetimeIndex(pd.bdate_range("2023-01-02", periods=110))
BUY_FEE = TransactionCostModel().effective_cost_rate("buy")


# ==================== 合成数据 ====================

def daily_frame(closes, vols=None, chgs=None, opens=None):
    n = len(closes)
    o = list(closes) if opens is None else list(opens)
    return pd.DataFrame({
        "日期": DATES[:n],
        "开盘": o,
        "收盘": list(closes),
        "最高": [x * 1.01 for x in o],
        "最低": [x * 0.99 for x in o],
        "成交量": [1e6] * n if vols is None else list(vols),
        "涨跌幅": [0.0] * n if chgs is None else list(chgs),
    })


def flat(level):
    return [float(level)] * len(DATES)


def idx_of(d):
    return list(DATES).index(pd.Timestamp(d))


def signals_from(weights_by_date):
    rows = []
    for d, ws in weights_by_date.items():
        for s, w in ws.items():
            rows.append({"date": d, "symbol": s, "weight": w, "score": w})
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    df.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(0)),
         df.index.get_level_values(1).astype(object)],
        names=["date", "symbol"])
    return df


def rebal_dates():
    """与引擎同口径:每月最后一个交易日。"""
    df = pd.DataFrame({"date": DATES})
    df["ym"] = df["date"].dt.strftime("%Y-%m")
    return df.groupby("ym")["date"].last().sort_values().tolist()


REBAL = rebal_dates()

# 每个调仓日的成交日 = 次一交易日;月末是数据末端时没有次日
EXEC_DAYS = [d + pd.offsets.BDay(1) for d in REBAL
             if (d + pd.offsets.BDay(1)) in set(DATES)]


def gap_on_exec_days(pct, level=10.0):
    """把每个成交日的**开盘价**打成一个跳空幅度(一字涨停/跌停天天如此)。

    成交判定看"开盘 vs 前收",所以标记必须打在开盘价上;早先版本改的是 `涨跌幅`
    列(收盘口径),那种数据在旧实现里能触发拦截、在新实现里什么也拦不住。
    """
    o = flat(level)
    for d in EXEC_DAYS:
        o[idx_of(d)] = level * (1 + pct / 100)
    return o


def run_bt(weights_by_date, data, freq="monthly", top_k=None, policy=None,
           cost_model=None):
    sig = signals_from(weights_by_date)
    if top_k is None:
        first = next(iter(weights_by_date.values()))
        top_k = len(first)
    eng = BacktestEngine(initial_capital=CAP, rebalance_frequency=freq,
                         max_positions=top_k,
                         cost_model=cost_model or TransactionCostModel(),
                         policy=policy)
    return eng.run(data, sig)


# ==================== sizing ====================

def test_rebalance_plan_targets_equal_weight():
    weights = {f"60000{i}": 0.1 for i in range(10)}
    prices = {f"60000{i}": 10.0 for i in range(10)}
    plan = rebalance_plan(weights, {}, prices, CAP)
    assert sum(plan.buys.values()) * 10.0 == pytest.approx(CAP, rel=2e-3)
    assert all(q % 100 == 0 for q in plan.buys.values())
    assert not plan.sells


def test_rebalance_plan_trims_overweight_and_tops_up_underweight():
    weights = {"600001": 0.5, "600002": 0.5}
    held = {"600001": 100_000, "600002": 25_000}    # 100万 vs 25万
    prices = {"600001": 10.0, "600002": 10.0}
    plan = rebalance_plan(weights, held, prices, total_value=1_250_000)
    assert plan.target_shares == {"600001": 62_500, "600002": 62_500}
    assert plan.sells == {"600001": 37_500}
    assert plan.buys == {"600002": 37_500}


def test_rebalance_plan_leaves_unpriced_names_alone():
    plan = rebalance_plan({"600001": 0.5, "600002": 0.5},
                          {"600009": 1000}, {"600001": 10.0, "600002": 10.0},
                          total_value=100_000)
    assert "600009" not in plan.sells and "600009" not in plan.target_shares


def test_scale_buys_to_budget_is_order_independent():
    """钱不够时等比缩量,结果与字典顺序无关(旧实现按行序买到没钱)。"""
    prices = {f"60000{i}": 10.0 for i in range(10)}
    weights = {s: 0.1 for s in prices}
    a = rebalance_plan(weights, {}, prices, CAP)
    b = rebalance_plan(dict(reversed(list(weights.items()))), {},
                       dict(reversed(list(prices.items()))), CAP)
    scale_buys_to_budget(a, 300_000.0, prices, fee_rate_buy=BUY_FEE)
    scale_buys_to_budget(b, 300_000.0, prices, fee_rate_buy=BUY_FEE)
    assert sorted(a.buys.items()) == sorted(b.buys.items())
    assert sum(q * 10.0 for q in a.buys.values()) * (1 + BUY_FEE) <= 300_000.0


# ==================== 交易成本 ====================

STAMP_SCHED = {"1990-01-01": 0.001, "2023-08-28": 0.0005}


def test_stamp_tax_rate_follows_effective_date():
    """回测区间跨 2023-08-28：卖出按成交日取档，早于首档取最早那档，买入恒 0。"""
    m = TransactionCostModel(stamp_tax_schedule=STAMP_SCHED)
    assert m.stamp_rate_on("2023-08-27") == 0.001
    assert m.stamp_rate_on("2023-08-28") == 0.0005
    assert m.stamp_rate_on("1995-01-01") == 0.001       # 早于全部生效日 → 最早那档
    assert m.stamp_rate_on(None) == m.stamp_tax_rate    # 不传日期 → 现行档
    assert m.stamp_tax(100_000, "buy", "2023-06-01") == 0.0
    assert TransactionCostModel().stamp_rate_on("2023-06-01") == 0.0005  # 不分档


def test_total_cost_accounts_the_stamp_tax_cut():
    """同一笔卖出金额，跨档前后差一整档印花税（不分档就是 2023 上半年少收一半）。"""
    m = TransactionCostModel(stamp_tax_schedule=STAMP_SCHED)
    before = m.total_cost(1_000_000, "sell", pd.Timestamp("2023-08-25"))
    after = m.total_cost(1_000_000, "sell", pd.Timestamp("2023-09-01"))
    assert round(before - after, 6) == 500.0            # 100 万 × (万十 − 万五)
    assert m.total_cost(1_000_000, "sell") == after     # 不给日期按现行档


def test_scale_buys_to_budget_reserves_the_commission_floor():
    """佣金 5 元下限：只按费率比例预留会少留钱，逐笔 cost_fn 才不会放行买不下的单。"""
    prices = {"600001": 10.0, "600002": 10.0}
    model = TransactionCostModel()                      # 万三/5 元下限 + 万十滑点
    cap, w = 10_000.0, {"600001": 0.5, "600002": 0.5}  # 权重按归一后的相对目标算

    def real_cost(p):
        return sum(q * 10.0 + model.total_cost(q * 10.0, "buy")
                   for q in p.buys.values())

    base = rebalance_plan(w, {}, prices, cap)
    assert base.buys == {"600001": 500, "600002": 500}
    assert real_cost(base) == 10_020.0                  # 5,000×2 + (5 元佣金+5 元滑点)×2
    # 比例估算只有 10,013 元 —— 佣金下限那 5 元/笔没算进去
    with_fn = rebalance_plan(w, {}, prices, cap)
    scale_buys_to_budget(with_fn, 10_016.0, prices, fee_rate_buy=BUY_FEE,
                         cost_fn=lambda a: model.total_cost(a, "buy"))
    assert real_cost(with_fn) <= 10_016.0
    without = rebalance_plan(w, {}, prices, cap)
    scale_buys_to_budget(without, 10_016.0, prices, fee_rate_buy=BUY_FEE)
    assert real_cost(without) > 10_016.0                # 以为放得下，实际超 4 元


def test_paper_orders_reserve_the_same_fee_as_the_backtest():
    """模拟盘的费率与预留必须和回测同源 —— broker 不再自己写一遍 max()/0.0005。"""
    from live.orders import make_orders
    from paper_trade.broker import SimulatedBroker

    br = SimulatedBroker(initial_cash=10_016.0)
    model = TransactionCostModel(br.commission_rate, br.min_commission,
                                 br.stamp_tax_rate, br.slippage_rate)
    for amt in (10_000.0, 5_000.0, 1_000.0):
        for side in ("buy", "sell"):
            assert br.total_cost(amt, side) == model.total_cost(amt, side)

    prices = {"600011": 10.0, "600012": 10.0}
    w = {"600011": 0.5, "600012": 0.5}

    def need(orders):
        return sum(o["quantity"] * o["ref_price"]
                   + br.total_cost(o["quantity"] * o["ref_price"], "buy")
                   for o in orders)

    plain = make_orders(w, {}, br.cash, prices, fee_rate_buy=BUY_FEE)
    guard = make_orders(w, {}, br.cash, prices, fee_rate_buy=BUY_FEE,
                        cost_fn=br.total_cost)
    assert need(plain) > br.cash                     # 只按费率估 → 放行买不下的单
    assert need(guard) <= br.cash                    # 逐笔预留 → 缩到放得下


def test_daily_signal_path_reserves_the_brokers_fee():
    """每日指令这条路径(实盘与纸交易共用)留的钱要和 broker 收的钱一致。

    费率算式一旦在 live/daily_signal 里退回"佣金+滑点"比例,实盘就会下发回测里
    买不起的单子。这里不构造 DailySignalGenerator(__init__ 要加载模型),只把
    make_orders 挂到裸实例上跑一条真实 config。
    """
    import yaml

    from live.daily_signal import DailySignalGenerator, buy_cost_fn
    from paper_trade.broker import SimulatedBroker

    market = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config.yaml")
        .read_text(encoding="utf-8"))["market"]
    br = SimulatedBroker(initial_cash=10_016.0,
                         commission_rate=market["commission_rate"],
                         min_commission=market["min_commission"],
                         stamp_tax_rate=market["stamp_tax_rate"],
                         slippage_rate=market["slippage_rate"])
    f = buy_cost_fn(market)
    for amt in (10_000.0, 5_000.0, 1_000.0, 100_000.0):
        assert f(amt) == br.total_cost(amt, "buy")   # 含 5 元下限那一档

    gen = object.__new__(DailySignalGenerator)
    gen.config = {"market": market}
    orders = gen.make_orders(pd.Series({"600011": 0.5, "600012": 0.5}), {},
                             br.cash, {"600011": 10.0, "600012": 10.0})
    bought = sum(o["quantity"] * o["ref_price"]
                 + br.total_cost(o["quantity"] * o["ref_price"], "buy")
                 for o in orders if o["side"] == "buy")
    assert bought <= br.cash


def test_engine_charges_stamp_tax_of_the_exec_date():
    """成交日必须传到成本模型:合成数据全在 2023-08-28 之前,卖出该按万十计。

    不比对两档的总成本之差 —— 印花税少收的现金会回灌买入约束,成交量本身就会
    差一笔。改成逐笔反解税率: cost/amount - 佣金 - 滑点 剩下的就是印花税率,
    卖出必须正好落在生效档上,买入必须是 0。最低佣金置 0 才反解得干净。
    """
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)), b: daily_frame(flat(10.0))}
    sig = {d: ({a: 1.0} if i % 2 == 0 else {b: 1.0}) for i, d in enumerate(REBAL)}
    flat_res = run_bt(sig, data, cost_model=TransactionCostModel(min_commission=0.0))
    sched_res = run_bt(sig, data, cost_model=TransactionCostModel(
        min_commission=0.0, stamp_tax_schedule=STAMP_SCHED))

    def implied_stamp_rate(tr):
        # 佣金万三 + 滑点千一,双边都收;剩下的差额只可能是印花税
        return (tr["cost"] / tr["amount"]
                - 0.0003 - 0.001)

    for res, rate in ((flat_res, 0.0005), (sched_res, 0.001)):
        tr = res["trades"]
        sold = tr[tr["side"] == "sell"]
        assert len(sold) >= 2, "这份流水得有卖出才构成对照"
        assert sold["date"].max() < pd.Timestamp("2023-08-28")
        assert np.allclose(implied_stamp_rate(sold), rate, atol=1e-9)
        assert np.allclose(implied_stamp_rate(tr[tr["side"] == "buy"]),
                           0.0, atol=1e-9)


# ==================== market_rules ====================

def test_limit_pct_by_board():
    """创业板/科创板 20%,主板 10% —— 替掉实盘里写死的 9.5。"""
    assert get_limit_pct("600001") == 0.10
    assert get_limit_pct("000001") == 0.10
    assert get_limit_pct("300750") == 0.20
    assert get_limit_pct("688981") == 0.20
    assert at_limit_up("300750", 9.9) is False       # 主板的涨幅在创业板不是涨停
    assert at_limit_up("600001", 9.9) is True
    assert at_limit_down("688981", -12.0) is False
    assert at_limit_down("688981", -19.9) is True


def test_can_fill_blocks_no_bar_zero_volume_and_limit():
    assert can_fill(None, "600001", "buy", 10.0)[0] is False
    assert can_fill(pd.Series({"收盘": 10.0, "开盘": 10.0, "成交量": 0.0}),
                    "600001", "buy", 10.0)[0] is False
    ok, why = can_fill(pd.Series({"收盘": 10.0, "开盘": 11.0, "成交量": 1e6}),
                       "600001", "sell", 10.0)
    assert ok and why == "ok"
    ok, why = can_fill(pd.Series({"收盘": 10.0, "开盘": 11.0, "成交量": 1e6}),
                       "600001", "buy", 10.0)
    assert not ok and "开盘涨停" in why


def test_can_fill_judges_the_open_not_the_close():
    """撮合价是开盘价,涨跌停就必须按开盘判 —— 收盘涨跌幅会放行成交不了的单子。"""
    row = pd.Series({"收盘": 9.70, "开盘": 9.00, "成交量": 1e6, "涨跌幅": -3.0})
    assert can_fill(row, "600001", "sell", 10.0)[0] is False    # 开盘 -10%:跌停卖不掉
    assert can_fill(row, "600001", "buy", 10.0)[0] is True      # 买方向不受影响
    # 20% 板按 10% 跌幅放行,10% 板按 19.9% 跌幅拦下:幅度取代码前缀
    assert can_fill(row, "300750", "sell", 10.0)[0] is True
    row20 = pd.Series({"收盘": 8.0, "开盘": 8.02, "成交量": 1e6, "涨跌幅": -19.8})
    assert can_fill(row20, "688981", "sell", 10.0)[0] is False
    # 给不出前收(该股在这一天第一次有 bar)就不猜方向,按不可成交处理
    assert can_fill(row, "600001", "sell", None)[1] == "无前收盘价"


# ==================== tradable 接线 ====================

def test_tradable_mask_drops_no_volume_days():
    df = daily_frame(flat(10))
    df.loc[3, "成交量"] = 0.0
    mask = build_tradable_mask({"600001": df}, DATES[:5])
    seen = set(mask.index.get_level_values("date"))
    assert DATES[3] not in seen
    assert seen == {DATES[0], DATES[1], DATES[2], DATES[4]}


def test_suspended_names_do_not_consume_top_k_slots():
    d0, d1 = DATES[0], REBAL[0]
    preds = pd.Series(
        [5.0, 4.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [(d0, "600001"), (d0, "600002"), (d0, "600003")],
            names=["date", "symbol"]))
    tradable = pd.Series(
        True,
        index=pd.MultiIndex.from_tuples(
            [(d0, "600001"), (d0, "600003")], names=["date", "symbol"]))
    held = Predictor(trainer=None, top_k=2).generate_signals_from_series(
        preds, tradable=tradable)
    held = held[held["weight"] > 0]
    assert set(held.index.get_level_values("symbol")) == {"600001", "600003"}


# ==================== 回测引擎 ====================

def test_first_rebalance_is_equal_weight_and_fully_invested():
    names = [f"60000{i}" for i in range(10)]
    data = {s: daily_frame(flat(10.0 + i)) for i, s in enumerate(names)}
    res = run_bt({d: {s: 0.1 for s in names} for d in REBAL}, data)
    tr = res["trades"]
    buys = tr[tr["date"] == tr["date"].min()]
    buys = buys[buys["side"] == "buy"]
    assert len(buys) == 10
    amounts = buys["amount"].to_numpy()
    assert amounts.max() / amounts.min() < 1.35
    assert res["execution"]["n_underinvested_rebalances"] == 0


def test_low_turnover_rebalance_still_targets_equal_weight():
    """钉住 F0:换 1 只时新入选的要拿到约 1/N 仓位。

    旧实现 buy_capital = cash/len(target):低换手月份只回笼 1/N 的钱,
    于是新名字被建成 (1/N)² 的仓位 —— 这个塌陷在这里被直接测出来。
    """
    names = [f"60000{i}" for i in range(10)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    data["301010"] = daily_frame(flat(10.0))
    w1 = {s: 0.1 for s in names}
    w2 = {s: 0.1 for s in names[:-1]}
    w2["301010"] = 0.1
    res = run_bt({REBAL[0]: w1, REBAL[1]: w2, REBAL[2]: w2}, data)
    tr = res["trades"]
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    new_buy = tr[(tr["date"] == exec2) & (tr["symbol"] == "301010")]
    assert len(new_buy) == 1, "新入选股票没被建仓"
    equity = float(res["equity_curve"].iloc[0]) * CAP
    amt = float(new_buy["amount"].iloc[0])
    assert 0.07 * equity < amt < 0.13 * equity, (amt, equity)


def test_limit_up_blocks_the_buy_and_leaves_cash():
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame(flat(10.0), opens=gap_on_exec_days(10.0))}
    res = run_bt({d: {a: 0.5, b: 0.5} for d in REBAL}, data)
    assert (res["trades"]["symbol"] == b).sum() == 0, "涨停股不该成交"
    assert res["execution"]["blocked_buy"].get("开盘涨停 +10.00%", 0) >= 1
    tail = res["marks"]
    assert float(tail["cash"].iloc[-1]) > 0.4 * CAP, "买不进的钱要留在现金里"
    assert res["execution"]["n_underinvested_rebalances"] >= 1


def test_limit_down_keeps_the_position_valued():
    a, b = "600001", "600002"
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    exec3 = REBAL[2] + pd.offsets.BDay(1)
    opens = flat(10.0)
    opens[idx_of(exec2)] = 9.0
    data = {a: daily_frame(flat(10.0), opens=opens), b: daily_frame(flat(10.0))}
    keep, drop = {a: 0.5, b: 0.5}, {b: 1.0}
    res = run_bt({REBAL[0]: keep, REBAL[1]: drop, REBAL[2]: drop}, data)
    tr = res["trades"]
    sold_a = tr[(tr["symbol"] == a) & (tr["side"] == "sell")]
    assert exec2 not in set(sold_a["date"]), "跌停日卖不出去"
    assert set(sold_a["date"]) == {exec3}, "解除跌停后的下一次调仓该卖掉"
    assert res["execution"]["blocked_sell"].get("开盘跌停 -10.00%") == 1
    pos = res["positions"]
    assert (pos["symbol"] == a).sum() > 0, "卖不掉的持仓要继续计在净值里"


def test_missing_bars_do_not_zero_out_market_value():
    """钉住 F1:已建仓的 b 断档 6 个交易日,市值沿用最近有效收盘而不是归零。"""
    a, b = "600001", "600002"
    gap = daily_frame(flat(20.0))
    cut = idx_of(REBAL[0]) + 2          # 成交日(REBAL[0]+1)之后的 6 天没有日线
    gap = gap.drop(index=range(cut, cut + 6)).reset_index(drop=True)
    data = {a: daily_frame(flat(10.0)), b: gap}
    res = run_bt({d: {a: 0.5, b: 0.5} for d in REBAL}, data)
    window = res["marks"].loc[DATES[cut]: DATES[cut + 5]]
    assert len(window) == 6
    mv = window["market_value"]
    assert mv.min() > 0.99 * mv.max(), mv.to_dict()
    assert (window["n_positions"] == 2).all()
    assert mv.iloc[0] > 0.97 * CAP


def test_daily_marks_use_the_portfolio_actually_held_that_month():
    """钉住前视修复:下一个月末才决定的组合不能给这一个月估值。

    A 全期不动,B 每天 +1%。第一个月只持 A,第二个月末换成 B。
    旧实现先更新持仓再回补上一区间 → 第一个月的净值里就冒出 B 的涨幅。
    """
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame([10.0 * 1.01 ** i for i in range(len(DATES))])}
    res = run_bt({REBAL[0]: {a: 1.0}, REBAL[1]: {b: 1.0},
                  REBAL[2]: {b: 1.0}}, data)
    eq = res["marks"]
    exec1 = REBAL[0] + pd.offsets.BDay(1)
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    month1 = eq.loc[exec1: exec2 - pd.offsets.BDay(1)]
    assert len(month1) > 5
    tv = month1["total_value"].to_numpy()
    assert np.allclose(tv, tv[0]), "第一个月净值里混进了第二个月的持仓变动"
    after = eq.loc[exec2:]
    assert after["total_value"].iloc[-1] > tv[0], "换到 B 之后才该吃到涨幅"


def test_no_same_day_round_trip_under_daily_rebalancing():
    """daily 调仓 + 目标每天换:任何一只股票都不能同一天既买又卖。"""
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)), b: daily_frame(flat(10.0))}
    weights = {d: ({a: 1.0} if i % 2 == 0 else {b: 1.0})
               for i, d in enumerate(DATES[:-1])}
    res = run_bt(weights, data, freq="daily")
    tr = res["trades"]
    per_day = tr.groupby(["symbol", "date"])["side"].apply(set)
    assert not any("buy" in s and "sell" in s for s in per_day), per_day
    assert len(tr) > 0


def test_execution_diagnostics_report_holding_period():
    names = [f"60000{i}" for i in range(5)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    first = {s: 0.2 for s in names}
    later = {s: 0.25 for s in names[:-1]}      # 第二次换仓:踢掉最后一只
    res = run_bt({d: (first if d == REBAL[0] else later) for d in REBAL}, data)
    ex = res["execution"]
    exec2 = REBAL[1] + pd.offsets.BDay(1)
    tr = res["trades"]
    assert ex["n_exit_trades"] == 1
    assert 15 <= ex["median_hold_days"] <= 30, ex
    assert ex["n_blocked_buy"] == 0 and ex["n_blocked_sell"] == 0
    # 目标稳定下来之后不该再有任何成交(首轮建仓把 mean_turnover 摊薄,不能直接看它)
    assert (tr["date"] > exec2).sum() == 0, tr[tr["date"] > exec2]
    assert ex["n_underinvested_rebalances"] == 0


def test_backtest_and_live_size_the_same_plan():
    """钉住两端一致:同一份权重/价格/资产,回测与实盘给出同样的股数。"""
    names = [f"60000{i}" for i in range(5)]
    prices = {s: 10.0 for s in names}
    weights = {s: 0.2 for s in names}
    data = {s: daily_frame(flat(10.0)) for s in names}
    res = run_bt({REBAL[0]: weights}, data)
    exec1 = REBAL[0] + pd.offsets.BDay(1)
    bt = res["trades"]
    bt = bt[bt["date"] == exec1].set_index("symbol")["shares"].to_dict()
    assert len(bt) == 5

    lv = {o["symbol"]: o["quantity"] for o in
          make_orders(weights, {}, float(CAP), prices, fee_rate_buy=BUY_FEE)}
    assert bt == lv, (bt, lv)
    assert sum(q * 10.0 for q in lv.values()) > 0.97 * CAP


# ==================== 分数带位策略 × 回测引擎 ====================

def cum_shares(trades, sym):
    """{成交日: 该票当日收盘时的累计股数}。"""
    out, t = {}, 0
    rows = trades[trades["symbol"] == sym].sort_values("date")
    for _, r in rows.iterrows():
        t += r["shares"] if r["side"] == "buy" else -r["shares"]
        out[r["date"]] = t
    return out


def test_policy_ramps_to_the_cap_then_stops_trading():
    """建仓 5% → 补两档到 15% → 顶到 16% 上限 → 分数不变就彻底不动。

    这里钉的是"同一档只触发一次":如果缩量/挡单被判成已成交,参考分数会假
    推进、仓位停在半档;如果未成交不推进状态,第 4 轮还会再买一次。
    """
    a = "600001"
    data = {a: daily_frame(flat(10.0))}
    scores = {d: {a: 0.02 if i == 0 else 0.05} for i, d in enumerate(REBAL)}
    res = run_bt(scores, data, policy=PolicyConfig(min_trade_value=5000))
    cum = cum_shares(res["trades"], a)
    assert list(cum) == EXEC_DAYS[:3], cum            # 最后一次评估没有成交
    assert cum[EXEC_DAYS[0]] == 5000                  # 建仓线 → 5%
    assert 14900 <= cum[EXEC_DAYS[1]] <= 15000        # 两档补仓 → 15%
    assert 15900 <= cum[EXEC_DAYS[2]] <= 16000        # 压到单票上限 16%
    st = res["policy_states"][a]
    assert (st.entry_score, st.adds) == (0.02, 2)
    # 第 3 轮只走掉一档的量(15%→16% 被单票上限截断),但档位按请求的 2 档一起
    # 结清:仓位已经贴顶,把没吃到上限的那一档留着记账只会让它每轮重复挂单。
    assert st.ref_score == pytest.approx(0.04)
    assert res["execution"]["policy_actions"] == {"entry": 1, "add": 2}


def test_policy_blocked_entry_retries_every_round():
    """涨停买不进 → 状态不落地、不留"已建仓"的假记录,下一轮继续重试。"""
    a, b = "600001", "600002"
    data = {a: daily_frame(flat(10.0)),
            b: daily_frame(flat(10.0), opens=gap_on_exec_days(10.0))}
    scores = {d: {a: 0.02, b: 0.02} for d in REBAL}
    res = run_bt(scores, data, policy=PolicyConfig())
    tr = res["trades"]
    assert (tr["symbol"] == b).sum() == 0
    assert b not in res["policy_states"] and a in res["policy_states"]
    assert res["execution"]["blocked_buy"]["开盘涨停 +10.00%"] == len(EXEC_DAYS)


def test_policy_entry_budget_is_fee_aware_and_capped():
    """20 个高分候选:含费预算只放得下 11 只,总仓位不越 95%,之后不再换手。"""
    names = [f"6000{i:02d}" for i in range(20)]
    data = {s: daily_frame(flat(10.0)) for s in names}
    scores = {d: {s: 0.05 for s in names} for d in REBAL}
    res = run_bt(scores, data, policy=PolicyConfig())
    ex = res["execution"]
    assert ex["policy_actions"] == {"entry": 11}       # 0.88 含费,第 12 挤不进
    assert ex["policy_final_names"] == 11
    m = res["marks"]
    assert (m["market_value"] / m["total_value"]).max() < 0.96
    assert 0.85 < ex["policy_mean_gross_weight"] < 0.90
    # 第 2 轮起分数没变 → 一分钱都不该再动(补仓预留额度没被吃掉也不会乱补)
    assert set(res["trades"]["date"]) == {EXEC_DAYS[0]}


def test_policy_exit_below_the_sell_line_clears_the_state():
    a = "600001"
    data = {a: daily_frame(flat(10.0))}
    scores = {REBAL[0]: {a: 0.03}, REBAL[1]: {a: -0.01},
              REBAL[2]: {a: -0.01}}
    res = run_bt(scores, data, policy=PolicyConfig())
    tr = res["trades"]
    assert cum_shares(tr, a)[EXEC_DAYS[1]] == 0        # 跌破清仓线 → 全清
    assert res["policy_states"] == {}                  # 真止损,不锁价格
    assert res["execution"]["policy_actions"] == {"entry": 1, "exit": 1}
    assert float(res["marks"]["market_value"].iloc[-1]) == pytest.approx(0.0)


def test_live_and_backtest_issue_the_same_policy_orders():
    """两端一致(策略版):同一份分数/持仓/现金/价格 → 同一张指令单。

    回测引擎消费 position_policy.plan 的 Plan,实盘消费 plan_orders 包住的同一个
    Plan。这条测试钉的就是这个接缝不会分叉。
    """
    from live.orders import plan_orders
    from utils.position_policy import NameState
    from utils.position_policy import plan as policy_plan

    pc = PolicyConfig()
    scores = {"600001": 0.05, "600002": 0.02}     # 补两档 / 刚过建仓线
    prices = {"600001": 10.0, "600002": 10.0}
    states = {"600001": NameState(entry_score=0.03, ref_score=0.03, step=0.01)}
    held = {"600001": 5000}

    class P:
        def __init__(self, shares):
            self.shares, self.available_shares = shares, shares
            self.market_price = 10.0

    live_orders, pol = plan_orders(scores, {"600001": P(5000)}, 950_000.0,
                                   prices, dict(states), pc, asof=DATES[0])
    bt = policy_plan(scores, held, prices, dict(states), 1_000_000.0, pc,
                     asof=DATES[0])
    assert {o["symbol"]: o["quantity"] for o in live_orders
            if o["side"] == "buy"} == bt.plan.buys == {"600001": 10000,
                                                       "600002": 5000}
    assert not [o for o in live_orders if o["side"] == "sell"] and not bt.plan.sells
    assert set(pol.intents) == set(bt.intents) == {"600001", "600002"}


def test_policy_state_commits_only_after_a_real_fill():
    """跌停卖不掉 → 状态不推进;次日成交后按**实际成交价**记减仓价。"""
    from live.orders import plan_orders
    from paper_trade.broker import Position, SimulatedBroker
    from utils.position_policy import NameState, apply_fills

    pc = PolicyConfig()
    states = {"600001": NameState(entry_score=0.03, ref_score=0.03, step=0.01)}
    br = SimulatedBroker(initial_cash=850_000.0, lot_size=100)
    br.positions["600001"] = Position(symbol="600001", shares=15000,
                                      available_shares=15000, avg_cost=10.0,
                                      market_price=10.0)
    d1, d2 = EXEC_DAYS[0], EXEC_DAYS[1]
    orders, pol = plan_orders({"600001": 0.01}, br.positions, br.cash,
                              {"600001": 10.0}, states, pc, asof=d1)
    assert [(o["side"], o["quantity"]) for o in orders] == [("sell", 10000)]
    before = {"600001": 15000}
    for o in orders:
        br.place_market_order(o["symbol"], o["side"], o["quantity"])

    def bar(o, down):
        return {"600001": {"open": o, "high": o + 0.2, "low": o - 0.2,
                           "close": o, "volume": 1e6, "at_limit_up": False,
                           "at_limit_down": down}}

    assert br.process_daily(d1, bar(9.0, True)) == []       # 跌停卖不掉
    rep = apply_fills(states, pol.intents, before,
                      {s: p.shares for s, p in br.positions.items()}, {},
                      asof=d1)
    assert "未足量成交" in rep["600001"]
    assert (states["600001"].ref_score, states["600001"].trim_price) == (0.03, None)

    filled = br.process_daily(d2, bar(9.6, False))
    assert [(f.filled_quantity, f.filled_price) for f in filled] == [(10000, 9.6)]
    apply_fills(states, pol.intents, before,
                {s: p.shares for s, p in br.positions.items()},
                {"600001": 9.6}, asof=d2)
    st = states["600001"]
    assert st.ref_score == pytest.approx(0.01)               # 两档一起结清
    assert (st.trim_price, st.trim_date) == (9.6, str(d2.date()))


# ==================== 按笔盈亏口径 ====================

def test_trade_win_rate_is_measured_against_cost_not_proceeds():
    """卖出的 net_proceeds 是现金流(卖额 − 卖出费),不含买入成本。

    A 涨 B 跌、C/D 平价接棒 → 两笔卖出必然一盈一亏。旧实现拿净现金流当盈亏,
    任何一笔卖出都为正 → 胜率恒 100%、利润因子 inf。
    """
    a, b, c, d = "600001", "600002", "600003", "600004"
    n1 = idx_of(REBAL[1]) + 1                     # 换仓日之前一律 10 元
    rest = len(DATES) - n1
    data = {a: daily_frame([10.0] * n1 + [10.0 * 1.02 ** i for i in range(1, rest + 1)]),
            b: daily_frame([10.0] * n1 + [10.0 * 0.98 ** i for i in range(1, rest + 1)]),
            c: daily_frame(flat(10.0)), d: daily_frame(flat(10.0))}
    res = run_bt({REBAL[0]: {a: 0.5, b: 0.5}, REBAL[1]: {c: 0.5, d: 0.5}}, data)

    sells = res["trades"].query("side == 'sell'").set_index("symbol")["realized_pnl"]
    assert set(sells.index) == {a, b}, sells
    assert sells[a] > 0 > sells[b], sells
    m = res["metrics"]
    assert m["win_rate_by_trade"] == pytest.approx(0.5)
    assert m["profit_factor"] != float("inf")


def test_journal_win_rate_subtracts_the_purchase_cost(tmp_path):
    from paper_trade.journal import TradeJournal

    def trade(symbol, side, qty, amount, cost):
        return {"symbol": symbol, "side": side, "quantity": qty, "amount": amount,
                "commission": cost, "stamp_tax": 0.0, "slippage": 0.0,
                "total_cost": cost}

    j = TradeJournal(output_dir=str(tmp_path))
    j.trades_log = [
        trade("600001", "buy", 100, 1000.0, 5.0),
        trade("600001", "sell", 100, 900.0, 5.0),        # 亏 110(买入还垫了 5 元费)
        trade("600002", "buy", 200, 2000.0, 10.0),       # 均价 10.05/股
        trade("600002", "sell", 100, 1100.0, 10.0),      # 赚 85,余下成本要跟着减
        trade("600002", "sell", 100, 900.0, 10.0),       # 亏 115
    ]
    assert j.realized_pnl() == pytest.approx([-110.0, 85.0, -115.0])
    st = j.compute_statistics()
    assert st["win_rate"] == pytest.approx(1 / 3)
    assert st["total_pnl"] == pytest.approx(-140.0)
    assert st["profit_factor"] == pytest.approx(85.0 / 225.0)


