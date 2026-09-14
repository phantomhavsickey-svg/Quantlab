"""加载上次持仓 + 今日信号 → 打印操作清单"""
import os, sys, pandas as pd
sys.path.insert(0, ".")

from data.cache import CacheManager
from models.trainer import LightGBMTrainer
from models.predictor import Predictor
from live import SinaQuoteFeed
import yaml

# ==== 1. 加载上次模拟盘持仓 ====
trades = pd.read_csv("logs/paper_trade_result_trades.csv")
positions = {}
for _, t in trades.iterrows():
    sym = str(int(t["symbol"])).zfill(6)
    qty = int(t["quantity"])
    if t["side"] == "buy":
        positions[sym] = positions.get(sym, 0) + qty
    else:
        positions[sym] = positions.get(sym, 0) - qty

current = {k: v for k, v in positions.items() if v > 0}
print(f"=== 上次模拟盘最终持仓: {len(current)} 只 ===\n")

# ==== 2. 加载今日信号 ====
config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
trainer = LightGBMTrainer(model_type=config["model"]["type"])
models = sorted([f for f in os.listdir("models/saved") if f.endswith(".txt")])
trainer.load(os.path.join("models/saved", models[-1]))

# 加载因子面板
factor_panel = pd.read_parquet("data/cache/factor_panel.parquet")
factor_panel["date"] = pd.to_datetime(factor_panel["date"])
factor_cols = [c for c in factor_panel.columns if c not in ["date", "symbol"]]
X = factor_panel.set_index(["date", "symbol"])[factor_cols]

predictor = Predictor(trainer, top_k=30, position_sizing="equal_weight")
signals = predictor.generate_signals(X)

latest_date = sorted(signals.index.get_level_values("date").unique())[-1]
latest = signals.xs(latest_date, level="date")
target = set(latest[latest["weight"] > 0].index)

# ==== 3. 实时价格 ====
feed = SinaQuoteFeed()
quotes = feed.fetch(list(target | set(current.keys())))

# ==== 4. 对比持仓 ====
to_sell = set(current.keys()) - target
to_buy = target - set(current.keys())
to_hold = set(current.keys()) & target

print(f"日期: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")
print(f"上次持仓: {len(current)} 只 | 今日目标: {len(target)} 只")
print(f"卖出: {len(to_sell)} 只 | 买入: {len(to_buy)} 只 | 持有: {len(to_hold)} 只")
print()

# 计算持仓估值
total_value = 0
total_cost = 0
if to_sell:
    print(f"[卖出] ({len(to_sell)} 只)")
    for sym in sorted(to_sell):
        shares = current[sym]
        q = quotes.get(sym)
        price = q.price if q else 0
        print(f"  {sym}: {shares}股 @ {price:.2f} = {shares*price:,.0f}")
    print()

if to_buy:
    print(f"[买入] ({len(to_buy)} 只)")
    for sym in sorted(to_buy):
        q = quotes.get(sym)
        price = q.price if q else 0
        name = q.name if q else ""
        print(f"  {sym} {name}: 现价 {price:.2f}")
    print()

if to_hold:
    print(f"[继续持有] ({len(to_hold)} 只)")
    for sym in sorted(to_hold)[:10]:
        shares = current[sym]
        q = quotes.get(sym)
        if q:
            pnl = (q.price - q.prev_close) / q.prev_close * 100
            value = shares * q.price
            total_value += value
            print(f"  {sym} {q.name}: {shares}股 @ {q.price:.2f} = {value:,.0f} ({pnl:+.1f}%)")
    if len(to_hold) > 10:
        print(f"  ... 还有 {len(to_hold)-10} 只")
    print()

print("=" * 55)
total_value = sum(current.get(s,0) * (quotes.get(s).price if quotes.get(s) else 0) for s in current)
print(f"  持仓总市值(估算): {total_value:,.0f} 元")
print(f"  上次终值: 1,459,683 元")
print("=" * 55)
