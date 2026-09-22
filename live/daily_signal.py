# -*- coding: utf-8 -*-
"""
每日信号生成器 — 与回测完全一致的因子处理链路。

流程:
    缓存日线(近 lookback 日) → 技术因子 → 缩尾/标准化/滞后(与训练相同)
    → 最新 Walk-Forward 模型预测 → 截面排名 Top-K → 目标组合
    (position_policy.enabled 时不出名单,出**全截面分数**,买卖由
     utils/position_policy.py 的分数带位状态机决定 —— 与回测同一份实现)

这是回测与实盘一致的唯一信号入口：任何因子口径改动必须先改回测链路。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from data.cache import CacheManager
from factors.technical import TechnicalFactors
from factors.processor import FactorProcessor
from live.orders import make_orders as _make_orders
from live.orders import plan_orders as _plan_orders
from models.trainer import LightGBMTrainer
from utils.market_rules import build_tradable_mask


class DailySignalGenerator:
    """从缓存数据生成当日目标组合。"""

    def __init__(self, config: dict):
        self.config = config
        self.cfg_factors = config["factors"]
        self.cfg_backtest = config["backtest"]
        self.cache = CacheManager(config["cache"]["directory"])
        self.symbols = self.cache.list_cached_symbols()
        self.trainer = self._load_latest_model()

    # ==================== 模型加载 ====================

    def _load_latest_model(self) -> LightGBMTrainer:
        """加载最新的 Walk-Forward 模型（与 config 中 model.type 匹配）。"""
        model_type = self.config["model"]["type"]
        prefix = f"lgb_{model_type}_"
        saved_dir = self.config.get("model", {}).get("save_dir", "models/saved")
        candidates = sorted(
            f for f in os.listdir(saved_dir)
            if f.startswith(prefix) and f.endswith(".txt"))
        if not candidates:
            raise FileNotFoundError(
                f"{saved_dir} 中没有 {prefix}*.txt 模型，请先运行 train")
        path = os.path.join(saved_dir, candidates[-1])
        trainer = LightGBMTrainer(model_type=model_type)
        trainer.load(path)
        logger.info(f"信号模型: {path}")
        return trainer

    # ==================== 因子面板 ====================

    def _load_daily(self, asof: str, lookback_days: int) -> dict:
        """缓存里取窗口内日线,返回 {symbol: DataFrame}。

        因子面板和可交易掩码必须来自同一份切片 —— 两边各读一遍缓存会看到
        两个版本的"最新一天"(增量下载与写盘之间有窗口),分数和门控就错位了。
        """
        end_ts = pd.Timestamp(asof)
        start_ts = end_ts - pd.Timedelta(days=lookback_days * 2 + 30)
        frames = {}
        for sym in self.symbols:
            df = self.cache.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            df = df[(df["日期"] >= start_ts) & (df["日期"] <= end_ts)]
            if len(df) < 60:
                continue
            frames[sym] = df
        return frames

    def build_processed_panel(self, asof: str,
                              lookback_days: int,
                              daily: dict | None = None) -> pd.DataFrame:
        """构建最近 lookback_days 的因子面板，走与训练相同的预处理管道。"""
        tech_periods = self.cfg_factors["technical"]
        frames = []
        for sym, df in (daily if daily is not None
                        else self._load_daily(asof, lookback_days)).items():
            f = TechnicalFactors.compute_all(df, tech_periods)
            f["date"] = df["日期"].values
            f["symbol"] = sym
            frames.append(f)

        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        # 与回测相同的处理：截面缩尾 → 截面标准化 → 滞后1期
        processed = FactorProcessor().process(panel, self.cfg_factors)
        return processed

    def _predict_last(self, processed: pd.DataFrame) -> pd.Series:
        """最新截面的全截面预测,索引 (date, symbol)。"""
        last_date = processed["date"].max()
        logger.info(f"因子面板最新截面: {pd.Timestamp(last_date).date()}")

        fnames = list(self.trainer.feature_names)
        X = (processed[processed["date"] == last_date]
             .set_index(["date", "symbol"])[fnames])
        if X.empty:
            logger.error("最新截面特征为空")
            return pd.Series(dtype=float,
                             index=pd.MultiIndex.from_arrays(
                                 [[], []], names=["date", "symbol"]))
        preds = self.trainer.model.predict(X.fillna(np.nan))
        return pd.Series(preds, index=X.index, name="score")

    # ==================== 信号生成 ====================

    def generate(self, asof: str, lookback_days: int = 250) -> pd.Series:
        """生成最新截面的目标持仓。

        Args:
            asof: 基准日期 'YYYY-MM-DD'（一般传今天；数据未更新时取最新截面）
            lookback_days: 回看交易日数

        Returns:
            Series (symbol -> weight)，Top-K 等权
        """
        daily = self._load_daily(asof, lookback_days)
        processed = self.build_processed_panel(asof, lookback_days, daily)
        scores = self._predict_last(processed)
        if scores.empty:
            return pd.Series(dtype=float)

        last_date = scores.index.get_level_values("date")[0]
        # 信号日没有成交的股票不占 Top-K 名额(与回测 build_tradable_mask 同口径)
        tradable = build_tradable_mask(daily, [last_date])
        if len(tradable):
            scores = scores[scores.index.isin(tradable.index)]
        if scores.empty:
            logger.warning("最新截面全部不可交易,无目标组合")
            return pd.Series(dtype=float)

        top_k = self.cfg_backtest["max_positions"]
        top = scores.nlargest(top_k)
        weights = pd.Series(1.0 / top_k, index=top.index.get_level_values("symbol"),
                            name="weight")
        logger.info(f"目标组合: Top-{top_k}（{len(scores)} 只股票参与排名）")
        return weights

    def signal_scores(self, asof: str,
                      lookback_days: int = 250) -> pd.Series:
        """分数带位策略的输入:最新截面的**全截面**分数(symbol → score)。

        与 generate() 的区别是这条路径不截 Top-K —— 掉出名单不等于跌破清仓线,
        拿名单当持仓会把"分数仍然很高但排名下降"错读成卖出信号。
        门控仍按信号日口径(当日无成交不许建仓)。
        """
        daily = self._load_daily(asof, lookback_days)
        processed = self.build_processed_panel(asof, lookback_days, daily)
        scores = self._predict_last(processed)
        if scores.empty:
            return pd.Series(dtype=float)

        last_date = scores.index.get_level_values("date")[0]
        tradable = build_tradable_mask(daily, [last_date])
        kept = scores[scores.index.isin(tradable.index)] if len(tradable) else scores
        out = pd.Series(kept.values,
                        index=kept.index.get_level_values("symbol"), name="score")
        logger.info(f"全截面分数: {len(out)} 只(门控摘掉 {len(scores) - len(out)} 只"
                    f" 信号日无成交),最高 {out.max() if len(out) else float('nan'):.4f}")
        return out

    # ==================== 指令生成与输出 ====================

    def make_policy_orders(self, scores: pd.Series, positions: dict,
                           cash: float, ref_price: dict, states: dict,
                           policy, asof=None):
        """分数带位策略版指令(与回测引擎同一个 utils.position_policy.plan)。

        Returns:
            (orders, PolicyPlan);成交回报到手后调用 apply_fills 推进 states。
        """
        cfg_market = self.config.get("market", {})
        return _plan_orders(scores, positions, cash, ref_price, states, policy,
                            lot_size=cfg_market.get("lot_size", 100), asof=asof)

    def make_orders(self, weights: pd.Series,
                    positions: dict, cash: float,
                    ref_price: dict) -> list[dict]:
        """对比当前持仓生成买卖指令(委托给 live.orders,与回测同一份算式)。

        Args:
            weights: 目标组合（symbol -> weight）
            positions: 当前持仓 {symbol: shares}
            cash: 当前可用现金
            ref_price: {symbol: 参考价}（今日收盘价，用于股数估算）

        Returns:
            [{"symbol", "side", "quantity", "ref_price"}]
        """
        cfg_market = self.config.get("market", {})
        return _make_orders(
            weights, positions, cash, ref_price,
            lot_size=cfg_market.get("lot_size", 100),
            fee_rate_buy=float(cfg_market.get("commission_rate", 0.0))
            + float(cfg_market.get("slippage_rate", 0.0)))

    def export_target(self, weights: pd.Series, path: str):
        """输出目标组合 CSV。"""
        df = weights.reset_index()
        df.columns = ["symbol", "weight"]
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"目标组合已输出: {path}")

    def export_orders(self, orders: list[dict], path: str):
        """输出调仓指令 CSV。"""
        pd.DataFrame(orders).to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"调仓指令已输出: {path}（共 {len(orders)} 笔）")
