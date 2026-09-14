# -*- coding: utf-8 -*-
"""
每日信号生成器 — 与回测完全一致的因子处理链路。

流程:
    缓存日线(近 lookback 日) → 技术因子 → 缩尾/标准化/滞后(与训练相同)
    → 最新 Walk-Forward 模型预测 → 截面排名 Top-K → 目标组合

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
from models.trainer import LightGBMTrainer


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

    def build_processed_panel(self, asof: str,
                              lookback_days: int) -> pd.DataFrame:
        """构建最近 lookback_days 的因子面板，走与训练相同的预处理管道。"""
        end_ts = pd.Timestamp(asof)
        start_ts = end_ts - pd.Timedelta(days=lookback_days * 2 + 30)
        tech_periods = self.cfg_factors["technical"]

        frames = []
        for sym in self.symbols:
            df = self.cache.get_daily(sym)
            if df is None or len(df) < 60:
                continue
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            df = df[(df["日期"] >= start_ts) & (df["日期"] <= end_ts)]
            if len(df) < 60:
                continue
            f = TechnicalFactors.compute_all(df, tech_periods)
            f["date"] = df["日期"].values
            f["symbol"] = sym
            frames.append(f)

        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"])
        # 与回测相同的处理：截面缩尾 → 截面标准化 → 滞后1期
        processed = FactorProcessor().process(panel, self.cfg_factors)
        return processed

    # ==================== 信号生成 ====================

    def generate(self, asof: str, lookback_days: int = 250) -> pd.Series:
        """生成最新截面的目标持仓。

        Args:
            asof: 基准日期 'YYYY-MM-DD'（一般传今天；数据未更新时取最新截面）
            lookback_days: 回看交易日数

        Returns:
            Series (symbol -> weight)，Top-K 等权
        """
        processed = self.build_processed_panel(asof, lookback_days)
        last_date = processed["date"].max()
        logger.info(f"因子面板最新截面: {last_date.date()}")

        fnames = list(self.trainer.feature_names)
        X = (processed[processed["date"] == last_date]
             .set_index(["date", "symbol"])[fnames])
        if X.empty:
            logger.error("最新截面特征为空")
            return pd.Series(dtype=float)

        preds = self.trainer.model.predict(X.fillna(np.nan))
        scores = pd.Series(preds, index=X.index.get_level_values("symbol"),
                           name="score")

        top_k = self.cfg_backtest["max_positions"]
        top = scores.nlargest(top_k)
        weights = pd.Series(1.0 / top_k, index=top.index, name="weight")
        logger.info(f"目标组合: Top-{top_k}（{len(scores)} 只股票参与排名）")
        return weights

    # ==================== 指令生成与输出 ====================

    def make_orders(self, weights: pd.Series,
                    positions: dict, cash: float,
                    ref_price: dict) -> list[dict]:
        """对比当前持仓生成买卖指令。

        Args:
            weights: 目标组合（symbol -> weight）
            positions: 当前持仓 {symbol: shares}
            cash: 当前可用现金
            ref_price: {symbol: 参考价}（今日收盘价，用于股数估算）

        Returns:
            [{"symbol", "side", "quantity", "ref_price"}]
        """
        target_symbols = set(weights.index)
        orders = []

        # 卖出：持仓中不在目标组合的（含全部可用股数）
        for sym, shares in positions.items():
            if sym not in target_symbols and shares > 0:
                orders.append({
                    "symbol": sym, "side": "sell",
                    "quantity": int(shares),
                    "ref_price": ref_price.get(sym, 0.0),
                })

        # 买入：目标组合（等权近似；已持有的也按目标股数对齐）
        cash_per = cash / max(len(target_symbols), 1)
        lot_size = self.config["market"].get("lot_size", 100)
        for sym in target_symbols:
            price = ref_price.get(sym, 0.0)
            if price <= 0:
                continue
            shares = lot_size * (int(cash_per / price) // lot_size)
            if shares <= 0:
                continue
            orders.append({
                "symbol": sym, "side": "buy",
                "quantity": int(shares), "ref_price": price,
            })
        return orders

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
