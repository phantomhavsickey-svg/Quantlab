"""
预测器 — 用训练好的模型生成选股信号。
"""

import pandas as pd
import numpy as np
from loguru import logger


class Predictor:
    """将模型预测转化为交易信号。

    流程:
        1. 输入因子面板 → 模型预测概率/收益率
        2. 按截面排名 → 选出 Top-K 股票
        3. 生成交易信号 (buy/sell/hold)
    """

    def __init__(self, trainer=None, top_k: int = 30,
                 position_sizing: str = "equal_weight"):
        """
        Args:
            trainer: LightGBMTrainer 实例（含已训练模型）；
                     为 None 时使用 generate_signals_from_series 传入外部预测
            top_k: 每期持仓股票数
            position_sizing: 权重分配方式
                - "equal_weight": 等权
                - "signal_strength": 按预测概率加权
        """
        self.trainer = trainer
        self.top_k = top_k
        self.position_sizing = position_sizing

    # ==================== 预测 ====================

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """对特征矩阵做预测。

        Args:
            X: 特征矩阵 (MultiIndex: date x symbol 或普通 DataFrame)

        Returns:
            Series，值为预测概率/收益率
        """
        if self.trainer.model is None:
            raise RuntimeError("模型未加载，请先训练或加载模型")

        # 对齐列
        X_aligned = X[self.trainer.feature_names].fillna(np.nan)
        preds = self.trainer.model.predict(X_aligned)

        return pd.Series(preds, index=X.index, name="prediction")

    # ==================== 排名和选股 ====================

    def rank_predictions(self, predictions: pd.Series) -> pd.Series:
        """对预测值做截面降序排名（高分在前）。"""
        if isinstance(predictions.index, pd.MultiIndex):
            ranked = predictions.groupby(level="date").rank(ascending=False)
        else:
            ranked = predictions.rank(ascending=False)
        return ranked

    def select_top_k(self, predictions: pd.Series,
                     tradable: pd.Series | None = None) -> pd.Series:
        """对每个截面选出 Top-K 股票。

        Args:
            predictions: 预测值 Series (MultiIndex: date x symbol)
            tradable: 可交易性布尔 Series（同索引），True=可交易

        Returns:
            Series (MultiIndex: date x symbol)，值为持仓权重
        """
        if isinstance(predictions.index, pd.MultiIndex):
            dates = predictions.index.get_level_values("date").unique()
        else:
            dates = predictions["date"].unique() \
                if "date" in dir(predictions) else [predictions.index]

        weights_list = []

        for d in sorted(dates):
            if isinstance(predictions.index, pd.MultiIndex):
                day_preds = predictions.xs(d, level="date")
            else:
                day_preds = predictions.loc[predictions.index.get_level_values("date") == d]

            if day_preds.empty:
                continue

            # 过滤不可交易股票
            if tradable is not None:
                if isinstance(tradable.index, pd.MultiIndex):
                    day_tradable = tradable.xs(d, level="date")
                else:
                    day_tradable = tradable.loc[tradable.index.get_level_values("date") == d]
                day_preds = day_preds[day_tradable]

            if day_preds.empty:
                continue

            # 选出 Top-K
            top_symbols = day_preds.nlargest(self.top_k).index

            # 计算权重
            if self.position_sizing == "equal_weight":
                weight = 1.0 / self.top_k
                for sym in top_symbols:
                    weights_list.append({
                        "date": d,
                        "symbol": sym,
                        "weight": weight,
                        "score": day_preds.get(sym, np.nan),
                    })
            elif self.position_sizing == "signal_strength":
                top_preds = day_preds.loc[top_symbols]
                total = top_preds.sum()
                if total > 0:
                    weights = top_preds / total
                else:
                    weights = pd.Series(1.0 / len(top_symbols),
                                        index=top_symbols.index)
                for sym, w in weights.items():
                    weights_list.append({
                        "date": d,
                        "symbol": sym,
                        "weight": w,
                        "score": day_preds.get(sym, np.nan),
                    })

        if not weights_list:
            logger.warning("没有生成任何选股信号")
            return pd.Series(dtype=float)

        weights_df = pd.DataFrame(weights_list)
        return weights_df.set_index(["date", "symbol"])["weight"]

    # ==================== 信号生成 ====================

    def generate_signals(self, X: pd.DataFrame,
                         tradable: pd.Series | None = None) -> pd.DataFrame:
        """完整信号生成管道。

        Args:
            X: 因子特征矩阵
            tradable: 可交易性掩码

        Returns:
            DataFrame (MultiIndex: date x symbol):
                - weight: 目标权重
                - score: 预测分数
                - rank: 截面排名
        """
        if self.trainer is None:
            raise RuntimeError(
                "未加载模型，请使用 generate_signals_from_series 传入外部预测")

        # 1. 预测
        predictions = self.predict(X)
        logger.info(f"预测完成: {len(predictions)} 条")
        return self._signals_from_predictions(predictions, tradable)

    def generate_signals_from_series(self, predictions: pd.Series,
                                     tradable: pd.Series | None = None
                                     ) -> pd.DataFrame:
        """从外部预测值生成信号（Walk-Forward 样本外预测等场景）。

        Args:
            predictions: Series (MultiIndex: date x symbol)，值为预测收益率/概率
            tradable: 可交易性布尔 Series（同索引）

        Returns:
            与 generate_signals 相同的信号 DataFrame
        """
        return self._signals_from_predictions(predictions, tradable)

    def _signals_from_predictions(self, predictions: pd.Series,
                                  tradable: pd.Series | None = None
                                  ) -> pd.DataFrame:
        # 1. 排名
        rankings = self.rank_predictions(predictions)

        # 2. 选股 + 权重
        weights = self.select_top_k(predictions, tradable)

        # 3. 合并
        signals = pd.DataFrame({
            "score": predictions,
            "rank": rankings,
        })
        signals["weight"] = weights.reindex(signals.index).fillna(0.0)

        logger.info(f"信号生成完毕: {len(weights)} 条持仓信号")
        return signals
