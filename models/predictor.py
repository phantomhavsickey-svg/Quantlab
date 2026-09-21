"""
预测器 — 用训练好的模型生成选股信号。
"""

import pandas as pd
import numpy as np
from loguru import logger


def _normalize_multiindex(mi: pd.MultiIndex) -> pd.MultiIndex:
    """规范化 MultiIndex 层级：date → datetime64[ns]，symbol → object。

    不同来源构造的 MultiIndex 层级 dtype 可能不一致（str vs object、
    datetime64[s] vs datetime64[ns] vs object-of-Timestamp），混用时 `xs`
    会静默失配 —— 过滤条件看着加了，实际一条都没生效。
    """
    dates = pd.to_datetime(mi.get_level_values(0)).astype("datetime64[ns]")
    syms = mi.get_level_values(1).astype(str).astype(object)
    return pd.MultiIndex.from_arrays([dates, syms], names=mi.names)


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
            tradable: 可交易性布尔 Series (MultiIndex: date x symbol，只含 True
                      项，缺席即不可选)。口径必须是**信号日当天已知**的信息（当日
                      有没有成交），不能塞成交日的涨跌停状态 —— 那是未来函数。
                      任何"识别到某种状态才允许买入"的门控都从这里进。

        Returns:
            Series (MultiIndex: date x symbol)，值为持仓权重
        """
        if isinstance(predictions.index, pd.MultiIndex):
            predictions = predictions.copy()
            predictions.index = _normalize_multiindex(predictions.index)
        if tradable is not None and isinstance(tradable.index, pd.MultiIndex):
            tradable = tradable.copy()
            tradable.index = _normalize_multiindex(tradable.index)

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
                # 掩码可以只含 True 项（缺席即不可选，build_tradable_mask 就是这样），
                # 也可以是全量 bool 序列；reindex 之后两种都按"缺席=False"处理
                day_tradable = day_tradable.reindex(day_preds.index) \
                    .fillna(False).astype(bool)
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
        # 排名/选股/合并三步必须用同一份索引 dtype,否则第 3 步的 reindex
        # 会静默失配,权重整列变 0
        if isinstance(predictions.index, pd.MultiIndex):
            predictions = predictions.copy()
            predictions.index = _normalize_multiindex(predictions.index)
        if tradable is not None and isinstance(tradable.index, pd.MultiIndex):
            tradable = tradable.copy()
            tradable.index = _normalize_multiindex(tradable.index)

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
