"""
因子处理 — 缩尾、标准化、中性化、滞后处理。
防止过拟合和未来函数的关键模块。
"""

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression


class FactorProcessor:
    """因子后处理管道。

    推荐处理顺序（Barra 风格）:
        1. winsorize    — 缩尾去极值
        2. normalize    — 截面标准化
        3. neutralize   — 行业/市值中性化（可选）
        4. lag          — 滞后对齐（防未来函数）
    """

    # ==================== 缩尾 ====================

    @staticmethod
    def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
        """缩尾：将超出分位数的值裁剪到边界。

        Args:
            series: 因子值
            limits: (下分位数, 上分位数)，默认 (1%, 99%)

        Returns:
            缩尾后的 Series
        """
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower, upper)

    @staticmethod
    def winsorize_cross_sectional(factor_df: pd.DataFrame,
                                  limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
        """按日期分组的截面缩尾。

        Args:
            factor_df: MultiIndex (date, symbol) 或 含 'date' 列的 DataFrame
            limits: 缩尾分位数

        Returns:
            缩尾后的 DataFrame
        """
        if "date" in factor_df.columns:
            result = factor_df.copy()
            factor_cols = [c for c in result.columns
                           if c not in ["date", "symbol"]]
            for col in factor_cols:
                result[col] = result.groupby("date")[col].transform(
                    lambda x: FactorProcessor.winsorize(x, limits)
                )
            return result
        elif isinstance(factor_df.index, pd.MultiIndex):
            result = factor_df.copy()
            for col in result.columns:
                result[col] = result.groupby(level="date")[col].transform(
                    lambda x: FactorProcessor.winsorize(x, limits)
                )
            return result
        else:
            # 单截面
            result = factor_df.copy()
            for col in result.columns:
                result[col] = FactorProcessor.winsorize(result[col], limits)
            return result

    # ==================== 标准化 ====================

    @staticmethod
    def zscore(series: pd.Series) -> pd.Series:
        """Z-Score 标准化: (x - mean) / std。"""
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    @staticmethod
    def rank_normalize(series: pd.Series) -> pd.Series:
        """Rank 归一化: 将排名映射到 [-1, 1]。"""
        ranked = series.rank(pct=True)  # 0 到 1
        return (ranked - 0.5) * 2.0     # -1 到 1

    @staticmethod
    def normalize_cross_sectional(factor_df: pd.DataFrame,
                                  method: str = "zscore") -> pd.DataFrame:
        """按日期分组的截面标准化。

        Args:
            factor_df: MultiIndex (date, symbol) 或 含 'date' 列的 DataFrame
            method: "zscore" 或 "rank"

        Returns:
            标准化后的 DataFrame
        """
        result = factor_df.copy()

        if method == "zscore":
            norm_func = FactorProcessor.zscore
        elif method == "rank":
            norm_func = FactorProcessor.rank_normalize
        else:
            raise ValueError(f"未知标准化方法: {method}")

        if "date" in result.columns:
            factor_cols = [c for c in result.columns
                           if c not in ["date", "symbol"]]
            for col in factor_cols:
                result[col] = result.groupby("date")[col].transform(norm_func)
        elif isinstance(result.index, pd.MultiIndex):
            for col in result.columns:
                result[col] = result.groupby(level="date")[col].transform(norm_func)
        else:
            for col in result.columns:
                result[col] = norm_func(result[col])

        return result

    # ==================== 中性化 ====================

    @staticmethod
    def neutralize(factor: pd.Series, neutralizers: pd.DataFrame) -> pd.Series:
        """行业/市值中性化。

        用线性回归残差：factor = β·neutralizers + ε
        返回 ε，即去除 neutralizers 线性影响后的纯 alpha。

        Args:
            factor: 待中性化的因子 Series
            neutralizers: 中性化变量 DataFrame（行业哑变量 + log市值）

        Returns:
            中性化后的因子
        """
        y = factor.values.reshape(-1, 1)
        X = neutralizers.values

        # 剔除 NaN
        valid = ~np.isnan(y.flatten()) & ~np.isnan(X).any(axis=1)
        if valid.sum() < 10:
            return factor.copy()

        model = LinearRegression()
        model.fit(X[valid], y[valid])
        residuals = y - model.predict(X)
        return pd.Series(residuals.flatten(), index=factor.index)

    @staticmethod
    def neutralize_cross_sectional(factor_df: pd.DataFrame,
                                   neutralizer_cols: list[str]) -> pd.DataFrame:
        """按日期分组的截面中性化。

        Args:
            factor_df: 含 date 和 symbol 列（或 MultiIndex）的因子面板数据
            neutralizer_cols: 用于中性化的列名列表

        Returns:
            中性化后的 DataFrame
        """
        result = factor_df.copy()
        factor_cols = [c for c in result.columns
                       if c not in ["date", "symbol"] + neutralizer_cols]

        if "date" in result.columns:
            for date, group in result.groupby("date"):
                neutralizers = group[neutralizer_cols]
                for fcol in factor_cols:
                    result.loc[group.index, fcol] = \
                        FactorProcessor.neutralize(group[fcol], neutralizers)
        elif isinstance(result.index, pd.MultiIndex):
            for date in result.index.get_level_values("date").unique():
                mask = result.index.get_level_values("date") == date
                group = result.loc[mask]
                neutralizers = group[neutralizer_cols]
                for fcol in factor_cols:
                    result.loc[mask, fcol] = \
                        FactorProcessor.neutralize(group[fcol], neutralizers)

        return result

    # ==================== 滞后处理 ====================

    @staticmethod
    def lag(series: pd.Series, periods: int = 1) -> pd.Series:
        """将因子值滞后 N 期，防止未来函数。

        核心约定：
            用 T 日收盘价算出的因子 → 预测 T+1 日收益率
            所以因子值必须 shift(1)，确保 T 日因子对应的是 T-1 日已知的信息。
            实际上 shift 发生在构建数据集时（dataset.py），
            这里的 lag 用于因子面板内部的延迟对齐。
        """
        return series.shift(periods)

    @staticmethod
    def lag_factor_panel(factor_df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """对因子面板数据进行滞后。

        如果是 MultiIndex (date, symbol) 面板：
            按 symbol 分组后 shift，确保每只股票的因子独立滞后。
        """
        result = factor_df.copy()
        factor_cols = [c for c in result.columns
                       if c not in ["date", "symbol"]]

        if "date" in result.columns and "symbol" in result.columns:
            result = result.sort_values(["symbol", "date"])
            for col in factor_cols:
                result[col] = result.groupby("symbol")[col].shift(periods)
        elif isinstance(result.index, pd.MultiIndex):
            for col in factor_cols:
                result[col] = result.groupby(level="symbol")[col].shift(periods)
        else:
            for col in factor_cols:
                result[col] = result[col].shift(periods)

        return result

    # ==================== 完整管道 ====================

    @classmethod
    def process(cls, factor_df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """一键运行完整因子处理管道。

        Args:
            factor_df: 原始因子面板
            config: 处理配置

        Returns:
            标准化、可选中性化、滞后后的因子面板
        """
        proc_config = config.get("processing", {})
        result = factor_df.copy()

        # 1. 缩尾
        limits = tuple(proc_config.get("winsorize_limits", [0.01, 0.99]))
        result = cls.winsorize_cross_sectional(result, limits=limits)

        # 2. 标准化
        method = proc_config.get("normalize_method", "zscore")
        result = cls.normalize_cross_sectional(result, method=method)

        # 3. 中性化（可选）
        if proc_config.get("neutralize", False):
            neutralizer_cols = proc_config.get("neutralizers",
                                                ["log_market_cap"])
            available = [c for c in neutralizer_cols if c in result.columns]
            if available:
                result = cls.neutralize_cross_sectional(result, available)
            else:
                logger.warning("中性化跳过：指定的列不在因子面板中")

        # 4. 滞后
        result = cls.lag_factor_panel(result, periods=1)

        logger.info("因子处理完成: 缩尾 → 标准化 → 滞后")
        return result
