"""
因子处理 — 缩尾、标准化、中性化、滞后处理。
防止过拟合和未来函数的关键模块。
"""

import pandas as pd
import numpy as np
from loguru import logger


class FactorProcessor:
    """因子后处理管道。

    推荐处理顺序（Barra 风格）:
        1. winsorize    — 缩尾去极值
        2. normalize    — 截面标准化
        3. neutralize   — 行业/市值中性化（可选）
        4. lag          — 滞后对齐（防未来函数）
    """

    #: 风格轴，不是因子。无论 neutralize 开不开，都不允许流到特征矩阵里 ——
    #: 下游十几处 "factor_cols = 除 date/symbol 外的所有列" 会把它当第 26 个
    #: 特征喂进模型，于是"关掉中性化"和"打开中性化"两组数字比的不再是同一件事。
    STYLE_COLS = ("log_float_cap", "log_market_cap")

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
        """单截面中性化。

        用线性回归残差：factor = β·neutralizers + ε
        返回 ε，即去除 neutralizers 线性影响后的纯 alpha。

        Args:
            factor: 待中性化的因子 Series
            neutralizers: 中性化变量 DataFrame（行业哑变量 + log市值）

        Returns:
            中性化后的因子。缺中性化变量的行保留原值（该列此前已截面标准化，
            均值本就为 0，退化有限）；因子本身是 NaN 的行仍是 NaN。
        """
        df = neutralizers.copy()
        df["_factor_"] = factor.reindex(df.index).to_numpy(dtype=float)
        return FactorProcessor.neutralize_cross_sectional(
            df, list(neutralizers.columns))["_factor_"]

    @staticmethod
    def neutralize_cross_sectional(factor_df: pd.DataFrame,
                                   neutralizer_cols: list[str]) -> pd.DataFrame:
        """按日期分组的截面中性化。

        Args:
            factor_df: 含 date 和 symbol 列（或 MultiIndex）的因子面板数据
            neutralizer_cols: 用于中性化的列名列表

        Returns:
            中性化后的 DataFrame（中性化变量自身原样保留，剔不剔除由调用方决定）

        实现要点：面板一次转成 numpy，按日切片解最小二乘，最后整体写回。
        早先的写法是「逐日逐因子调 sklearn，再逐格 df.loc[...]=」，在上百万行
        的面板上慢到不可用；而且 sklearn ≥1.4 起拒绝含 NaN 的 predict 输入，
        所以只能把有中性化变量的行单独挑出来拟合。
        """
        result = factor_df.copy()
        if isinstance(result.index, pd.MultiIndex):
            dates = np.asarray(result.index.get_level_values("date"))
        elif "date" in result.columns:
            dates = np.asarray(result["date"])
        else:
            dates = np.zeros(len(result))          # 只有一个截面

        factor_cols = [c for c in result.columns
                       if c not in ["date", "symbol"] + list(neutralizer_cols)]
        X_all = np.asarray(result[neutralizer_cols], dtype=float)
        Y = np.asarray(result[factor_cols], dtype=float).copy()
        ok_x = ~np.isnan(X_all).any(axis=1)

        codes = pd.factorize(dates)[0]
        for g in range(int(codes.max()) + 1):
            rows = np.flatnonzero(codes == g)
            use = rows[ok_x[rows]]
            if len(use) < 10:
                continue
            A = np.column_stack([np.ones(len(use)), X_all[use]])
            for j in range(len(factor_cols)):
                yj = Y[use, j]
                m = ~np.isnan(yj)
                if m.sum() < 10:
                    continue
                Am, ym = A[m], yj[m]
                beta = np.linalg.lstsq(Am, ym, rcond=None)[0]
                Y[use[m], j] = ym - Am @ beta
        result[factor_cols] = Y
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
            if not available:
                # 只 warning 的话,flag 打开了却什么都没发生 —— 这种"静默空转"
                # 比直接报错坏得多(所有下游数字都按"已中性化"被记录)
                raise ValueError(
                    f"neutralize=true 但中性化变量 {neutralizer_cols} 不在面板里;"
                    f" 现有列 {sorted(result.columns)[:6]}...")
            result = cls.neutralize_cross_sectional(result, available)
            logger.info(f"已按 {available} 做截面中性化")

        # 4. 滞后
        result = cls.lag_factor_panel(result, periods=1)

        # 5. 风格轴出面板（不管第 3 步跑没跑）
        style = [c for c in cls.STYLE_COLS if c in result.columns]
        if style:
            result = result.drop(columns=style)
            logger.info(f"已剔除风格变量列 {style}（不是因子，不进特征矩阵）")

        logger.info("因子处理完成: 缩尾 → 标准化 → 滞后")
        return result
