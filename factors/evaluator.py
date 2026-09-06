"""
因子评估 — IC 分析、因子相关性、分层回测、IR。
"""

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


class FactorEvaluator:
    """因子有效性评估工具。

    核心指标：
        - Rank IC: Spearman 秩相关系数（因子值 vs 未来收益）
        - ICIR: IC 均值 / IC 标准差
        - IC > 0 比例
        - 因子相关性矩阵
    """

    # ==================== IC 分析 ====================

    @staticmethod
    def rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
        """计算单期的 Rank IC（Spearman 相关系数）。

        Args:
            factor_values: 因子值
            forward_returns: 同期的未来收益率

        Returns:
            Rank IC 值
        """
        valid = factor_values.notna() & forward_returns.notna()
        if valid.sum() < 10:
            return np.nan
        return stats.spearmanr(factor_values[valid],
                               forward_returns[valid])[0]

    @staticmethod
    def pearson_ic(factor_values: pd.Series,
                   forward_returns: pd.Series) -> float:
        """计算单期的 Pearson IC。"""
        valid = factor_values.notna() & forward_returns.notna()
        if valid.sum() < 10:
            return np.nan
        return stats.pearsonr(factor_values[valid],
                              forward_returns[valid])[0]

    @staticmethod
    def ic_series(factor_df: pd.DataFrame, forward_returns: pd.Series,
                  ic_type: str = "rank") -> pd.Series:
        """计算每个因子在每个截面上的 IC 序列。

        Args:
            factor_df: 因子面板 (MultiIndex: date x symbol)
            forward_returns: 未来收益率 Series (index 对齐)
            ic_type: "rank" 或 "pearson"

        Returns:
            DataFrame (index=date, columns=因子名) 每行是该日期的各因子 IC
        """
        ic_func = (FactorEvaluator.rank_ic if ic_type == "rank"
                   else FactorEvaluator.pearson_ic)

        if isinstance(factor_df.index, pd.MultiIndex):
            dates = factor_df.index.get_level_values("date").unique()
        else:
            dates = factor_df["date"].unique()

        results = {}
        for factor_name in factor_df.columns:
            ic_values = {}
            for d in dates:
                if isinstance(factor_df.index, pd.MultiIndex):
                    fv = factor_df.xs(d, level="date")[factor_name]
                else:
                    fv = factor_df[factor_df["date"] == d][factor_name]
                # forward_returns 也取对应日期
                if d in forward_returns.index.get_level_values("date"):
                    fr = forward_returns.xs(d, level="date")
                else:
                    continue
                ic_values[d] = ic_func(fv, fr)
            results[factor_name] = pd.Series(ic_values)

        return pd.DataFrame(results)

    # ==================== IC 汇总统计 ====================

    @staticmethod
    def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
        """对 IC 序列做汇总统计。

        Returns:
            DataFrame 包含每个因子的:
                IC_mean, IC_std, ICIR, IC_positive_ratio, IC_sig(t值)
        """
        summary = pd.DataFrame(index=ic_df.columns)
        summary["IC_mean"] = ic_df.mean()
        summary["IC_std"] = ic_df.std()
        summary["ICIR"] = summary["IC_mean"] / summary["IC_std"]
        summary["IC_positive_ratio"] = (ic_df > 0).mean()
        summary["IC_t_stat"] = summary["IC_mean"] / \
            (summary["IC_std"] / np.sqrt(ic_df.count()))
        summary["IC_significant"] = summary["IC_t_stat"].abs() > 2.0
        return summary.sort_values("ICIR", ascending=False)

    # ==================== 因子相关性 ====================

    @staticmethod
    def factor_correlation(factor_df: pd.DataFrame) -> pd.DataFrame:
        """计算因子间截面相关性矩阵。"""
        # 堆叠所有截面
        if isinstance(factor_df.index, pd.MultiIndex):
            return factor_df.corr()
        else:
            factor_cols = [c for c in factor_df.columns
                           if c not in ["date", "symbol"]]
            return factor_df[factor_cols].corr()

    @staticmethod
    def find_redundant_factors(corr_matrix: pd.DataFrame,
                               threshold: float = 0.7) -> list[tuple]:
        """找出高度相关的因子对。

        Returns:
            [(因子A, 因子B, 相关系数), ...]
        """
        redundant = []
        cols = corr_matrix.columns
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                if abs(corr_matrix.loc[c1, c2]) > threshold:
                    redundant.append((c1, c2, corr_matrix.loc[c1, c2]))
        return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)

    # ==================== 分层回测 ====================

    @staticmethod
    def quantile_returns(factor: pd.Series, forward_returns: pd.Series,
                         n_quantiles: int = 5) -> pd.DataFrame:
        """按因子值分组，计算每组平均收益率。

        Args:
            factor: 因子值 Series
            forward_returns: 前向收益 Series
            n_quantiles: 分组数（默认5组）

        Returns:
            DataFrame (index=分组, columns=[均值收益, 股票数])
        """
        valid = factor.notna() & forward_returns.notna()
        fv = factor[valid]
        fr = forward_returns[valid]

        if len(fv) < n_quantiles * 5:
            return pd.DataFrame()

        # 按因子值排名分组
        quantiles = pd.qcut(fv, n_quantiles, duplicates="drop")

        result = pd.DataFrame({
            "quantile": quantiles,
            "return": fr,
        })

        grouped = result.groupby("quantile")["return"].agg(["mean", "count"])
        return grouped

    @staticmethod
    def quantile_spread(factor: pd.Series, forward_returns: pd.Series,
                        n_quantiles: int = 5) -> float:
        """多空收益差（最高组 - 最低组）。"""
        qr = FactorEvaluator.quantile_returns(factor, forward_returns,
                                               n_quantiles)
        if qr.empty:
            return np.nan
        # Q5 (最高因子值) - Q1 (最低因子值)
        return float(qr["mean"].iloc[-1] - qr["mean"].iloc[0])

    # ==================== 因子报告 ====================

    @staticmethod
    def full_report(factor_df: pd.DataFrame,
                    forward_returns: pd.Series,
                    ic_type: str = "rank") -> dict:
        """生成完整因子评估报告。

        Returns:
            dict with keys:
                - ic_summary: IC 汇总表
                - correlation: 因子相关性矩阵
                - redundant_pairs: 高度相关因子对
                - quantile_spreads: 各因子多空收益差
        """
        # IC 分析
        ic_df = FactorEvaluator.ic_series(factor_df, forward_returns, ic_type)
        ic_summ = FactorEvaluator.ic_summary(ic_df)

        # 相关性
        corr = FactorEvaluator.factor_correlation(factor_df)
        redundant = FactorEvaluator.find_redundant_factors(corr)

        # 分层回测
        spreads = {}
        for col in factor_df.columns:
            if isinstance(factor_df.index, pd.MultiIndex):
                all_fv = factor_df[col]
            else:
                all_fv = factor_df.set_index(["date", "symbol"])[col] \
                    if "date" in factor_df.columns else factor_df[col]

            spreads[col] = FactorEvaluator.quantile_spread(all_fv,
                                                            forward_returns)

        return {
            "ic_summary": ic_summ,
            "correlation": corr,
            "redundant_pairs": redundant,
            "quantile_spreads": pd.Series(spreads).sort_values(ascending=False),
        }
