"""
基本面因子 — PE、PB、ROE、市值等价值和质量因子。
"""

import pandas as pd
import numpy as np
from loguru import logger


class FundamentalFactors:
    """基本面因子计算器。

    数据来源：
        - AKShare stock_zh_a_spot_em() 快照（PE、PB、市值）
        - AKShare stock_financial_analysis_indicator_em() 财报（ROE等）
    """

    # ==================== 估值因子 ====================

    @staticmethod
    def pe_ratio(fundamentals: pd.DataFrame) -> pd.Series:
        """市盈率（PE）。"""
        col = None
        for c in ["pe_dynamic", "市盈率-动态", "市盈率"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            raise KeyError("基本面数据中找不到市盈率列")
        pe = pd.to_numeric(fundamentals[col], errors="coerce")
        # PE 负值设为 NaN（亏损公司）
        pe = pe.where(pe > 0, np.nan)
        return pe

    @staticmethod
    def earnings_yield(fundamentals: pd.DataFrame) -> pd.Series:
        """盈利收益率 = 1/PE（用倒数使因子方向一致：越高越好）。"""
        pe = FundamentalFactors.pe_ratio(fundamentals)
        # 对于负PE，盈利收益率为负
        return 1.0 / pe

    @staticmethod
    def pb_ratio(fundamentals: pd.DataFrame) -> pd.Series:
        """市净率（PB）。"""
        col = None
        for c in ["pb", "市净率"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            raise KeyError("基本面数据中找不到市净率列")
        pb = pd.to_numeric(fundamentals[col], errors="coerce")
        return pb.where(pb > 0, np.nan)

    # ==================== 规模因子 ====================

    @staticmethod
    def market_cap(fundamentals: pd.DataFrame) -> pd.Series:
        """总市值（元）。"""
        col = None
        for c in ["total_market_cap", "总市值"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            raise KeyError("基本面数据中找不到总市值列")
        return pd.to_numeric(fundamentals[col], errors="coerce")

    @staticmethod
    def log_market_cap(fundamentals: pd.DataFrame) -> pd.Series:
        """对数市值 — A股小市值溢价的核心因子。"""
        cap = FundamentalFactors.market_cap(fundamentals)
        return np.log(cap.replace(0, np.nan))

    @staticmethod
    def float_market_cap(fundamentals: pd.DataFrame) -> pd.Series:
        """流通市值。"""
        col = None
        for c in ["float_market_cap", "流通市值"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            # 没有流通市值，用总市值
            return FundamentalFactors.market_cap(fundamentals)
        return pd.to_numeric(fundamentals[col], errors="coerce")

    # ==================== 质量/盈利因子 ====================

    @staticmethod
    def roe(fundamentals: pd.DataFrame) -> pd.Series:
        """净资产收益率 ROE。"""
        col = None
        for c in ["roe", "净资产收益率"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            # 如果快照中没有 ROE，返回 NaN
            return pd.Series(np.nan, index=fundamentals.index)
        return pd.to_numeric(fundamentals[col], errors="coerce")

    # ==================== 动量/技术混合 ====================

    @staticmethod
    def ret_60d(fundamentals: pd.DataFrame) -> pd.Series:
        """60日涨跌幅。"""
        col = None
        for c in ["ret_60d", "60日涨跌幅"]:
            if c in fundamentals.columns:
                col = c
                break
        if col is None:
            return pd.Series(np.nan, index=fundamentals.index)
        return pd.to_numeric(fundamentals[col], errors="coerce")

    # ==================== 批量计算 ====================

    @classmethod
    def compute_all(cls, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """从基本面快照计算所有基本面因子。

        Args:
            fundamentals: AKShare stock_zh_a_spot_em() 返回的 DataFrame
                          必须用股票代码作为 index

        Returns:
            DataFrame（index=symbol, columns=因子）
        """
        factors = pd.DataFrame(index=fundamentals.index)

        # 估值因子
        factors["earnings_yield"] = cls.earnings_yield(fundamentals)
        factors["pb_inverse"] = 1.0 / cls.pb_ratio(fundamentals)  # 1/PB，越高越好

        # 规模因子
        factors["log_market_cap"] = cls.log_market_cap(fundamentals)

        # 质量因子
        factors["roe"] = cls.roe(fundamentals)

        # 动量
        factors["ret_60d"] = cls.ret_60d(fundamentals)

        # 移除全 NaN 列
        factors = factors.dropna(axis=1, how="all")

        # 无穷值处理
        factors = factors.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"基本面因子: {len(factors.columns)} 个, {len(factors)} 只股票")
        return factors
