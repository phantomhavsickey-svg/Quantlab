"""
数据清洗 — ST/退市处理、停牌检测、涨跌停板识别、T+1掩码。
"""

import pandas as pd
import numpy as np
from loguru import logger


class DataCleaner:
    """A股数据清洗管道。

    处理顺序:
        1. 剔除 ST / *ST 股票
        2. 剔除上市不足 N 天的新股
        3. 标记停牌日期（成交量=0）
        4. 标记涨跌停板
        5. 构建可交易性掩码
    """

    # 不同板块的涨跌停幅度
    LIMIT_RULES = {
        # 主板 (60xxxx, 00xxxx): 10%
        "main": {"limit_pct": 0.10, "prefixes": ("60", "00")},
        # 创业板 (30xxxx): 20%
        "chinext": {"limit_pct": 0.20, "prefixes": ("30",)},
        # 科创板 (688xxx): 20%
        "star": {"limit_pct": 0.20, "prefixes": ("688",)},
    }

    @staticmethod
    def get_board_type(symbol: str) -> str:
        """根据股票代码判断板块类型。"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith("688"):
            return "star"
        elif symbol.startswith("30"):
            return "chinext"
        else:
            return "main"

    @staticmethod
    def get_limit_pct(symbol: str) -> float:
        """获取涨跌停幅度。"""
        for rule in DataCleaner.LIMIT_RULES.values():
            if str(symbol).zfill(6).startswith(rule["prefixes"]):
                return rule["limit_pct"]
        return 0.10

    # ==================== ST 股票过滤 ====================

    @staticmethod
    def filter_st_stocks(symbols_df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
        """剔除 ST / *ST 股票。

        Args:
            symbols_df: 包含股票代码和名称的 DataFrame
            name_col: 名称列名

        Returns:
            过滤后的 DataFrame
        """
        before = len(symbols_df)
        mask = ~symbols_df[name_col].str.contains("ST", na=False)
        result = symbols_df[mask].copy()
        removed = before - len(result)
        if removed > 0:
            logger.info(f"剔除 ST: {before} → {len(result)} (-{removed})")
        return result

    # ==================== 停牌检测 ====================

    @staticmethod
    def mark_suspension(df: pd.DataFrame) -> pd.DataFrame:
        """标记停牌日（成交量为0或换手率为0）。

        添加列:
            - is_suspended: 是否停牌
        """
        df = df.copy()
        df["is_suspended"] = (df.get("成交量", 0) == 0) | \
                             (df.get("换手率", 0) == 0)
        return df

    # ==================== 涨跌停检测 ====================

    @staticmethod
    def mark_price_limits(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """标记涨跌停板日期。

        添加列:
            - prev_close: 前收盘价
            - limit_up: 涨停价
            - limit_down: 跌停价
            - at_limit_up: 是否涨停
            - at_limit_down: 是否跌停

        Args:
            df: OHLCV DataFrame，必须包含 '涨跌幅' 列
            symbol: 股票代码（用于判断涨跌停幅度）

        Returns:
            增加涨跌停标记的 DataFrame
        """
        df = df.copy()
        limit_pct = DataCleaner.get_limit_pct(symbol)

        # 用前一日收盘价计算涨跌停价
        # 注意：q前复权会让历史价格不是真实价格，用涨跌幅来判断更可靠
        # 涨跌停时涨跌幅会恰好等于 ±limit_pct（ST股票为±5%）
        df["at_limit_up"] = (df["涨跌幅"] >= (limit_pct * 100 - 0.5))  # 容差
        df["at_limit_down"] = (df["涨跌幅"] <= (-limit_pct * 100 + 0.5))

        return df

    # ==================== 可交易性掩码 ====================

    @staticmethod
    def build_tradability_mask(df: pd.DataFrame) -> pd.DataFrame:
        """构建可交易性掩码。

        不可交易的情况:
            - 停牌日
            - 涨停日 (买不到)
            - 跌停日 (卖不掉)
            - 成交量异常（为0或NaN）

        添加列:
            - tradable: 是否可交易
        """
        df = df.copy()
        df["tradable"] = True

        # 停牌 / 无成交
        if "is_suspended" in df.columns:
            df.loc[df["is_suspended"], "tradable"] = False
        df.loc[df["成交量"].fillna(0) <= 0, "tradable"] = False

        # 涨跌停不可交易
        if "at_limit_up" in df.columns:
            df.loc[df["at_limit_up"], "tradable"] = False
        if "at_limit_down" in df.columns:
            df.loc[df["at_limit_down"], "tradable"] = False

        # NaN 价格
        for col in ["开盘", "收盘", "最高", "最低"]:
            if col in df.columns:
                df.loc[df[col].isna(), "tradable"] = False

        return df

    # ==================== 新股过滤 ====================

    @staticmethod
    def filter_new_listings(df: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
        """剔除上市不足 min_days 个交易日的次新股。

        Args:
            df: 单只股票的日线数据（至少包含 '日期' 列）
            min_days: 最少上市天数

        Returns:
            过滤后的 DataFrame
        """
        if len(df) < min_days:
            return df.iloc[0:0]  # 全删
        return df.iloc[min_days:]  # 保留第min_days行及之后

    # ==================== 完整清洗管道 ====================

    @staticmethod
    def clean_single_stock(df: pd.DataFrame, symbol: str,
                           min_listed_days: int = 60) -> pd.DataFrame | None:
        """对单只股票执行完整清洗管道。

        Args:
            df: 原始日线数据
            symbol: 股票代码
            min_listed_days: 最少上市天数

        Returns:
            清洗后的 DataFrame，数据不足返回 None
        """
        if df is None or df.empty:
            return None

        df = df.copy()
        df = df.sort_values("日期").reset_index(drop=True)

        # 1. 过滤新股
        df = DataCleaner.filter_new_listings(df, min_listed_days)
        if df.empty:
            return None

        # 2. 停牌标记
        df = DataCleaner.mark_suspension(df)

        # 3. 涨跌停标记
        df = DataCleaner.mark_price_limits(df, symbol)

        # 4. 可交易性掩码
        df = DataCleaner.build_tradability_mask(df)

        return df

    @staticmethod
    def clean_all(data_dict: dict, min_listed_days: int = 60) -> dict:
        """批量清洗。

        Args:
            data_dict: {symbol: DataFrame} 字典
            min_listed_days: 最少上市天数

        Returns:
            {symbol: DataFrame} 清洗后的字典
        """
        cleaned = {}
        removed = 0
        for sym, df in data_dict.items():
            result = DataCleaner.clean_single_stock(df, sym, min_listed_days)
            if result is not None and not result.empty:
                cleaned[sym] = result
            else:
                removed += 1

        logger.info(f"清洗: {len(data_dict)} → {len(cleaned)} (-{removed})")
        return cleaned
