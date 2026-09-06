"""
技术因子 — 基于价格和成交量的技术面 alpha 因子。
所有因子输入都是 pandas Series，输出也是 Series（对齐索引）。
"""

import pandas as pd
import numpy as np
from loguru import logger


class TechnicalFactors:
    """技术因子计算器。

    约定：
        - close: 收盘价 Series（已按日期升序排列）
        - high, low, open_: OHLC 价格 Series
        - volume: 成交量 Series
        - turnover: 换手率 Series
        - period: 窗口期（交易日数）
    """

    # ==================== 动量类 ====================

    @staticmethod
    def momentum(close: pd.Series, period: int) -> pd.Series:
        """N日收益率。"""
        return close.pct_change(period)

    @staticmethod
    def momentum_log(close: pd.Series, period: int) -> pd.Series:
        """N日对数收益率。"""
        return np.log(close / close.shift(period))

    @staticmethod
    def momentum_reversal(close: pd.Series, short_period: int,
                          long_period: int) -> pd.Series:
        """短期动量 vs 长期反转 = short_ret - long_ret。"""
        short_ret = close.pct_change(short_period)
        long_ret = close.pct_change(long_period)
        return short_ret - long_ret

    # ==================== 波动率类 ====================

    @staticmethod
    def volatility(close: pd.Series, period: int) -> pd.Series:
        """N日年化波动率（基于日收益率标准差）。"""
        daily_ret = close.pct_change()
        return daily_ret.rolling(period).std() * np.sqrt(252)

    @staticmethod
    def downside_volatility(close: pd.Series, period: int) -> pd.Series:
        """下行波动率（只计入负收益日的波动）。"""
        daily_ret = close.pct_change()
        neg_ret = daily_ret.copy()
        neg_ret[neg_ret > 0] = 0
        return neg_ret.rolling(period).std() * np.sqrt(252)

    @staticmethod
    def max_drawdown(close: pd.Series, period: int) -> pd.Series:
        """N日内最大回撤（正值表示亏损）。"""
        rolling_high = close.rolling(period, min_periods=1).max()
        drawdown = (close - rolling_high) / rolling_high
        return drawdown  # 负值 = 浮亏，越大越好（越接近0）

    # ==================== 成交量/流动性类 ====================

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int) -> pd.Series:
        """量比 = 当日成交量 / N日均量。"""
        avg_volume = volume.rolling(period).mean()
        return volume / avg_volume.replace(0, np.nan)

    @staticmethod
    def turnover_avg(turnover: pd.Series, period: int) -> pd.Series:
        """N日平均换手率。"""
        return turnover.rolling(period).mean()

    @staticmethod
    def turnover_std(turnover: pd.Series, period: int) -> pd.Series:
        """N日换手率波动（流动性风险指标）。"""
        return turnover.rolling(period).std()

    @staticmethod
    def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
        """量价趋势 (VPT) = 累计(涨跌幅 * 成交量)。"""
        pct = close.pct_change()
        vpt = (pct * volume).cumsum()
        return vpt

    # ==================== RSI / MACD / 布林带 ====================

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标 (RSI)。"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> tuple:
        """MACD 指标。

        Returns:
            (MACD线, 信号线, 柱状图) 三个 Series
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def macd_histogram(close: pd.Series) -> pd.Series:
        """MACD柱状图（简化接口）。"""
        _, _, hist = TechnicalFactors.macd(close)
        return hist

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20,
                        num_std: float = 2.0) -> tuple:
        """布林带。

        Returns:
            (上轨, 中轨, 下轨)
        """
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def bollinger_position(close: pd.Series, period: int = 20) -> pd.Series:
        """价格在布林带中的位置: (close - lower) / (upper - lower)。"""
        upper, middle, lower = TechnicalFactors.bollinger_bands(close, period)
        return (close - lower) / (upper - lower).replace(0, np.nan)

    @staticmethod
    def bollinger_width(close: pd.Series, period: int = 20) -> pd.Series:
        """布林带宽度: (upper - lower) / middle。"""
        upper, middle, lower = TechnicalFactors.bollinger_bands(close, period)
        return (upper - lower) / middle.replace(0, np.nan)

    # ==================== 价格形态类 ====================

    @staticmethod
    def price_position(close: pd.Series, period: int) -> pd.Series:
        """价格在N日内的相对位置: (close - low) / (high - low)。"""
        roll_high = close.rolling(period).max()
        roll_low = close.rolling(period).min()
        return (close - roll_low) / (roll_high - roll_low).replace(0, np.nan)

    @staticmethod
    def ma_divergence(close: pd.Series, fast: int, slow: int) -> pd.Series:
        """均线偏离: (MA_fast - MA_slow) / MA_slow。"""
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        return (ma_fast - ma_slow) / ma_slow.replace(0, np.nan)

    @staticmethod
    def skewness(close: pd.Series, period: int) -> pd.Series:
        """N日收益率偏度（正偏=大概率有异常正收益）。"""
        daily_ret = close.pct_change()
        return daily_ret.rolling(period).skew()

    @staticmethod
    def kurtosis(close: pd.Series, period: int) -> pd.Series:
        """N日收益率峰度（高峰=肥尾风险）。"""
        daily_ret = close.pct_change()
        return daily_ret.rolling(period).kurt()

    # ==================== 批量计算 ====================

    @classmethod
    def compute_all(cls, df: pd.DataFrame, periods: dict) -> pd.DataFrame:
        """对单只股票计算所有技术因子。

        Args:
            df: OHLCV DataFrame（必须含 开盘/收盘/最高/最低/成交量/换手率）
            periods: 参数配置 dict，如 {"momentum": [5,10,20], ...}

        Returns:
            DataFrame，每列一个因子
        """
        close = df["收盘"]
        high = df["最高"]
        low = df["最低"]
        open_ = df["开盘"]
        volume = df["成交量"]
        turnover = df.get("换手率", pd.Series(np.nan, index=df.index))

        factors = pd.DataFrame(index=df.index)

        # 动量因子
        for p in periods.get("momentum_periods", [5, 10, 20, 60]):
            factors[f"momentum_{p}d"] = cls.momentum(close, p)

        # 反转因子 (5日 vs 60日)
        factors["momentum_5_60_reversal"] = cls.momentum_reversal(close, 5, 60)

        # 波动率因子
        for p in periods.get("volatility_periods", [10, 20, 60]):
            factors[f"volatility_{p}d"] = cls.volatility(close, p)

        factors["downside_vol_20d"] = cls.downside_volatility(close, 20)

        # 回撤因子
        factors["max_dd_60d"] = cls.max_drawdown(close, 60)

        # 量价因子
        for p in periods.get("volume_ratio_periods", [5, 20]):
            factors[f"volume_ratio_{p}d"] = cls.volume_ratio(volume, p)

        if not turnover.isna().all():
            factors["turnover_avg_5d"] = cls.turnover_avg(turnover, 5)
            factors["turnover_std_20d"] = cls.turnover_std(turnover, 20)

        # 技术指标
        factors["rsi_14"] = cls.rsi(close, 14)
        factors["macd_hist"] = cls.macd_histogram(close)
        factors["bb_position"] = cls.bollinger_position(close, 20)
        factors["bb_width"] = cls.bollinger_width(close, 20)

        # 价格形态
        factors["price_pos_20d"] = cls.price_position(close, 20)
        factors["price_pos_60d"] = cls.price_position(close, 60)
        factors["ma_div_5_20"] = cls.ma_divergence(close, 5, 20)
        factors["ma_div_20_60"] = cls.ma_divergence(close, 20, 60)
        factors["skewness_20d"] = cls.skewness(close, 20)
        factors["kurtosis_20d"] = cls.kurtosis(close, 20)

        # 移除全 NaN 列
        factors = factors.dropna(axis=1, how="all")

        return factors
