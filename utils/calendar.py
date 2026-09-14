"""
A股交易日历 — 基于 cn-stock-holidays 库。
"""

from datetime import date, datetime, timedelta
from typing import List, Optional


try:
    import cn_stock_holidays.data as holiday_data
    from cn_stock_holidays.cn_stock_holidays import get_cached_holidays

    def get_trading_calendar(start: str, end: str) -> List[date]:
        """获取指定区间内的交易日列表。

        Args:
            start: 起始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'

        Returns:
            交易日列表（date 对象）
        """
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()

        # 获取节假日集合
        holidays = set(get_cached_holidays())

        trading_days = []
        current = start_date
        while current <= end_date:
            # 排除周末和节假日
            if current.weekday() < 5 and current not in holidays:
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

except ImportError:
    # 降级：仅排除周末，不考虑节假日
    from loguru import logger

    logger.warning("cn-stock-holidays 未安装，交易日历仅排除周末，不含节假日")

    def get_trading_calendar(start: str, end: str) -> List[date]:
        """降级版：仅排除周末。"""
        start_date = datetime.strptime(start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date()

        trading_days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days


def get_next_trading_day(dt: date) -> date:
    """获取下一个交易日。"""
    dt = dt + timedelta(days=1)
    holidays = set()
    try:
        holidays = set(get_cached_holidays())
    except Exception:
        pass
    while dt.weekday() >= 5 or dt in holidays:
        dt = dt + timedelta(days=1)
    return dt


def get_previous_trading_day(dt: date) -> date:
    """获取上一个交易日。"""
    dt = dt - timedelta(days=1)
    holidays = set()
    try:
        holidays = set(get_cached_holidays())
    except Exception:
        pass
    while dt.weekday() >= 5 or dt in holidays:
        dt = dt - timedelta(days=1)
    return dt


def is_trading_day(dt: date) -> bool:
    """判断是否为交易日。"""
    if dt.weekday() >= 5:
        return False
    try:
        holidays = set(get_cached_holidays())
        return dt not in holidays
    except Exception:
        return True


def get_month_end_trading_days(start: str, end: str) -> List[date]:
    """获取每月最后一个交易日列表（用于月度调仓）。"""
    all_days = get_trading_calendar(start, end)
    month_ends = {}
    for d in all_days:
        month_ends[(d.year, d.month)] = d
    return sorted(month_ends.values())
