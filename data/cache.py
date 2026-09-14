"""
Parquet 缓存管理 — 避免重复下载，本地缓存股票日线数据。
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger


class CacheManager:
    """管理本地 Parquet 缓存。

    目录结构:
        data/cache/
        ├── daily/           # 日线数据（每只股票一个 parquet）
        │   ├── 000001.parquet
        │   └── 600000.parquet
        ├── fundamentals/    # 基本面快照（按日期分文件）
        │   └── 2024-01-01.parquet
        └── calendar.parquet # 交易日历缓存
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.daily_dir = self.cache_dir / "daily"
        self.fundamentals_dir = self.cache_dir / "fundamentals"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保缓存目录存在。"""
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)

    # ---- 日线数据 ----

    def get_daily(self, symbol: str) -> pd.DataFrame | None:
        """读取某只股票的缓存日线。"""
        path = self.daily_dir / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            logger.debug(f"缓存命中: {symbol} ({len(df)} 条)")
            return df
        return None

    def put_daily(self, symbol: str, df: pd.DataFrame):
        """写入日线缓存。"""
        path = self.daily_dir / f"{symbol}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        logger.debug(f"缓存写入: {symbol} ({len(df)} 条)")

    def update_daily(self, symbol: str, new_df: pd.DataFrame):
        """增量更新日线缓存（append新数据并去重）。"""
        existing = self.get_daily(symbol)
        if existing is not None:
            # 合并去重（以日期为key）
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined["日期"] = pd.to_datetime(combined["日期"])
            combined = combined.drop_duplicates(subset=["日期"], keep="last")
            combined = combined.sort_values("日期").reset_index(drop=True)
            self.put_daily(symbol, combined)
        else:
            self.put_daily(symbol, new_df)

    def is_daily_stale(self, symbol: str, max_age_days: int = 1) -> bool:
        """判断日线缓存是否过期。"""
        path = self.daily_dir / f"{symbol}.parquet"
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime > timedelta(days=max_age_days)

    def get_daily_date_range(self, symbol: str) -> tuple:
        """获取缓存中日线数据的日期范围。"""
        df = self.get_daily(symbol)
        if df is not None and not df.empty:
            dates = pd.to_datetime(df["日期"])
            return dates.min(), dates.max()
        return None, None

    # ---- 基本面数据 ----

    def get_fundamentals(self, date_str: str) -> pd.DataFrame | None:
        """读取某日的基本面快照。"""
        path = self.fundamentals_dir / f"{date_str}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def put_fundamentals(self, date_str: str, df: pd.DataFrame):
        """写入基本面快照。"""
        path = self.fundamentals_dir / f"{date_str}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        logger.debug(f"基本面缓存写入: {date_str} ({len(df)} 条)")

    def is_fundamentals_stale(self, date_str: str, max_age_days: int = 90) -> bool:
        """判断基本面缓存是否过期（默认90天，约一个季度）。"""
        path = self.fundamentals_dir / f"{date_str}.parquet"
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.now() - mtime > timedelta(days=max_age_days)

    # ---- 缓存管理 ----

    def list_cached_symbols(self) -> list[str]:
        """列出所有已缓存的股票代码。"""
        return [p.stem for p in self.daily_dir.glob("*.parquet")]

    def invalidate_daily(self, symbol: str):
        """删除某只股票的缓存。"""
        path = self.daily_dir / f"{symbol}.parquet"
        if path.exists():
            path.unlink()
            logger.info(f"已删除缓存: {symbol}")

    def clear_all(self):
        """清空所有缓存。"""
        import shutil
        shutil.rmtree(self.daily_dir, ignore_errors=True)
        shutil.rmtree(self.fundamentals_dir, ignore_errors=True)
        self._ensure_dirs()
        logger.info("已清空所有缓存")
