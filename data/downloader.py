"""
AKShare 数据下载 — A股日K线、指数成分股、基本面数据。
"""

import time
import random
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from tqdm import tqdm

try:
    import akshare as ak
except ImportError:
    logger.error("请先安装 akshare: pip install akshare")
    raise

from data.cache import CacheManager


class DataDownloader:
    """A股数据下载器，封装 AKShare API，带本地缓存。"""

    # 指数代码映射
    INDEX_MAP = {
        "000300": "沪深300",
        "000905": "中证500",
        "000016": "上证50",
        "399006": "创业板指",
        "000688": "科创50",
        "000852": "中证1000",
        "932000": "中证2000",
    }

    def __init__(self, cache: CacheManager | None = None):
        self.cache = cache or CacheManager()
        self._rate_limit = 0.5  # API 调用间隔（秒），腾讯接口较宽容
        self._max_retries = 3   # 失败重试次数

    # ==================== 指数成分股 ====================

    def download_index_constituents(self, index_code: str) -> pd.DataFrame:
        """下载指数成分股。

        Args:
            index_code: 指数代码，如 "000300" (沪深300)

        Returns:
            DataFrame with columns: 指数代码, 指数名称, 成分券代码, 成分券名称
        """
        try:
            # 中证指数 (000xxx, 399xxx, 0006xx)
            df = ak.index_stock_cons_csindex(symbol=index_code)
            df = df.rename(columns={
                "成分券代码": "symbol",
                "成分券名称": "name",
            })
            logger.info(f"{self.INDEX_MAP.get(index_code, index_code)} "
                        f"成分股: {len(df)} 只")
            return df
        except Exception as e:
            logger.error(f"下载 {index_code} 成分股失败: {e}")
            raise

    def get_universe_symbols(self, index_codes: list[str]) -> list[str]:
        """获取多个指数的合并成分股列表。

        Args:
            index_codes: 指数代码列表

        Returns:
            去重后的股票代码列表
        """
        all_symbols = []
        for code in index_codes:
            try:
                df = self.download_index_constituents(code)
                symbols = df["symbol"].astype(str).str.zfill(6).tolist()
                all_symbols.extend(symbols)
            except Exception as e:
                logger.warning(f"获取 {code} 成分股失败，跳过: {e}")
                continue

        symbols = sorted(set(all_symbols))
        logger.info(f"股票池合计: {len(symbols)} 只（去重后）")
        return symbols

    # ==================== 日K线数据 ====================

    def download_daily_ohlcv(self, symbol: str, start: str, end: str,
                             adjust: str = "qfq") -> pd.DataFrame | None:
        """下载单只股票日K线。

        Args:
            symbol: 6位股票代码 (如 "000001")
            start: 起始日期 "YYYYMMDD"
            end: 结束日期 "YYYYMMDD"
            adjust: 复权方式 "qfq"(前复权) / "hfq"(后复权) / ""(不复权)

        Returns:
            DataFrame with columns:
            日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        """
        # 先检查缓存
        cached = self.cache.get_daily(symbol)
        fetch_start = start  # 实际向接口请求的起点（增量时从缓存末日起）
        if cached is not None:
            cached_dates = pd.to_datetime(cached["日期"])
            cached_start = cached_dates.min().strftime("%Y%m%d")
            cached_end = cached_dates.max().strftime("%Y%m%d")

            # 缓存完全覆盖请求区间
            if cached_start <= start and cached_end >= end:
                mask = (cached_dates >= pd.Timestamp(start)) & \
                       (cached_dates <= pd.Timestamp(end))
                return cached[mask].reset_index(drop=True)

            # 增量更新：缓存起点足够早时，只补缺失的尾部区间
            # （从缓存末日起重下，重叠部分按日期去重）
            if cached_start <= start:
                fetch_start = cached_end
            # 否则缓存起点晚于请求起点，全量重下

        try:
            # 随机抖动避免被识别为爬虫
            jitter = random.uniform(0.3, 0.8)
            time.sleep(self._rate_limit + jitter)

            # 腾讯接口需要 sz000001 / sh600000 格式
            if symbol.startswith(("60", "68")):
                tx_symbol = f"sh{symbol}"
            else:
                tx_symbol = f"sz{symbol}"

            # 带重试的下载 — 优先用腾讯接口
            df = None
            last_error = None
            for attempt in range(self._max_retries):
                try:
                    # 传入复权参数，否则腾讯接口默认不复权，
                    # 分红送转会造成价格跳空、动量因子失真
                    try:
                        df = ak.stock_zh_a_hist_tx(
                            symbol=tx_symbol,
                            start_date=f"{fetch_start[:4]}-{fetch_start[4:6]}-{fetch_start[6:]}",
                            end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}",
                            adjust=adjust,
                        )
                    except TypeError:
                        # 旧版 akshare 无 adjust 参数
                        df = ak.stock_zh_a_hist_tx(
                            symbol=tx_symbol,
                            start_date=f"{fetch_start[:4]}-{fetch_start[4:6]}-{fetch_start[6:]}",
                            end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}",
                        )
                    break  # 成功则跳出重试循环
                except Exception as e:
                    last_error = e
                    if attempt < self._max_retries - 1:
                        wait = (2 ** attempt) * 2
                        logger.debug(f"{symbol}: 第{attempt+1}次失败, {wait}s后重试...")
                        time.sleep(wait)
                    else:
                        # 腾讯失败，尝试东方财富
                        try:
                            logger.debug(f"{symbol}: 腾讯接口失败，尝试东方财富...")
                            df = ak.stock_zh_a_hist(
                                symbol=symbol,
                                period="daily",
                                start_date=fetch_start,
                                end_date=end,
                                adjust=adjust,
                            )
                            break
                        except Exception:
                            raise last_error

            if df is None or df.empty:
                logger.debug(f"{symbol}: 无数据")
                return None

            # 统一列名（兼容东方财富和腾讯两种来源）
            if "open" in df.columns:
                # 腾讯接口格式: date, open, close, high, low, volume
                df = df.rename(columns={
                    "date": "日期", "open": "开盘", "close": "收盘",
                    "high": "最高", "low": "最低", "volume": "成交量",
                    "amount": "成交额", "turnover": "换手率",
                })
                # 腾讯接口不返回涨跌幅，从价格计算
                df["涨跌幅"] = df["收盘"].pct_change() * 100
                df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
                df["涨跌额"] = df["收盘"].diff()
            else:
                # 东方财富格式
                df = df.rename(columns={
                    "日期": "日期", "开盘": "开盘", "收盘": "收盘",
                    "最高": "最高", "最低": "最低", "成交量": "成交量",
                    "成交额": "成交额", "振幅": "振幅",
                    "涨跌幅": "涨跌幅", "涨跌额": "涨跌额", "换手率": "换手率",
                })

            # 数值类型转换
            for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅", "换手率"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["日期"] = pd.to_datetime(df["日期"])

            # 增量更新：与已有缓存合并（保留完整历史，重叠按日期去重取新值）
            if cached is not None:
                merged = pd.concat([cached, df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["日期"], keep="last")
                merged = merged.sort_values("日期").reset_index(drop=True)
                df = merged

            # 缓存
            self.cache.update_daily(symbol, df)
            return df

        except Exception as e:
            logger.warning(f"下载 {symbol} 日线失败: {e}")
            return None

    def download_batch_daily(self, symbols: list[str], start: str, end: str,
                             adjust: str = "qfq") -> dict:
        """批量下载日K线。

        Returns:
            {symbol: DataFrame} 字典
        """
        results = {}
        failed = []

        logger.info(f"开始下载 {len(symbols)} 只股票日线 "
                    f"({start[:4]}-{start[4:6]}-{start[6:]} ~ "
                    f"{end[:4]}-{end[4:6]}-{end[6:]})")

        for sym in tqdm(symbols, desc="下载日线"):
            df = self.download_daily_ohlcv(sym, start, end, adjust)
            if df is not None and not df.empty:
                results[sym] = df
            else:
                failed.append(sym)

        logger.info(f"下载完成: {len(results)} 成功, {len(failed)} 失败")
        if failed:
            logger.warning(f"失败的股票: {failed[:10]}{'...' if len(failed)>10 else ''}")

        return results

    # ==================== 基本面数据 ====================

    def download_fundamentals_snapshot(self) -> pd.DataFrame | None:
        """下载全市场基本面快照（PE、PB、市值等）。

        这个接口一次返回所有A股的实时数据，非常快。

        Returns:
            DataFrame with columns: 代码, 名称, 市盈率, 市净率, 总市值, 流通市值, etc.
        """
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.rename(columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "prev_close",
                "量比": "volume_ratio",
                "换手率": "turnover_rate",
                "市盈率-动态": "pe_dynamic",
                "市净率": "pb",
                "总市值": "total_market_cap",
                "流通市值": "float_market_cap",
                "60日涨跌幅": "ret_60d",
            })
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            logger.info(f"基本面快照: {len(df)} 只股票")
            return df
        except Exception as e:
            logger.error(f"下载基本面快照失败: {e}")
            return None

    def download_fundamentals_for_symbols(self, symbols: list[str]) -> pd.DataFrame | None:
        """获取指定股票池的基本面数据。"""
        df_all = self.download_fundamentals_snapshot()
        if df_all is not None:
            filtered = df_all[df_all["symbol"].isin(symbols)].copy()
            logger.info(f"基本面数据: {len(filtered)} / {len(symbols)} 只")
            return filtered
        return None

    # ==================== 交易日历 ====================

    def download_trading_calendar(self, start: str, end: str) -> list[str]:
        """获取A股交易日列表。

        Returns:
            交易日字符串列表 ["YYYY-MM-DD", ...]
        """
        try:
            df = ak.tool_trade_date_hist_sina()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            mask = (df["trade_date"] >= start) & (df["trade_date"] <= end)
            dates = df[mask]["trade_date"].dt.strftime("%Y-%m-%d").tolist()
            logger.info(f"交易日: {len(dates)} 天 ({start} ~ {end})")
            return dates
        except Exception as e:
            logger.warning(f"下载交易日历失败: {e}，使用工作日推算")
            # 降级：使用工作日
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
            dates = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            return dates
