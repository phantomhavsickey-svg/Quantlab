"""
AKShare 数据下载 — A股日K线、指数成分股、基本面数据。

慢在哪里（旧实现）:
    每只股票先 sleep(0.5 + 0.3~0.8s) 再**串行**调 ak.stock_zh_a_hist_tx，
    而该接口按日历年每年发一次 HTTP 请求（7 年历史 = 7~8 次），且每次
    requests.get 都新建 TLS 连接。1000 只全量回补 ≈ 75 分钟。

现在的口径:
    - 直连腾讯 kline 接口：单次最多 640 根（实测 >640 也按 640 截断），
      接口忽略起始日、返回截止日之前的最后 640 根，故按 end 向前翻页，
      7 年历史只需 3 次请求；每个工作线程复用一个 Session（keep-alive）。
    - ThreadPoolExecutor 并发 + 全局令牌桶限速（默认 8 线程 / 20 请求每秒），
      连续失败惩罚性降速、成功一半速率恢复，避免把对端打出限流。
    - 缓存已完全覆盖请求区间的股票零请求直接返回。
    - 单只股票三级降级：直连腾讯 → akshare 腾讯 → 东方财富，
      腾讯改版不会让批量任务整体失败。
    列名/单位换算（volume 手→股、turnover /100、amount ×10000）与 akshare
    逐值一致，tests/test_downloader.py 用桩响应钉住翻页与换算。
"""

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

try:
    import akshare as ak
except ImportError:
    logger.error("请先安装 akshare: pip install akshare")
    raise

from data.cache import CacheManager

TX_KLINE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
TX_PAGE_ROWS = 640        # 接口单次返回上限
TX_MAX_PAGES = 40         # 翻页兜底(40×640≈70 年,足够任何 A 股全历史)
DEFAULT_WORKERS = 8
DEFAULT_RATE = 20.0       # 全局请求速率上限(个/秒)
DEFAULT_TIMEOUT = 20.0

# 腾讯接口"已经以股为单位"给出的前缀:科创板/创业板以外的指数代码。
# akshare 的排除表里还有 sz000，而 sz000 同时命中平安银行、万科这类**深市主板
# 000 号段股票**，它们的成交量其实和 001/002 一样是"手"。实测
# 成交量×价格 vs 成交额:000001/000002 差 100 倍，其余号段一致 → 这里剔除
# sz000。修正只影响成交量绝对值，而全项目对成交量的用法(量比=自身均量之比、
# 停牌判定=是否为 0)都是尺度无关的，因此不改任何已发布数字。
TX_VOLUME_IN_SHARES = ("sh688", "sh689", "sh000", "sz399")


def volume_in_shares(sym: str) -> bool:
    """该前缀下腾讯给的成交量已是"股"，无需 ×100。"""
    return sym.startswith(TX_VOLUME_IN_SHARES)


def tx_symbol(symbol: str) -> str:
    """补市场前缀，规则与 akshare 的 _normalize_tx_symbol 一致。

    旧实现只按 ("60","68") 判沪市，北交所(43/83/87/92)会被错标成 sz 而
    永远取不到数据；这里补齐，指数/科创板的前缀也影响成交量单位换算。
    """
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if s.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{s}"
    if s.startswith(("430", "830", "831", "832", "833", "834", "835",
                     "836", "837", "838", "839", "870", "871", "872",
                     "873", "874", "875", "876", "877", "878", "879", "920")):
        return f"bj{s}"
    return f"sz{s}"


def index_symbol(code: str) -> str:
    """指数的腾讯代码：399 号段在深市，其余（000/880/999）在上海。

    不能复用 tx_symbol()——它按个股规则把 000852（中证1000）判给深市，
    而指数 000 号段全部挂在 sh 下。
    """
    s = str(code).strip().lower()
    if s.startswith(("sh", "sz")):
        return s
    return f"sz{s}" if s.startswith("399") else f"sh{s}"


def tx_rows_to_frame(rows: list, sym: str, start: str, end: str) -> pd.DataFrame:
    """原始 kline 行 → akshare 同形状帧(date/open/close/high/low/volume/turnover/amount)。

    取列位置 [0,1,2,3,4,5,7,8] 与单位换算分支照抄 akshare，位置 6 是接口
    多给的辅助字段(不计入行情)，两边必须一致否则前复权量纲会错位。
    date 列同为 datetime.date(akshare 在 .dt.date 之后没有再转回 Timestamp)。
    """
    cols = ["date", "open", "close", "high", "low", "volume", "turnover", "amount"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 7, 8]].copy()
    df.columns = cols
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if not volume_in_shares(sym):
        df["volume"] = df["volume"] * 100      # 手 → 股
    df["turnover"] = df["turnover"] / 100      # % → 小数
    df["amount"] = df["amount"] * 10000        # 万元 → 元
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.drop_duplicates(ignore_index=True)
    df.index = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_index()[start:end].reset_index(drop=True)
    return df


class TokenBucket:
    """线程安全令牌桶：对同一数据源的整体速率做硬限制。

    penalize() 把速率减半(对端在限流时唯一正确的反应是慢下来)，
    recover() 每次 ×1.5 回到设定速率，不会冲回去。
    """

    def __init__(self, rate: float = DEFAULT_RATE, burst: float | None = None):
        self._full_rate = max(0.1, float(rate))
        self._rate = self._full_rate
        self._cap = max(1.0, float(burst if burst is not None else self._full_rate))
        self._tokens = self._cap
        self._last = time.monotonic()
        self._cond = threading.Condition()

    @property
    def rate(self) -> float:
        return self._rate

    def acquire(self, n: float = 1.0) -> None:
        with self._cond:
            while True:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                self._cond.wait((n - self._tokens) / self._rate + 0.001)

    def penalize(self) -> None:
        with self._cond:
            self._refill()
            self._rate = max(0.5, self._rate / 2)
            self._tokens = min(self._tokens, 1.0)
            self._cond.notify_all()

    def recover(self) -> None:
        with self._cond:
            self._refill()
            if self._rate < self._full_rate:
                self._rate = min(self._full_rate, self._rate * 1.5)
                self._cond.notify_all()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._cap, self._tokens + (now - self._last) * self._rate)
        self._last = now


class _ThreadSessions:
    """每线程一个 requests.Session —— Session 本身不是线程安全的。"""

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def __init__(self):
        self._local = threading.local()

    def get(self) -> requests.Session:
        sess = getattr(self._local, "sess", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update(self.HEADERS)
            self._local.sess = sess
        return sess


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

    def __init__(self, cache: CacheManager | None = None,
                 max_workers: int = DEFAULT_WORKERS,
                 requests_per_second: float = DEFAULT_RATE,
                 timeout: float = DEFAULT_TIMEOUT,
                 max_retries: int = 3):
        self.cache = cache or CacheManager()
        self._max_workers = max(1, int(max_workers))
        self._bucket = TokenBucket(requests_per_second)
        self._timeout = float(timeout)
        self._max_retries = max(1, int(max_retries))
        self._sessions = _ThreadSessions()

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

    def download_index_daily(self, symbol: str, start: str,
                             end: str) -> pd.DataFrame:
        """指数日线（回测基准用），走与个股同一条腾讯直连通道。

        akshare 的指数接口是东财域名，本机 IP 常被限流（RemoteDisconnected），
        而腾讯这条通道一次给 640 根、无复权口径。返回中文列名，与日线缓存同形状。
        """
        code = index_symbol(symbol)
        df = self._fetch_tx_direct(code, start, end, "")
        if df.empty:
            raise RuntimeError(f"腾讯指数接口对 {code} 返回空数据")
        out = df.rename(columns={
            "date": "日期", "open": "开盘", "close": "收盘", "high": "最高",
            "low": "最低", "volume": "成交量", "amount": "成交额"})
        out["日期"] = pd.to_datetime(out["日期"])
        return out

    def download_daily_ohlcv(self, symbol: str, start: str, end: str,
                             adjust: str = "qfq") -> pd.DataFrame | None:
        """下载单只股票日K线（命中缓存则不请求）。

        Args:
            symbol: 6位股票代码 (如 "000001")
            start: 起始日期 "YYYYMMDD"
            end: 结束日期 "YYYYMMDD"
            adjust: 复权方式 "qfq"(前复权) / "hfq"(后复权) / ""(不复权)

        Returns:
            DataFrame with columns:
            日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        """
        cached, fetch_start = self._cache_slice(symbol, start, end)
        if fetch_start is None:
            return cached                      # 缓存完全覆盖，零请求

        try:
            df = self._fetch(symbol, fetch_start, end, adjust)
        except Exception as e:
            logger.warning(f"下载 {symbol} 日线失败: {e}")
            return None
        if df is None or df.empty:
            logger.debug(f"{symbol}: 无数据")
            return None
        return self._normalize_and_store(symbol, df, cached)

    def _cache_slice(self, symbol: str, start: str, end: str):
        """返回 (缓存帧, 实际请求起点)。

        (cached, None) → 缓存已覆盖 [start, end]，只需切片，不发请求；
        (cached|None, "YYYYMMDD") → 需要下载；cached 为 None 时无需合并。
        """
        cached = self.cache.get_daily(symbol)
        if cached is None:
            return None, start
        cached_dates = pd.to_datetime(cached["日期"])
        cached_start = cached_dates.min().strftime("%Y%m%d")
        cached_end = cached_dates.max().strftime("%Y%m%d")

        if cached_start <= start and cached_end >= end:
            mask = (cached_dates >= pd.Timestamp(start)) & \
                   (cached_dates <= pd.Timestamp(end))
            return cached[mask].reset_index(drop=True), None

        # 增量更新：缓存起点足够早时只补尾部（从缓存末日起重下，重叠按日期去重）；
        # 否则缓存起点晚于请求起点，全量重下。
        return cached, (cached_end if cached_start <= start else start)

    def _fetch(self, symbol: str, start: str, end: str,
               adjust: str) -> pd.DataFrame:
        """单只股票的行情拉取：直连腾讯 → akshare 腾讯 → 东方财富。"""
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                df = self._fetch_tx_direct(symbol, start, end, adjust)
                self._bucket.recover()
                return df
            except Exception as e:
                last_error = e
                self._bucket.penalize()
                if attempt < self._max_retries - 1:
                    time.sleep(min(6.0, (2 ** attempt) * 0.4 + random.random() * 0.3))
                else:
                    logger.debug(f"{symbol}: 直连腾讯失败({e})，退回 akshare")

        try:
            df = self._fetch_tx_akshare(symbol, start, end, adjust)
            self._bucket.recover()
            return df
        except Exception as e:
            logger.debug(f"{symbol}: akshare 腾讯接口也失败({e})，改用东方财富"
                         f"[直连首次错误: {last_error}]")

        # _fetch_em 内部照样先过令牌桶，降级路径不绕过限速
        return self._fetch_em(symbol, start, end, adjust)

    def _fetch_tx_direct(self, symbol: str, start: str, end: str,
                         adjust: str) -> pd.DataFrame:
        """直连腾讯 newfqkline：按 end 向前翻页，每页最多 640 根。"""
        sym = tx_symbol(symbol)
        sess = self._sessions.get()
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        rows: list = []
        seen: set = set()
        w_end = end_ts
        for _ in range(TX_MAX_PAGES):
            self._bucket.acquire()
            r = sess.get(TX_KLINE_URL, params={
                "param": f"{sym},day,,{w_end:%Y-%m-%d},{TX_PAGE_ROWS},{adjust}",
                "r": "0.8205512681390605"}, timeout=self._timeout)
            page = _tx_page_rows(r.text, sym)
            if not page:
                break
            fresh = [x for x in page if x[0] not in seen]
            seen.update(x[0] for x in fresh)
            rows.extend(fresh)
            first = pd.Timestamp(str(page[0][0]))
            if len(page) < TX_PAGE_ROWS or first <= start_ts:
                break                          # 已到上市首日 / 已覆盖请求起点
            w_end = first - pd.Timedelta(days=1)
        return tx_rows_to_frame(rows, sym, start_ts.strftime("%Y-%m-%d"),
                                end_ts.strftime("%Y-%m-%d"))

    def _fetch_tx_akshare(self, symbol: str, start: str, end: str,
                          adjust: str) -> pd.DataFrame:
        """akshare 的腾讯接口（旧数据源路径，带旧版本签名兼容）。

        akshare 把 sz000 当指数、漏乘 100，直连路径已修正；降级时要把同一批
        行补回来，否则一次降级就会在同一只股票的 history 里混进两种单位。
        """
        args = dict(symbol=tx_symbol(symbol),
                    start_date=f"{start[:4]}-{start[4:6]}-{start[6:]}",
                    end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}")
        try:
            # 必须传复权参数，否则腾讯默认不复权，分红送转造成价格跳空、
            # 动量因子失真
            df = ak.stock_zh_a_hist_tx(adjust=adjust, **args)
        except TypeError:
            df = ak.stock_zh_a_hist_tx(**args)      # 旧版 akshare 无 adjust
        sym = tx_symbol(symbol)
        if "volume" in df.columns and sym.startswith("sz000"):
            df = df.copy()
            df["volume"] = df["volume"] * 100
        return df

    def _fetch_em(self, symbol: str, start: str, end: str,
                  adjust: str) -> pd.DataFrame:
        """东方财富兜底（列名已是中文）。

        实测东财 成交量 单位是"手"（与腾讯/100 精确对上，成交额则完全一致），
        不换算会让降级行的成交量小 100 倍，同一文件里混两种单位。
        """
        self._bucket.acquire()
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end, adjust=adjust)
        if df is not None and "成交量" in df.columns:
            df = df.copy()
            df["成交量"] = df["成交量"] * 100
        return df

    def _normalize_and_store(self, symbol: str, df: pd.DataFrame,
                             cached: pd.DataFrame | None) -> pd.DataFrame:
        """统一列名/类型，与缓存合并去重后落盘（返回完整历史）。"""
        if "open" in df.columns:
            # 腾讯接口格式: date, open, close, high, low, volume, turnover, amount
            df = df.rename(columns={
                "date": "日期", "open": "开盘", "close": "收盘",
                "high": "最高", "low": "最低", "volume": "成交量",
                "amount": "成交额", "turnover": "换手率",
            })
            # 腾讯接口不返回涨跌幅，从价格计算
            df["涨跌幅"] = df["收盘"].pct_change() * 100
            df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
            df["涨跌额"] = df["收盘"].diff()
        # 否则已是东方财富格式，列名无需改动

        for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额",
                    "涨跌幅", "换手率"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["日期"] = pd.to_datetime(df["日期"])

        if cached is not None:
            merged = pd.concat([cached, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["日期"], keep="last")
            merged = merged.sort_values("日期").reset_index(drop=True)
            df = merged

        self.cache.update_daily(symbol, df)
        return df

    def download_batch_daily(self, symbols: list[str], start: str, end: str,
                             adjust: str = "qfq",
                             max_workers: int | None = None) -> dict:
        """并发批量下载日K线。

        Args:
            max_workers: 本次并发数，None = 用构造时的设置。网络型任务，
                线程数只受令牌桶速率约束，加大线程数不会突破限速。

        Returns:
            {symbol: DataFrame} 字典
        """
        workers = max(1, int(max_workers or self._max_workers))
        workers = min(workers, len(symbols)) if symbols else 1
        t0 = time.monotonic()
        logger.info(f"开始下载 {len(symbols)} 只股票日线 "
                    f"({start[:4]}-{start[4:6]}-{start[6:]} ~ "
                    f"{end[:4]}-{end[4:6]}-{end[6:]}) | "
                    f"并发 {workers} 线程, 速率上限 {self._bucket.rate:.0f} 请求/秒")

        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.download_daily_ohlcv, sym, start, end,
                                 adjust): sym for sym in symbols}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="下载日线", ncols=80):
                sym = futures[fut]
                try:
                    df = fut.result()
                except Exception as e:
                    logger.warning(f"下载 {sym} 异常: {e}")
                    failed.append(sym)
                    continue
                if df is not None and not df.empty:
                    results[sym] = df
                else:
                    failed.append(sym)

        elapsed = max(1e-9, time.monotonic() - t0)
        logger.info(f"下载完成: {len(results)} 成功, {len(failed)} 失败 | "
                    f"耗时 {elapsed/60:.1f} 分钟 "
                    f"({len(symbols)/elapsed:.1f} 只/秒)")
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


def _tx_page_rows(text: str, sym: str) -> list:
    """解析 newfqkline 响应。

    接口带 `_var` 时返回 `var=({...});` 形式的 JS；这里不发 `_var`，
    正常是纯 JSON，但仍用 raw_decode 剥掉可能存在的前缀与尾随分号
    （json.loads 会因为尾随字符直接失败，demjson 太慢且已废弃）。
    行情数组的键随复权方式变化 (day / hfqday / qfqday)，判定顺序照抄
    akshare 以保证两边同形。
    """
    i = text.find("={")
    payload = text[i + 1:] if i >= 0 else text
    node = json.JSONDecoder().raw_decode(payload.strip())[0]["data"][sym]
    if not isinstance(node, dict):
        raise ValueError(f"{sym}: 接口返回异常载荷")
    for key in ("day", "hfqday", "qfqday"):
        if key in node and node[key]:
            return node[key]
    return []
