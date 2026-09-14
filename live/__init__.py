"""
Live 实时行情模块

数据源: 新浪财经 HTTP API
格式: http://hq.sinajs.cn/list=sh600000,sz000001

字段解析 (A股):
  0:  股票名称
  1:  今开盘
  2:  昨收盘
  3:  当前价
  4:  最高价
  5:  最低价
  6:  买一价
  7:  卖一价
  8:  成交量(股)
  9:  成交额(元)
  10-14: 买一~买五量
  15-19: 卖一~卖五量
  30: 日期
  31: 时间
"""

import time
import re
import requests
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Quote:
    """实时行情快照。"""
    symbol: str
    name: str
    price: float        # 当前价
    open: float         # 今开
    high: float         # 最高
    low: float          # 最低
    prev_close: float   # 昨收
    volume: float       # 成交量(股)
    amount: float       # 成交额(元)
    bid: float          # 买一价
    ask: float          # 卖一价
    change_pct: float   # 涨跌幅 %
    time: str           # 时间 HH:MM:SS
    date: str           # 日期 YYYY-MM-DD

    def __repr__(self):
        sign = "+" if self.change_pct >= 0 else ""
        return (f"<{self.name}({self.symbol}) {self.price:.2f} "
                f"({sign}{self.change_pct:.2f}%) "
                f"V={self.volume/10000:.0f}万手>")


class SinaQuoteFeed:
    """新浪财经实时行情源。

    用法:
        feed = SinaQuoteFeed()
        quotes = feed.fetch(["000001", "600519", "000300"])

        # 持续监控
        while True:
            quotes = feed.fetch(symbols)
            for q in quotes.values():
                print(q)
            time.sleep(3)
    """

    BASE_URL = "http://hq.sinajs.cn/list="
    HEADERS = {"Referer": "https://finance.sina.com.cn"}

    # 新浪代码前缀
    @staticmethod
    def to_sina_code(symbol: str) -> str:
        """将 6位代码 转为新浪格式: 000001 -> sz000001, 600519 -> sh600519。"""
        symbol = str(symbol).zfill(6)
        if symbol.startswith(("60", "68")):
            return f"sh{symbol}"
        else:
            return f"sz{symbol}"

    @staticmethod
    def from_sina_code(code: str) -> str:
        """从新浪格式转回6位代码。"""
        return code[2:]

    def fetch(self, symbols: list[str],
              include_index: bool = False) -> dict[str, Quote]:
        """获取实时行情。

        Args:
            symbols: 6位股票代码列表
            include_index: 是否包含上证指数

        Returns:
            {symbol: Quote} 字典
        """
        # 构建请求代码
        codes = [self.to_sina_code(s) for s in symbols]
        if include_index:
            codes.append("s_sh000001")  # 上证指数

        # 分批请求（每次最多 50 个）
        results = {}
        batch_size = 50
        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            url = self.BASE_URL + ",".join(batch)
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                resp.encoding = "gbk"
                batch_results = self._parse_response(resp.text)
                results.update(batch_results)
            except Exception as e:
                logger.warning(f"行情请求失败 (batch {i}): {e}")
            time.sleep(0.1)  # 批次间短暂间隔

        return results

    def fetch_single(self, symbol: str) -> Optional[Quote]:
        """获取单只股票行情。"""
        results = self.fetch([symbol])
        return results.get(symbol)

    # ==================== 解析 ====================

    def _parse_response(self, text: str) -> dict[str, Quote]:
        """解析新浪返回的 var hq_str_xxx="..." 格式。"""
        results = {}
        # 正则提取: var hq_str_XXXX="内容";
        pattern = r'var hq_str_(\w+)="([^"]*)"'
        matches = re.findall(pattern, text)

        for code, data_str in matches:
            symbol = self.from_sina_code(code)
            try:
                quote = self._parse_quote(symbol, data_str)
                if quote:
                    results[symbol] = quote
            except Exception as e:
                logger.debug(f"解析 {symbol} 失败: {e}")

        return results

    def _parse_quote(self, symbol: str, data_str: str) -> Optional[Quote]:
        """解析单只股票的行情数据。"""
        fields = data_str.split(",")
        if len(fields) < 32:
            return None

        try:
            name = fields[0]
            price = self._float(fields[3])
            if price <= 0:
                return None

            prev_close = self._float(fields[2])
            change_pct = 0.0
            if prev_close > 0:
                change_pct = (price - prev_close) / prev_close * 100

            return Quote(
                symbol=symbol,
                name=name,
                price=price,
                open=self._float(fields[1]),
                high=self._float(fields[4]),
                low=self._float(fields[5]),
                prev_close=prev_close,
                volume=self._float(fields[8]),
                amount=self._float(fields[9]),
                bid=self._float(fields[6]),
                ask=self._float(fields[7]),
                change_pct=round(change_pct, 2),
                date=fields[30],
                time=fields[31] if len(fields) > 31 else "",
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"解析 {symbol} 字段错误: {e}")
            return None

    @staticmethod
    def _float(val: str) -> float:
        """安全转 float。"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0


# ==================== 行情监控器 ====================

class QuoteMonitor:
    """行情监控器 — 定时拉取行情，触发回调。

    用法:
        def on_quotes(quotes):
            for sym, q in quotes.items():
                if q.change_pct > 5:
                    print(f"异动: {q}")

        monitor = QuoteMonitor(feed, symbols)
        monitor.on_update = on_quotes
        monitor.start(interval=3)  # 每3秒拉一次
    """

    def __init__(self, feed: SinaQuoteFeed, symbols: list[str],
                 interval: float = 3.0):
        self.feed = feed
        self.symbols = symbols
        self.interval = interval
        self.latest: dict[str, Quote] = {}
        self.running = False
        self.on_update = None  # callback(quotes: dict)

    def start(self, max_iterations: int | None = None):
        """开始监控循环。

        Args:
            max_iterations: 最大迭代次数 (None = 无限)
        """
        self.running = True
        iteration = 0
        logger.info(f"行情监控启动: {len(self.symbols)} 只股票, "
                     f"间隔 {self.interval}s")

        while self.running:
            try:
                quotes = self.feed.fetch(self.symbols)
                self.latest = quotes

                if self.on_update:
                    self.on_update(quotes)

                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break

                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("行情监控已停止")
                break
            except Exception as e:
                logger.error(f"行情拉取异常: {e}")
                time.sleep(self.interval)

    def stop(self):
        """停止监控。"""
        self.running = False

    def get(self, symbol: str) -> Optional[Quote]:
        """获取最新行情。"""
        return self.latest.get(symbol)


# ==================== 快捷测试 ====================

if __name__ == "__main__":
    # 测试
    feed = SinaQuoteFeed()

    # 拉取一次
    test_symbols = ["000001", "000002", "600519", "600036"]
    quotes = feed.fetch(test_symbols)
    for sym, q in quotes.items():
        print(q)

    # 持续监控5次
    print("\n持续监控 (3秒间隔 × 5次)...")
    monitor = QuoteMonitor(feed, test_symbols, interval=3)
    monitor.on_update = lambda qs: print(
        f"  [{datetime.now().strftime('%H:%M:%S')}] "
        f"{' | '.join(f'{q.name} {q.price:.2f}({q.change_pct:+.2f}%)' for q in qs.values())}"
    )
    monitor.start(max_iterations=5)
    print("测试完成")
