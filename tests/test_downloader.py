"""数据下载器的口径测试（全部离线，打网络的地方用桩）。

钉住四件事,每一件都对应"改快"时最容易悄悄改掉的东西:
    1. 腾讯 newfqkline 的翻页语义 —— 接口忽略起始日、只返回 end 之前的最后
       640 根,所以必须从 end 往回翻,翻页之间不能重叠也不能丢行;
    2. 单位换算与列位置选取必须和 akshare 逐值一致(volume 手→股、
       turnover /100、amount ×10000,取列 [0,1,2,3,4,5,7,8]);
    3. 缓存覆盖判定:全命中零请求、尾部增量、缓存起点偏晚则全量重下;
    4. 三级降级(直连→akshare 腾讯→东财)与并发批量下载的返回值。
真实接口的逐值等价性另外用一次性脚本对 7 只股票(含科创板/001 号段/长
历史)比对过 akshare,各字段 maxdiff 全为 0。
"""

import json
import threading
import time
import types

import pandas as pd
import pytest

import data.downloader as dl
from data.cache import CacheManager
from data.downloader import (DataDownloader, TokenBucket, _tx_page_rows,
                             tx_rows_to_frame, tx_symbol, volume_in_shares)


def bar(day, close=10.0, vol=1000.0, turnover=1.5, amount=2.0):
    """一根 kline:字段顺序照接口实测,下标 6 是不进结果的辅助列。"""
    return [day, str(close - 0.1), str(close), str(close + 0.1),
            str(close - 0.2), str(vol), "0.00", str(turnover), str(amount),
            "{}"]


def bars(days):
    return [bar(d) for d in days]


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeSession:
    """腾讯 kline 替身:param = {sym},day,,{end},{cap},{adjust}。

    和真接口一样**忽略起始日**,只返回 end 之前(含 end)的最后 cap 根。
    """

    def __init__(self, sym, rows):
        self.sym = sym
        self.rows = rows                       # 升序 [date, ...]
        self.calls = []
        self.params = []

    def get(self, url, params=None, timeout=None):
        fields = params["param"].split(",")
        end, cap = fields[3], int(fields[4])
        self.calls.append((end, cap))
        self.params.append(params["param"])
        page = [r for r in self.rows if str(r[0]) <= end][-cap:]
        # 真接口不带 _var 时是纯 JSON;这里刻意加上 `var=` 前缀和尾随分号,
        # 检验解析用的是 raw_decode 而不是 json.loads
        text = "kline_dayqfq=" + json.dumps(
            {"code": 0, "data": {self.sym: {"qfqday": page}}},
            ensure_ascii=False) + ";"
        return FakeResponse(text)


def make_downloader(tmp_path, rows, sym="sz000001", **kw):
    cache = CacheManager(str(tmp_path / "cache"))
    d = DataDownloader(cache, requests_per_second=100000.0, **kw)
    sess = FakeSession(sym, rows)
    d._sessions = types.SimpleNamespace(get=lambda: sess)
    return d, sess


def cached_frame(days, close=9.0):
    return pd.DataFrame({"日期": pd.to_datetime(days),
                         "收盘": [close] * len(days),
                         "成交量": [1e6] * len(days)})


# ==================== 前缀与载荷解析 ====================

def test_tx_symbol_prefix_matches_akshare_rule():
    assert tx_symbol("600519") == "sh600519"
    assert tx_symbol("605100") == "sh605100"
    assert tx_symbol("688981") == "sh688981"
    assert tx_symbol("000001") == "sz000001"
    assert tx_symbol("301234") == "sz301234"
    # 旧实现只按 ("60","68") 判沪市,北交所会被错标成 sz → 永远取不到数据
    assert tx_symbol("430047") == "bj430047"
    assert tx_symbol("920008") == "bj920008"
    assert tx_symbol("sh000001") == "sh000001"


def test_tx_page_rows_handles_prefix_and_key_variants():
    for key in ("day", "hfqday", "qfqday"):
        text = "kline_dayqfq2024=" + json.dumps(
            {"data": {"sz000001": {key: bars(["2024-01-02"])}}}) + ";"
        assert _tx_page_rows(text, "sz000001")[0][0] == "2024-01-02"
    # 无数据(未上市/停牌) → 空列表而不是异常
    assert _tx_page_rows(
        json.dumps({"data": {"sz000001": {"qfqday": []}}}), "sz000001") == []
    with pytest.raises(ValueError):
        _tx_page_rows(json.dumps({"data": {"sz000001": []}}), "sz000001")


def test_tx_rows_to_frame_units():
    out = tx_rows_to_frame(bars(["2024-01-02"]), "sz000001",
                           "2024-01-01", "2024-12-31")
    assert list(out.columns) == ["date", "open", "close", "high", "low",
                                 "volume", "turnover", "amount"]
    assert out.loc[0, "turnover"] == pytest.approx(0.015)  # % → 小数
    assert out.loc[0, "amount"] == pytest.approx(20000.0)  # 万元 → 元
    assert out.loc[0, "close"] == 10.0
    # 成交量单位:除科创板/指数外一律 手→股。akshare 的排除表把 sz000 当指数,
    # 于是平安银行/万科这类深市主板 000 号段被漏乘,实测
    # 成交量×收盘价 vs 成交额 差 100 倍(其余号段为 1.00),这里修正。
    # 全项目对成交量只有"与自身均量之比"(量比)和"是否为 0"(停牌)两种用法,
    # 都尺度无关,所以修正不改任何已发布数字。
    assert volume_in_shares("sh688981") and volume_in_shares("sh000001")
    assert volume_in_shares("sz399001")
    assert not volume_in_shares("sz000001")
    assert out.loc[0, "volume"] == 1000.0 * 100             # sz000 号段要乘
    for sym, expect in [("sh688981", 1000.0), ("sh600519", 100000.0),
                        ("sz001979", 100000.0), ("sz300750", 100000.0),
                        ("sh000852", 1000.0)]:
        got = tx_rows_to_frame(bars(["2024-01-02"]), sym,
                               "2024-01-01", "2024-12-31")
        assert got.loc[0, "volume"] == expect, sym


def test_tx_rows_to_frame_slices_to_range():
    rows = bars([f"2024-01-{d:02d}" for d in range(1, 11)])
    out = tx_rows_to_frame(rows, "sz000001", "2024-01-03", "2024-01-06")
    # date 列是 datetime.date(与 akshare 同),不是 Timestamp:下游 _normalize
    # 会 pd.to_datetime 一次,这里不能提前改成 Timestamp 掩盖差异
    assert [str(x) for x in out["date"]] == \
        ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
    assert tx_rows_to_frame([], "sz000001", "2024-01-01", "2024-12-31").empty


# ==================== 翻页 ====================

def test_paging_covers_range_in_one_request_when_small(tmp_path):
    days = [f"2024-{m:02d}-05" for m in range(1, 13)]
    d, sess = make_downloader(tmp_path, bars(days))
    out = d._fetch_tx_direct("000001", "20240101", "20241231", "qfq")
    assert [str(x) for x in out["date"]] == days
    assert sess.calls == [("2024-12-31", 640)]             # 12 根 < 一页,一次够


def test_paging_multi_page_no_dup_no_gap(tmp_path, monkeypatch):
    days = [f"2024-01-{d:02d}" for d in range(1, 21)]
    monkeypatch.setattr(dl, "TX_PAGE_ROWS", 6)
    d, sess = make_downloader(tmp_path, bars(days))
    out = d._fetch_tx_direct("000001", "20240101", "20240120", "qfq")
    assert [str(x) for x in out["date"]] == days    # 顺序/无重/无漏
    assert [c[0] for c in sess.calls] == ["2024-01-20", "2024-01-14",
                                          "2024-01-08", "2024-01-02"]


def test_paging_stops_at_listing_date_even_if_start_is_earlier(tmp_path,
                                                              monkeypatch):
    """请求区间早于上市日:取到首日(不足一页)就停,不能空转到 TX_MAX_PAGES。"""
    days = [f"2024-01-{d:02d}" for d in range(10, 16)]
    monkeypatch.setattr(dl, "TX_PAGE_ROWS", 4)
    d, sess = make_downloader(tmp_path, bars(days))
    out = d._fetch_tx_direct("000001", "20150101", "20240116", "qfq")
    assert len(out) == 6
    assert len(sess.calls) == 2


def test_paging_page_cap_is_not_exceeded(tmp_path, monkeypatch):
    """页大小必须是 640:接口对更大的 cap 一律截到 640,写大会静默丢数据。"""
    assert dl.TX_PAGE_ROWS == 640
    days = [f"2024-01-{d:02d}" for d in range(1, 21)]
    monkeypatch.setattr(dl, "TX_PAGE_ROWS", 3)
    d, sess = make_downloader(tmp_path, bars(days))
    d._fetch_tx_direct("000001", "20240101", "20240120", "qfq")
    assert all(c[1] == 3 for c in sess.calls)


# ==================== 缓存覆盖判定 ====================

def test_full_cover_returns_slice_without_request(tmp_path):
    d, sess = make_downloader(tmp_path, bars(["2024-01-02"]))
    d.cache.put_daily("000001", cached_frame(
        ["2024-01-02", "2024-01-03", "2024-01-04"]))
    got, fetch_start = d._cache_slice("000001", "20240102", "20240104")
    assert fetch_start is None and len(got) == 3
    pd.testing.assert_frame_equal(
        d.download_daily_ohlcv("000001", "20240102", "20240104"), got)
    assert sess.calls == []                                 # 一次请求都没发


def test_tail_update_only_fetches_from_cached_end(tmp_path):
    d, _ = make_downloader(tmp_path, [])
    d.cache.put_daily("000001", cached_frame(
        ["2023-01-01", "2024-01-02", "2024-01-03"]))
    cached, fetch_start = d._cache_slice("000001", "20240101", "20240110")
    assert len(cached) == 3 and fetch_start == "20240103"


def test_cache_starting_later_than_request_forces_full_refetch(tmp_path):
    d, _ = make_downloader(tmp_path, [])
    d.cache.put_daily("000001", cached_frame(["2024-01-02", "2024-01-03"]))
    # 缓存起点晚于请求起点 → 从头重下(不能只补尾巴,否则丢历史)
    assert d._cache_slice("000001", "20230101", "20240110")[1] == "20230101"
    assert d._cache_slice("600000", "20240102", "20240110") == (None, "20240102")


def test_normalize_and_store_merges_keep_last(tmp_path):
    d, _ = make_downloader(tmp_path, [])
    old = cached_frame(["2024-01-02", "2024-01-03"], close=9.0)
    new = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
                        "open": [10.0, 11.0], "close": [10.0, 11.0],
                        "high": [10.5, 11.5], "low": [9.5, 10.5],
                        "volume": [2e6, 3e6], "turnover": [1.0, 2.0],
                        "amount": [2e7, 3e7]})
    out = d._normalize_and_store("000001", new, old)
    assert list(out["日期"].dt.strftime("%Y-%m-%d")) == \
        ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert out.loc[1, "收盘"] == 10.0                       # 新值覆盖旧值
    assert out.loc[2, "涨跌幅"] == pytest.approx(10.0)      # (11/10-1)*100
    assert {"振幅", "涨跌额"} <= set(out.columns)
    assert len(d.cache.get_daily("000001")) == 3            # 已落盘


def test_normalize_and_store_handles_eastmoney_columns(tmp_path):
    """东财兜底帧的列名已是中文,不能被再 rename 一遍。"""
    d, _ = make_downloader(tmp_path, [])
    em = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [9.9], "收盘": [10.0],
                       "最高": [10.2], "最低": [9.8], "成交量": [1e6],
                       "成交额": [1e7], "振幅": [4.0], "涨跌幅": [1.0],
                       "涨跌额": [0.1], "换手率": [1.2]})
    out = d._normalize_and_store("000001", em, None)
    assert out.loc[0, "收盘"] == 10.0 and out.loc[0, "涨跌幅"] == 1.0


# ==================== 降级与并发 ====================

def test_fallback_chain_direct_then_akshare_then_em(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("直连挂了")

    tx_frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]),
                             "open": [9.9], "close": [10.0], "high": [10.1],
                             "low": [9.8], "volume": [1e6], "turnover": [1.0],
                             "amount": [1e7]})
    calls = []
    monkeypatch.setattr(DataDownloader, "_fetch_tx_direct", boom)
    monkeypatch.setattr(dl.ak, "stock_zh_a_hist_tx",
                        lambda **k: calls.append("ak_tx") or tx_frame)
    d = DataDownloader(CacheManager(str(tmp_path / "c")),
                       requests_per_second=100000.0, max_retries=2)
    out = d._fetch("000001", "20240101", "20240110", "qfq")
    assert calls == ["ak_tx"] and len(out) == 1

    def ak_boom(**k):
        raise RuntimeError("akshare 也挂")

    em = pd.DataFrame({"日期": ["2024-01-02"], "收盘": [10.0]})
    monkeypatch.setattr(dl.ak, "stock_zh_a_hist_tx", ak_boom)
    monkeypatch.setattr(dl.ak, "stock_zh_a_hist",
                        lambda **k: calls.append("em") or em)
    assert len(d._fetch("000001", "20240101", "20240110", "qfq")) == 1
    assert calls == ["ak_tx", "em"]


def test_akshare_fallback_repairs_sz000_volume_unit(tmp_path, monkeypatch):
    """降级到 akshare 时,sz000 号段它漏乘的 100 必须补回来。

    不补的话:同一只股票的 history 里,直连拿到的行是"股"、降级那天拿到的行
    是"手",量比因子在交界处出现 100 倍的假跳变。
    """
    tx = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                       "open": [9.9, 10.0], "close": [10.0, 10.1],
                       "high": [10.1, 10.2], "low": [9.8, 9.9],
                       "volume": [1e6, 2e6], "turnover": [1.0, 2.0],
                       "amount": [1e7, 2e7]})
    monkeypatch.setattr(dl.ak, "stock_zh_a_hist_tx", lambda **k: tx)
    d = DataDownloader(CacheManager(str(tmp_path / "c")))
    assert d._fetch_tx_akshare("000001", "20240101", "20240110", "qfq")[
        "volume"].tolist() == [1e8, 2e8]
    # 688/指数它本就没错,不能再乘一遍
    assert d._fetch_tx_akshare("688981", "20240101", "20240110", "qfq")[
        "volume"].tolist() == [1e6, 2e6]
    assert tx["volume"].tolist() == [1e6, 2e6]            # 未就地改调用方的帧


def test_eastmoney_fallback_converts_hands_to_shares(tmp_path, monkeypatch):
    """东财 成交量 单位是手(实测与腾讯差恰好 100 倍,成交额一致)。"""
    em = pd.DataFrame({"日期": ["2024-01-02"], "收盘": [10.0],
                       "成交量": [24891.0], "成交额": [3135849108.0]})
    seen = {}

    def fake_hist(**k):
        seen.update(k)
        return em

    monkeypatch.setattr(dl.ak, "stock_zh_a_hist", fake_hist)
    d = DataDownloader(CacheManager(str(tmp_path / "c")))
    out = d._fetch_em("600519", "20240101", "20240110", "qfq")
    assert out["成交量"].tolist() == [2489100.0]
    assert seen["adjust"] == "qfq"
    assert em["成交量"].tolist() == [24891.0]              # 不改调用方的帧


def test_batch_returns_results_and_collects_failures(tmp_path, monkeypatch):
    d, _ = make_downloader(tmp_path, [], max_workers=4)

    def fake_ohlcv(sym, start, end, adjust):
        return None if sym.endswith("9") else cached_frame(["2024-01-02"])

    monkeypatch.setattr(d, "download_daily_ohlcv", fake_ohlcv)
    got = d.download_batch_daily(["000001", "000002", "000009", "600019"],
                                "20240101", "20240102")
    assert set(got) == {"000001", "000002"}
    assert d.download_batch_daily([], "20240101", "20240102") == {}


def test_batch_downloads_concurrently_and_keeps_every_symbol(tmp_path,
                                                            monkeypatch):
    """并发下载:8 只各写各的 parquet,谁也不能被别人的写入覆盖掉。"""
    days = [f"2024-01-0{i}" for i in range(1, 9)]
    syms = [f"00000{i}" for i in range(1, 9)]
    d, _ = make_downloader(tmp_path, [], max_workers=8)

    def fake_direct(symbol, start, end, adjust):
        return tx_rows_to_frame(bars(days), tx_symbol(symbol),
                                "2024-01-01", "2024-12-31")

    monkeypatch.setattr(d, "_fetch_tx_direct", fake_direct)
    got = d.download_batch_daily(syms, "20240101", "20240110")
    assert set(got) == set(syms)
    assert all(len(d.cache.get_daily(s)) == 8 for s in syms)


def test_batch_actually_overlaps_requests(tmp_path, monkeypatch):
    """串行时总耗时 = 各次之和;若退化回串行,这条测试就会红。"""
    d, _ = make_downloader(tmp_path, [], max_workers=6)
    gate = threading.Barrier(6, timeout=5)

    def slow_direct(symbol, start, end, adjust):
        gate.wait()                    # 6 只必须同时在飞,否则超时
        return cached_frame(["2024-01-02"])

    monkeypatch.setattr(d, "_fetch_tx_direct", slow_direct)
    assert len(d.download_batch_daily(
        [f"00000{i}" for i in range(1, 7)], "20240101", "20240102")) == 6


# ==================== 指数基准 ====================

def test_index_symbol_prefix_rule_differs_from_stock_rule():
    assert dl.index_symbol("000852") == "sh000852"          # 中证1000 在上海
    assert dl.index_symbol("399001") == "sz399001"
    assert dl.index_symbol("sh000300") == "sh000300"
    # 个股规则会把 000852 判给深市,基准就会静默变成一只股票
    assert tx_symbol("000852") == "sz000852"


def test_download_index_daily_requests_without_adjust(tmp_path):
    d, sess = make_downloader(tmp_path, bars(["2024-01-02", "2024-01-03"]),
                             sym="sh000852")
    df = d.download_index_daily("000852", "2024-01-02", "2024-01-03")
    assert sess.params == [f"sh000852,day,,2024-01-03,{dl.TX_PAGE_ROWS},"]
    assert list(df["收盘"]) == [10.0, 10.0]
    assert df["日期"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02",
                                                          "2024-01-03"]
    # sh000 已在"已是股"的前缀表里,指数成交量不得再乘 100
    assert list(df["成交量"]) == [1000.0, 1000.0]


def test_download_index_daily_raises_on_empty(tmp_path):
    d, _ = make_downloader(tmp_path, [], sym="sh000852")
    with pytest.raises(RuntimeError):
        d.download_index_daily("000852", "2024-01-02", "2024-01-03")


# ==================== 令牌桶 ====================

def test_bucket_spaces_requests_at_the_configured_rate():
    b = TokenBucket(rate=1000.0, burst=1.0)
    t0 = time.monotonic()
    for _ in range(21):
        b.acquire()
    assert time.monotonic() - t0 >= 0.015                  # 20 个间隔 ≥ 20ms


def test_bucket_penalize_halves_and_recover_crests_at_full_rate():
    b = TokenBucket(rate=100.0)
    b.penalize()
    assert b.rate == pytest.approx(50.0)
    b.penalize()
    assert b.rate == pytest.approx(25.0)
    for _ in range(10):
        b.penalize()
    assert b.rate == 0.5                                   # 有地板,不会归零
    for _ in range(20):
        b.recover()
    assert b.rate == pytest.approx(100.0)                  # 不冲过设定速率
