# -*- coding: utf-8 -*-
"""市场模式因子 —— 5 个"全池同值"的逐日市场状态列。

与 technical.py 的个股因子不同,这 5 列在同一交易日对所有股票取同一个数
(全池等权口径,与 utils/exposure.py 的已实现波动同源),所以:

1. 不能进 FactorProcessor.process()。截面缩尾对常数截面无效果,而 z-score
   在 std=0 时把整列写成 0.0(processor.py:85-86),信息全丢。本模块的输出
   由 main.py 的 cmd_factors 在 process() 之后直接 merge 进面板。
2. 时点语义与其它因子一致:面板 date=T 的行只携带 T-1 日收盘前算得出的值
   (整体 shift 1,与 lag_factor_panel 的约定相同)。
3. 除面板首日外不产生新的 NaN。dataset.dropna()(main.py cmd_train 第 274 行)
   丢掉任何一列为 NaN 的行;若新列在暖机段留 NaN,会连带丢掉整行,把
   walk-forward 的窗口起点整体推后,新旧两版回测就不可比。所以滚动窗口
   暖机段用替代值:ret/dd 用"起点至今",vol/snr/breadth 用 0.0 —— 后者与
   25 个技术列在"全列为 NaN 的暖机日被 z-score 填 0.0"的既有惯例一致。
"""
import numpy as np
import pandas as pd

MARKET_COLS = ("mkt_ret_20d", "mkt_vol_20d", "mkt_snr_20d",
               "mkt_breadth_20d", "mkt_dd_60d")

_RET_N = 20      # 动量/信噪比/宽度的窗口(交易日)
_VOL_N = 20      # 波动窗口,与 exposure_overlay.vol_lookback_days 同值同式
_DD_N = 60       # 回撤窗口
_TRADING_DAYS = 252


def compute_market_regime(data_dict: dict) -> pd.DataFrame:
    """{symbol: 日线 DataFrame} → date + 5 列市场模式表(已完成 shift 1)。

    列定义(全池等权,池 = data_dict 里的全部股票):
        mkt_ret_20d      全池等权净值 20 日涨跌幅
        mkt_vol_20d      全池等权日收益 20 日标准差 × √252(与暴露层 B 同式)
        mkt_snr_20d      20 日涨跌幅 ÷ (日收益标准差 × √20)
        mkt_breadth_20d  20 日收益为正的股票占比(分母 = 有 20 日历史的股票)
        mkt_dd_60d       全池等权净值相对近 60 日最高的回撤(≤0)
    """
    from utils.exposure import close_panel

    close = close_panel(data_dict)
    if close.empty:
        raise ValueError("日线缓存为空,无法计算市场模式因子")

    ew = close.pct_change().mean(axis=1).fillna(0.0)   # 与 exposure.py:192 同口径
    level = (1.0 + ew).cumprod()                       # 全池等权净值,起点=1

    ret20 = level / level.shift(_RET_N).fillna(level.iloc[0]) - 1.0
    daily_std = ew.rolling(_VOL_N, min_periods=2).std(ddof=1)
    vol20 = daily_std * np.sqrt(_TRADING_DAYS)
    snr20 = ret20 / (daily_std * np.sqrt(_RET_N))
    r20 = close / close.shift(_RET_N) - 1.0
    ready = r20.notna()
    breadth20 = (r20.where(ready) > 0).sum(axis=1) / ready.sum(axis=1).replace(0, np.nan)
    dd60 = level / level.rolling(_DD_N, min_periods=1).max() - 1.0

    out = pd.DataFrame({
        "mkt_ret_20d": ret20,
        "mkt_vol_20d": vol20,
        "mkt_snr_20d": snr20,
        "mkt_breadth_20d": breadth20,
        "mkt_dd_60d": dd60,
    })
    # 暖机段替代值:vol/snr/breadth 的 NaN 填 0.0(见模块头部第 3 条)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 与其它因子同语义:面板 T 行 = T-1 收盘前的信息
    out = out.shift(1)
    out.index.name = "date"
    return out.reset_index()
