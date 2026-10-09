"""factor_panel 里指定区间每个日历日有多少只股票有记录（只读）。

用法: python research/bar_days_check.py [起始日 2026-09-14] [结束日 2026-09-30]

用来区分"跳过的日期是非交易日"还是"数据没拉到"。只读 date 一列，约 3 秒。
注意 utils/calendar.py 依赖的 cn-stock-holidays 在本机未安装，is_trading_day()
降级成"只排除周末"，所以它会把 2026-09-25 误判成交易日 —— 以缓存里有没有 bar 为准。
"""
import sys

import pandas as pd

start = sys.argv[1] if len(sys.argv) > 1 else "2026-09-14"
end = sys.argv[2] if len(sys.argv) > 2 else "2026-09-30"

d = pd.read_parquet("data/cache/factor_panel.parquet", columns=["date"])
s = d["date"].dt.normalize().value_counts().sort_index()
w = s[(s.index >= start) & (s.index <= end)]
week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
print("factor_panel.parquet %s..%s 各日历日记录数:" % (start, end))
for k, v in w.items():
    print("  %s %s  %5d 只" % (k.date(), week[k.weekday()], int(v)))
if len(w):
    missing = [str(x.date()) for x in pd.date_range(start, end, freq="D")
               if x not in set(w.index) and x.weekday() < 5]
    print("区间内无记录的工作日: %s" % (missing or "无"))

