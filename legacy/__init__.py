# -*- coding: utf-8 -*-
"""旧链路（Top-K 等权调仓）：menu.py / daily_runner.py / portfolio_manager.py。

现行默认 position_policy.enabled=true，这三件都不实现分数带位规则，
portfolio_manager.rebalance() 在策略开启时会直接抛错。每日调仓请走
`python live/run_daily.py`；这里只保留作旧账可查。

目录里放代码而不是删掉，是为了让 portfolio_state.json / logs/daily_pnl.csv
这些历史状态文件还能被读出来。
"""
