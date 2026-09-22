# QuantLab — A股量化研究与回测框架

> 面向初学者的完整量化研究框架：数据 → 因子 → 模型 → 回测 → 模拟盘

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行全流程
python main.py pipeline --universe 000852 --start 2021-01-01 --end 2025-12-31

# 3. 或分步执行
python main.py download --universe 000852 --start 2021-01-01 --end 2025-12-31
python main.py factors --start 2021-01-01 --end 2025-12-31
python main.py train                      # Walk-Forward 滚动训练（可用 --min-train-months/--retrain-months 覆盖）
python main.py backtest --capital 1000000
```

## 项目结构

```
QuantLab/
├── main.py               # CLI 入口
├── config.yaml           # 全局配置
├── requirements.txt      # 依赖列表
├── data/                 # 数据层
│   ├── downloader.py     # AKShare 数据下载
│   ├── cleaner.py        # 数据清洗
│   └── cache.py          # Parquet 缓存
├── factors/              # 因子引擎
│   ├── technical.py      # 技术因子（动量、波动率、RSI、MACD等）
│   ├── fundamental.py    # 基本面因子（PE、PB、市值等）
│   ├── processor.py      # 因子处理（缩尾、标准化、中性化）
│   └── evaluator.py      # 因子评估（IC、IR、相关性）
├── models/               # 模型层
│   ├── dataset.py        # 数据集构建（时序交叉验证）
│   ├── trainer.py        # LightGBM 训练（含Optuna调参）
│   └── predictor.py      # 预测信号生成
├── backtest/             # 回测引擎
│   ├── engine.py         # 向量化回测
│   ├── cost.py           # 交易成本（佣金、印花税、滑点）
│   ├── metrics.py        # 绩效指标
│   └── reporter.py       # 报告生成（HTML + 控制台）
├── paper_trade/          # 模拟盘
│   ├── broker.py         # 模拟券商（T+1、涨跌停、整手）
│   ├── portfolio.py      # 持仓跟踪 + 风控
│   └── journal.py        # 交易日志
└── utils/
    ├── calendar.py        # A股交易日历
    └── logger.py          # 日志
```

## 核心设计

| 模块 | 关键特性 |
|------|---------|
| **数据** | 腾讯行情直连（前复权）+ akshare/东财两级降级，Parquet本地缓存，8线程并发 |
| **因子** | ~28个技术因子，缩尾→标准化→滞后防未来函数；基本面默认关闭（实时快照映射历史=未来函数） |
| **模型** | LightGBM收益率回归，Walk-Forward滚动训练（每6个月重训，全样本外预测，杜绝未来函数） |
| **回测** | 事件循环引擎，含佣金/印花税/滑点，月度调仓，涨停不买/跌停不卖/停牌不成交，缺行按最近收盘估值，支持基准指数超额对比 |
| **模拟盘** | T+1规则，100股整手，涨跌停不可交易（板块 10%/20%，与回测共用 `utils/market_rules.py`） |

> **2026-09-22 最新一轮全链路数字**（桌面机：新下载器重建缓存 → 因子 → 8 折
> Walk-Forward 训练 28 s → `verify_backtest.py`），完整表在 `reports/verify_results.txt`，
> `evidence/verify_results.txt` 留作执行层修复**前**的对照存档。
>
> | 区间 | 策略累计 | 年化 | 年化波动 | Sharpe | 最大回撤 | 成本 |
> |---|---|---|---|---|---|---|
> | 2023-02-01 ~ 2026-09-22（887 日） | +30.95% | 7.96% | 27.11% | 0.220 | 43.16% | 11.51% 本金 |
> | 2023-02-01 ~ 2025-12-31（与旧存档同区间） | +20.24% | 6.75% | 27.55% | — | 43.16% | — |
>
> 同区间从旧存档的 +54.79% / Sharpe 0.42 掉到 +20.24%：主要是执行层修复
> （差额建仓、无前视盯市、涨停/停牌不可成交）把原先过于乐观的撮合挤掉了，
> 但这一轮模型也在重训（缓存变了、`predictions.parquet` 是新生成的），
> 所以它不是逐值可比的对照实验。
>
> **基准口径纠正**：`verify_backtest.py` 过去把"今日 1000 只成分的等权组合"
> 标成"中证1000"来算超额。等权组合同期 +55.72%，于是结论是"跑输 20 多个点"；
> 真实指数 `sh000852` 同期只有 +12.18%（年化 3.32%），换成它以后是超额 **+18.77pp**、
> 信息比率 **0.44**。两个基准现在都印在报告里——在这个股票池上做多，等权组合才是该翻的
> hurdle，但股票池本身用的是**当前**成分，两侧都带幸存者偏差，绝对收益都偏乐观。
> （另修掉一处区间错配：基准原先按 `predictions` 末日裁掉，而净值曲线走到缓存末日，
> 最后约一个月的指数涨幅被抹掉了 3 个百分点。）

## 数据获取（`data/downloader.py`，2026-09-22 重写）

旧路径是 akshare 的 `stock_zh_a_hist_tx`：它按**自然年**一年一个 HTTP 请求，
复用不了连接，仓库里还额外对每只股票 `sleep(0.5+随机)` 串行跑 —— 1000 只约
75 分钟。新路径直连腾讯 `newfqkline`，一次 640 根、按 `end` 向前翻页，
线程池并发 + 全局令牌桶限速：

| | 旧 | 新 |
|---|---|---|
| 1000 只全量 | ≈75 min | **2.3 min**（8 线程 / 20 请求每秒，7.3 只每秒） |
| 12 只 A/B | 24.8 s | **0.7 s**（34.7×） |
| 缓存已覆盖 | 仍发请求 | 0 请求，直接切片 |

其余口径：

- **三级降级**：直连腾讯 → akshare 腾讯 → 东方财富。任一级失败才进下一级，
  每次失败把令牌桶速率减半（有地板），恢复时 ×1.5 不冲过设定值。
- **列与单位**（三条路径写进缓存后完全一致）：`成交量`=股、`成交额`=元、
  `换手率`=小数。两处是实测出来的坑：akshare 把 `sz000` 当指数漏乘 100，
  而 `sz000` 同时命中平安银行/万科这类深市主板股票（实测成交量×价格 vs 成交额
  差 100 倍）；东财 `stock_zh_a_hist` 的成交量是手。修正只动绝对值——项目里
  成交量只用于量比（与自身均量之比）和停牌判定（是否为 0），尺度无关，
  因此不改任何已发布数字。
- **等价性**：7 只股票（含科创板 / 001 号段 / 长历史）与 akshare 逐字段比对，
  maxdiff 全为 0；`tests/test_downloader.py` 24 项离线测试钉住翻页语义、
  单位换算、缓存增量与降级链。
- **基准指数**：`download_index_daily("000852")` 走同一条直连通道（东财指数接口
  在本机 IP 上常年 `RemoteDisconnected`）。指数前缀规则与个股不同，
  000 号段在上海，故单独走 `index_symbol()`，不能让 `tx_symbol()` 把它判成深市股票。

## 分数带位仓位管理（`position_policy`，默认关闭）

Top-K 等权 50 只的问题：权重与分数无关，第 50 名和第 1 名同仓，月末一次性换手
把全部筹码暴露在同一天的模型误差上。打开 `config.yaml` 的 `position_policy.enabled`
后，"该买谁"不再是名单，而是**分数带 + 每只股票自己的建仓基准**的函数
（状态机在 `utils/position_policy.py`，与 quantlab2 同一份实现）：

| 规则 | 常量 | 语义 |
|------|------|------|
| 建仓线 / 清仓线 | `buy_score` / `sell_score` | 分数 ≥ 建仓线才入场；跌破清仓线整只清空 |
| 基准仓 | `base_weight` = 5% | 每笔建仓的绝对市值占比（`normalize=False`，归一化会让"5%"失去含义） |
| 建仓上限 | `max_entry_weight` = 8% | 分数越高首笔越大，到 `strong_score` 给满，封顶 8% |
| 补仓触发 | Δ = `step_multiplier` × (建仓分数 − 建仓线) | 比 `ref_score` 再涨够一个 Δ 才补一档，Δ 冻结在该股状态里 |
| 持仓上限 | `max_position_weight` = 16% | 每档 `add_step_weight`，到 16% 锁死 |
| 减仓 | 带内回落 | 分数下降但未破清仓线 → 每档 `trim_step_weight`，减到低于 `min_hold_weight` 改清仓 |
| 回补禁令 | `trim_price` | 记实际减仓成交价，此后补仓参考价不得高于它（`trim_price_ttl_days` 自然日内有效） |
| 最小笔额 | `min_trade_value` = 10000 | 不足一万不下单，下一轮分数继续走再试 |

链路分工（**只有这三条**实现策略，其余调仓入口在策略打开时会明确报错/告警，
不会静默按等权摊派下单）：

- `python main.py backtest` — `backtest/engine.py` 逐评估日跑同一个状态机，
  摘要里多出"策略动作 / 平均目标仓位 / 平均持仓只数"三行诊断。
- `python live/run_daily.py [--simulate|--qmt]` — 策略模式下**每个交易日**评估
  （月末才看一次会把月内的补/减档全丢掉），每只股票的状态存
  `live/state/policy_state.json`，只在**实际股数发生变化**后推进；涨跌停、停牌、
  资金不足挡下的单子下一轮自动重做。
- `python main.py paper-trade` — 走同一条状态机，但节奏仍是月末（函数按
  `rebalance_dates` 循环），启动时会打警告：看策略表现请用 `backtest`。

⚠ 阈值量纲取决于 `model.type`：`regressor` 的分数是预测 20 日收益率
（`buy_score: 0.02` = 预期 +2%），`classifier` 的分数是上涨概率（0~1），
默认阈值在分类头下等于不建仓。改动任何带位参数都会让既有回测结论作废，
需要重跑回测（`predictions.parquet` 本身不变，**不需要重训**）。

## 命令参考

```bash
python main.py download      # 下载数据
python main.py factors        # 计算因子 + 评估
python main.py train          # 训练模型
python main.py backtest       # 运行回测 + 生成报告
python main.py paper-trade    # 启动模拟盘
python main.py pipeline       # 一键全流程
```

## 配置

编辑 `config.yaml` 修改：

- 数据下载（`download:`：并发线程数、全局请求速率上限、单次 HTTP 超时）
- 交易成本（佣金、印花税、滑点）
- 因子参数（周期、处理方法）
- 模型参数（LightGBM、调参）
- 回测参数（初始资金、调仓频率、持仓数）

## 要求

- Python 3.11+
- Windows / macOS / Linux
