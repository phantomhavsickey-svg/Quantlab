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
| **数据** | AKShare免费数据源（前复权），Parquet本地缓存 |
| **因子** | ~28个技术因子，缩尾→标准化→滞后防未来函数；基本面默认关闭（实时快照映射历史=未来函数） |
| **模型** | LightGBM收益率回归，Walk-Forward滚动训练（每6个月重训，全样本外预测，杜绝未来函数） |
| **回测** | 事件循环引擎，含佣金/印花税/滑点，月度调仓，涨停不买/跌停不卖/停牌不成交，缺行按最近收盘估值，支持基准指数超额对比 |
| **模拟盘** | T+1规则，100股整手，涨跌停不可交易（板块 10%/20%，与回测共用 `utils/market_rules.py`） |

> 2026-09-21 与 quantlab2 同步了执行层口径（差额建仓、盯市不再前视、可交易性约束、
> 缺行估值）。`evidence/verify_results.txt` 里的指标是那之前的引擎跑出来的，
> 需要在有缓存的机器上重跑 `python verify_backtest.py` 才会更新；因子、标签与模型未动。

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

- 交易成本（佣金、印花税、滑点）
- 因子参数（周期、处理方法）
- 模型参数（LightGBM、调参）
- 回测参数（初始资金、调仓频率、持仓数）

## 要求

- Python 3.11+
- Windows / macOS / Linux
