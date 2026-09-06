#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QuantLab — A股量化研究与回测框架 CLI 入口。

用法:
    python main.py download  下载数据
    python main.py factors   计算因子
    python main.py train     训练模型
    python main.py backtest  运行回测
    python main.py paper-trade  启动模拟盘
    python main.py report    生成报告
    python main.py pipeline  一键运行全流程
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.calendar import get_trading_calendar


def load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================== 子命令处理 ====================

def cmd_download(args):
    """下载股票数据。"""
    from data.cache import CacheManager
    from data.downloader import DataDownloader
    from data.cleaner import DataCleaner

    config = load_config(args.config)
    cfg_universe = config["universe"]

    cache = CacheManager(config["cache"]["directory"])
    downloader = DataDownloader(cache)

    # 获取股票池
    indices = args.universe.split(",") if args.universe else cfg_universe["indices"]
    symbols = downloader.get_universe_symbols(indices)

    if args.symbols:
        symbols = args.symbols.split(",")
        symbols = [s.strip().zfill(6) for s in symbols]

    # 转换日期格式
    start_ak = args.start.replace("-", "")
    end_ak = args.end.replace("-", "")

    # 下载日线
    data = downloader.download_batch_daily(symbols, start_ak, end_ak, adjust="qfq")

    # 清洗
    cleaner = DataCleaner()
    cleaned = cleaner.clean_all(data, min_listed_days=cfg_universe.get("min_listed_days", 60))

    logger.info(f"下载完成: {len(cleaned)} / {len(symbols)} 只有效数据")


def cmd_factors(args):
    """计算因子。"""
    from data.cache import CacheManager
    from data.downloader import DataDownloader
    from factors.technical import TechnicalFactors
    from factors.fundamental import FundamentalFactors
    from factors.processor import FactorProcessor
    from factors.evaluator import FactorEvaluator
    import pandas as pd

    config = load_config(args.config)
    cfg_factors = config["factors"]

    cache = CacheManager(config["cache"]["directory"])

    # 加载已有缓存数据
    symbols = cache.list_cached_symbols()
    if not symbols:
        logger.error("没有缓存数据，请先运行 download 命令")
        return

    logger.info(f"加载 {len(symbols)} 只股票的缓存数据...")

    # 加载日线
    data_dict = {}
    for sym in symbols:
        df = cache.get_daily(sym)
        if df is not None:
            df["日期"] = pd.to_datetime(df["日期"])
            mask = (df["日期"] >= args.start) & (df["日期"] <= args.end)
            df = df[mask]
            if not df.empty:
                data_dict[sym] = df

    logger.info(f"有效数据: {len(data_dict)} 只股票")

    # --- 技术因子 ---
    logger.info("计算技术因子...")
    all_tech_factors = []
    tech_periods = {
        "momentum_periods": cfg_factors["technical"]["momentum_periods"],
        "volatility_periods": cfg_factors["technical"]["volatility_periods"],
        "volume_ratio_periods": cfg_factors["technical"]["volume_ratio_periods"],
    }

    for sym, df in data_dict.items():
        factors = TechnicalFactors.compute_all(df, tech_periods)
        factors["date"] = df["日期"].values
        factors["symbol"] = sym
        all_tech_factors.append(factors)

    tech_panel = pd.concat(all_tech_factors, ignore_index=True)
    tech_panel["date"] = pd.to_datetime(tech_panel["date"])
    logger.info(f"技术因子: {len(tech_panel.columns) - 2} 个")

    # --- 基本面因子（默认关闭：实时快照平铺到历史日期是未来函数，回测会高估收益） ---
    use_spot = cfg_factors["fundamental"].get("use_spot_data", False)
    use_fin = cfg_factors["fundamental"].get("use_financial_data", False)
    if use_spot or use_fin:
        logger.warning("已启用基本面因子——实时快照数据映射到历史日期，"
                       "回测结果将被系统性高估，仅供研究参考！")
        downloader = DataDownloader(cache)
        fund_df = downloader.download_fundamentals_for_symbols(list(data_dict.keys()))

        if fund_df is not None and not fund_df.empty:
            fund_df = fund_df.set_index("symbol")
            fund_factors = FundamentalFactors.compute_all(fund_df)

            # 注意：这里把快照值映射到每个历史交易日，存在未来函数
            fund_rows = []
            for _, row in tech_panel[["date", "symbol"]].drop_duplicates().iterrows():
                sym = row["symbol"]
                d = row["date"]
                if sym in fund_factors.index:
                    frow = fund_factors.loc[sym].to_dict()
                    frow["date"] = d
                    frow["symbol"] = sym
                    fund_rows.append(frow)

            fund_panel = pd.DataFrame(fund_rows)
            logger.info(f"基本面因子: {len(fund_panel.columns) - 2} 个")

            # 合并技术+基本面
            factor_panel = tech_panel.merge(
                fund_panel, on=["date", "symbol"], how="left")
        else:
            logger.warning("无法获取基本面数据，仅使用技术因子")
            factor_panel = tech_panel
    else:
        logger.info("基本面因子已关闭（避免未来函数），仅使用技术因子")
        factor_panel = tech_panel

    # --- 因子处理 ---
    logger.info("因子处理...")
    processor = FactorProcessor()
    processed = processor.process(factor_panel, cfg_factors)

    # 保存
    os.makedirs("data/cache", exist_ok=True)
    processed.to_parquet("data/cache/factor_panel.parquet", index=False)
    logger.info(f"因子面板已保存: data/cache/factor_panel.parquet "
                f"({len(processed)} 条, {len(processed.columns)-2} 个因子)")

    # --- 因子评估 ---
    logger.info("因子评估...")
    # 构建前向收益标签
    forward_returns = {}
    for sym, df in data_dict.items():
        df = df.set_index("日期")
        ret = df["收盘"].pct_change(5).shift(-5)  # T+5 前向收益
        for d, val in ret.dropna().items():
            forward_returns[(d, sym)] = val

    fwd_series = pd.Series(forward_returns, name="forward_5d_return")
    fwd_series.index = pd.MultiIndex.from_tuples(
        fwd_series.index, names=["date", "symbol"])

    # 评估
    evaluator = FactorEvaluator()
    factor_cols = [c for c in processed.columns
                   if c not in ["date", "symbol"]]
    factor_panel_indexed = processed.set_index(["date", "symbol"])[factor_cols]

    # 只评估有 forward_return 对齐的日期
    common_dates = sorted(set(factor_panel_indexed.index.get_level_values("date")) &
                          set(fwd_series.index.get_level_values("date")))
    logger.info(f"对齐后评估日期: {len(common_dates)} 天")

    report = evaluator.full_report(
        factor_panel_indexed.loc[
            factor_panel_indexed.index.get_level_values("date").isin(common_dates)
        ],
        fwd_series,
    )

    print(report["ic_summary"].to_string())
    if report["redundant_pairs"]:
        print(f"\n高度相关因子对 (|corr| > 0.7):")
        for a, b, corr in report["redundant_pairs"][:10]:
            print(f"  {a} <-> {b}: {corr:.3f}")


def cmd_train(args):
    """Walk-Forward 滚动训练。

    核心原则：每个预测窗口只使用严格早于该窗口的数据训练，
    生成全样本外预测，保存到 data/cache/predictions.parquet，
    供回测和模拟盘直接使用（杜绝训练集内预测的未来函数）。
    """
    import pandas as pd
    import numpy as np
    from data.cache import CacheManager
    from models.dataset import DatasetBuilder
    from models.trainer import LightGBMTrainer

    config = load_config(args.config)
    cfg_model = config["model"]
    wf_cfg = cfg_model.get("walk_forward", {})
    min_train_months = args.min_train_months or wf_cfg.get("min_train_months", 24)
    retrain_months = args.retrain_months or wf_cfg.get("retrain_months", 6)

    # 加载因子面板
    factor_path = "data/cache/factor_panel.parquet"
    if not os.path.exists(factor_path):
        logger.error("因子面板不存在，请先运行 factors 命令")
        return

    factor_panel = pd.read_parquet(factor_path)
    factor_panel["date"] = pd.to_datetime(factor_panel["date"])
    logger.info(f"加载因子面板: {len(factor_panel)} 条")

    # 构建数据集
    builder = DatasetBuilder(
        horizon=cfg_model["horizon"],
        model_type=cfg_model["type"],
    )

    # 加载日线数据构建标签
    cache = CacheManager(config["cache"]["directory"])
    symbols = factor_panel["symbol"].unique().tolist()

    data_dict = {}
    for sym in symbols:
        df = cache.get_daily(sym)
        if df is not None:
            df["日期"] = pd.to_datetime(df["日期"])
            data_dict[sym] = df

    # 构建特征矩阵
    factor_cols = [c for c in factor_panel.columns
                   if c not in ["date", "symbol"]]
    feature_matrix = factor_panel.set_index(["date", "symbol"])[factor_cols]

    # 构建标签并合并
    dataset = builder.merge_features_labels(feature_matrix, data_dict)
    dataset = dataset.dropna()
    if dataset.empty:
        logger.error("数据集为空，无法训练")
        return

    feature_cols = [c for c in dataset.columns
                    if c not in ["date", "symbol", "label", "forward_return"]]
    label_col = "label" if cfg_model["type"] == "classifier" else "forward_return"

    dates = sorted(dataset["date"].unique())
    logger.info(f"数据集: {len(dataset)} 条 | {len(dates)} 个交易日 | "
                f"Walk-Forward: 训练≥{min_train_months}个月, "
                f"每{retrain_months}个月重训")

    params = cfg_model.get("params", {}).copy()
    trainer = LightGBMTrainer(model_type=cfg_model["type"], params=params)

    # --- 生成预测窗口（对齐到真实交易日） ---
    raw_windows = []
    ws = dates[0] + pd.DateOffset(months=min_train_months)
    while ws <= dates[-1]:
        raw_windows.append((ws, ws + pd.DateOffset(months=retrain_months)))
        ws = ws + pd.DateOffset(months=retrain_months)

    windows = []
    for ws, we in raw_windows:
        wsd = next((d for d in dates if d >= ws), None)
        if wsd is None:
            break
        wed = next((d for d in dates if d > wsd and d >= we), None)
        if wed is None:
            wed = dates[-1] + pd.Timedelta(days=1)
        windows.append((wsd, wed))
    logger.info(f"预测窗口: {len(windows)} 个")

    # --- 滚动训练 + 样本外预测 ---
    preds_all = []
    fold_metrics = []

    for fi, (ws, we) in enumerate(windows):
        train_mask = dataset["date"] < ws
        pred_mask = (dataset["date"] >= ws) & (dataset["date"] < we)
        if train_mask.sum() < 5000 or pred_mask.sum() == 0:
            logger.warning(f"窗口 {fi + 1} 训练集过小或预测集为空，跳过")
            continue

        X_train = dataset.loc[train_mask, feature_cols]
        y_train = dataset.loc[train_mask, label_col]

        # 训练集尾部 15% 作为早停验证集（仍严格早于窗口起点）
        mid = int(len(X_train) * 0.85)
        trainer.train(X_train.iloc[:mid], y_train.iloc[:mid],
                      X_train.iloc[mid:], y_train.iloc[mid:])

        X_pred = dataset.loc[pred_mask, feature_cols]
        y_true = dataset.loc[pred_mask, label_col]
        preds = trainer.model.predict(X_pred)

        fold = dataset.loc[pred_mask, ["date", "symbol"]].copy()
        fold["prediction"] = preds
        fold["forward_return"] = y_true.values

        rmse = float(np.sqrt(np.mean((y_true.values - preds) ** 2)))
        ic = fold.groupby("date").apply(
            lambda g: g["prediction"].corr(g["forward_return"],
                                          method="spearman")
        ).mean()
        fold_metrics.append({"fold": fi + 1, "train_end": ws,
                             "rmse": rmse, "ic": float(ic)})
        logger.info(f"Fold {fi + 1}: 训练截止 {ws.date()} | "
                    f"样本 {pred_mask.sum()} | RMSE={rmse:.4f} | "
                    f"日均RankIC={ic:.4f}")
        preds_all.append(fold)

    if not preds_all:
        logger.error("Walk-Forward 未生成任何预测")
        return

    # --- 保存全样本外预测 ---
    preds_df = pd.concat(preds_all, ignore_index=True)
    preds_df.to_parquet("data/cache/predictions.parquet", index=False)

    ic_all = float(np.mean([m["ic"] for m in fold_metrics]))
    rmse_all = float(np.mean([m["rmse"] for m in fold_metrics]))
    logger.info(f"预测已保存: data/cache/predictions.parquet "
                f"({len(preds_df)} 条 | 平均RankIC={ic_all:.4f} | "
                f"平均RMSE={rmse_all:.4f})")
    print("\nWalk-Forward 样本外汇总:")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: RMSE={m['rmse']:.4f}  RankIC={m['ic']:.4f}")
    print(f"  平均 RankIC = {ic_all:.4f}"
          f"（>0.02 勉强可用，>0.05 优秀，负值 = 信号无效）")

    # 保存最后一期模型（供参考/实盘续用）
    model_dir = config.get("model", {}).get("save_dir", "models/saved")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(
        model_dir,
        f"lgb_{cfg_model['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    trainer.save(model_path)

    # 特征重要性
    importance = trainer.feature_importance()
    print("\n特征重要性 (Top-20):")
    print(importance.head(20).to_string(index=False))


def cmd_backtest(args):
    """运行回测。"""
    import pandas as pd
    from data.cache import CacheManager
    from models.predictor import Predictor
    from models.trainer import LightGBMTrainer
    from backtest.engine import BacktestEngine
    from backtest.cost import TransactionCostModel
    from backtest.reporter import ReportGenerator

    config = load_config(args.config)
    cfg_backtest = config["backtest"]
    cfg_market = config["market"]

    # 加载因子面板
    factor_path = "data/cache/factor_panel.parquet"
    if not os.path.exists(factor_path):
        logger.error("因子面板不存在，请先运行 factors 命令")
        return

    factor_panel = pd.read_parquet(factor_path)
    factor_panel["date"] = pd.to_datetime(factor_panel["date"])

    # 构建特征矩阵
    factor_cols = [c for c in factor_panel.columns
                   if c not in ["date", "symbol"]]
    feature_matrix = factor_panel.set_index(["date", "symbol"])[factor_cols]

    # 优先使用 Walk-Forward 样本外预测；不存在时回退到单模型（有未来函数风险）
    pred_path = "data/cache/predictions.parquet"
    if os.path.exists(pred_path):
        logger.info("使用 Walk-Forward 样本外预测")
        preds_df = pd.read_parquet(pred_path)
        preds_df["date"] = pd.to_datetime(preds_df["date"])
        predictions = preds_df.set_index(["date", "symbol"])["prediction"]
        trainer = None
    else:
        logger.warning("未找到 predictions.parquet，回退到单模型"
                       "（训练集内预测，有未来函数风险！）")
        model_path = args.model_path
        if not os.path.isfile(model_path):
            # 尝试找最新的模型
            saved_dir = "models/saved"
            if os.path.exists(saved_dir):
                models = sorted([f for f in os.listdir(saved_dir) if f.endswith(".txt")])
                if models:
                    model_path = os.path.join(saved_dir, models[-1])
                    logger.info(f"使用最新模型: {model_path}")
                else:
                    logger.error("未找到已训练模型，请先运行 train 命令")
                    return
            else:
                logger.error(f"模型文件不存在: {model_path}")
                return
        trainer = LightGBMTrainer(model_type=config["model"]["type"])
        trainer.load(model_path)

    # 生成信号
    predictor = Predictor(
        trainer,
        top_k=cfg_backtest["max_positions"],
        position_sizing=cfg_backtest["position_sizing"],
    )
    if trainer is None:
        signals = predictor.generate_signals_from_series(predictions)
    else:
        signals = predictor.generate_signals(feature_matrix)

    # 加载日线
    cache = CacheManager(config["cache"]["directory"])
    symbols = factor_panel["symbol"].unique().tolist()
    data_dict = {}
    for sym in symbols:
        df = cache.get_daily(sym)
        if df is not None:
            df["日期"] = pd.to_datetime(df["日期"])
            data_dict[sym] = df

    # 成本模型
    cost = TransactionCostModel(
        commission_rate=cfg_market["commission_rate"],
        min_commission=cfg_market["min_commission"],
        stamp_tax_rate=cfg_market["stamp_tax_rate"],
        slippage_rate=cfg_market["slippage_rate"],
    )

    # 回测引擎
    engine = BacktestEngine(
        initial_capital=cfg_backtest.get("initial_capital", args.capital),
        rebalance_frequency=cfg_backtest["rebalance_frequency"],
        max_positions=cfg_backtest["max_positions"],
        cost_model=cost,
    )

    # 基准指数（用于超额对比，下载失败则跳过）
    benchmark_curve = None
    try:
        import akshare as ak
        bm_code = cfg_backtest.get("benchmark", "000852")
        bm_start = factor_panel["date"].min().strftime("%Y%m%d")
        bm_end = factor_panel["date"].max().strftime("%Y%m%d")
        bm_df = ak.index_zh_a_hist(symbol=bm_code, period="daily",
                                   start_date=bm_start, end_date=bm_end)
        if bm_df is not None and not bm_df.empty:
            bm_df["日期"] = pd.to_datetime(bm_df["日期"])
            benchmark_curve = bm_df.set_index("日期")["收盘"]
            logger.info(f"基准 {bm_code}: {len(benchmark_curve)} 天")
    except Exception as e:
        logger.warning(f"基准指数下载失败，跳过: {e}")

    result = engine.run(data_dict, signals,
                        benchmark_prices=benchmark_curve)

    if not result:
        return

    # 生成报告
    reporter = ReportGenerator()
    print(reporter.console_report(result["metrics"]))

    # HTML 报告
    import plotly
    if hasattr(plotly, "graph_objects"):
        html_path = reporter.html_report(
            equity_curve=result["equity_curve"],
            benchmark_curve=result.get("benchmark_curve"),
            daily_returns=result["daily_returns"],
            trades=result["trades"],
            metrics=result["metrics"],
        )
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        logger.info(f"HTML 报告已打开: {html_path}")


def cmd_paper_trade(args):
    """启动模拟盘交易。"""
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    from data.cache import CacheManager
    from data.downloader import DataDownloader
    from models.predictor import Predictor
    from models.trainer import LightGBMTrainer
    from paper_trade.broker import SimulatedBroker
    from paper_trade.portfolio import PortfolioTracker
    from paper_trade.journal import TradeJournal
    from utils.calendar import get_trading_calendar, get_month_end_trading_days

    config = load_config(args.config)
    cfg_market = config["market"]
    cfg_backtest = config["backtest"]

    # 加载因子面板和模型
    factor_path = "data/cache/factor_panel.parquet"
    if not os.path.exists(factor_path):
        logger.error("因子面板不存在，请先运行 factors 命令")
        return

    factor_panel = pd.read_parquet(factor_path)
    factor_panel["date"] = pd.to_datetime(factor_panel["date"])

    # 优先使用 Walk-Forward 样本外预测
    pred_path = "data/cache/predictions.parquet"
    if os.path.exists(pred_path):
        logger.info("使用 Walk-Forward 样本外预测")
        preds_df = pd.read_parquet(pred_path)
        preds_df["date"] = pd.to_datetime(preds_df["date"])
        predictions = preds_df.set_index(["date", "symbol"])["prediction"]
        trainer = None
    else:
        model_path = args.model_path
        if not os.path.isfile(model_path):
            saved_dir = "models/saved"
            models = sorted([f for f in os.listdir(saved_dir) if f.endswith(".txt")])
            if models:
                model_path = os.path.join(saved_dir, models[-1])
            else:
                logger.error("未找到已训练模型")
                return
        trainer = LightGBMTrainer(model_type=config["model"]["type"])
        trainer.load(model_path)

    # 初始化模拟券商
    broker = SimulatedBroker(
        initial_cash=args.capital,
        commission_rate=cfg_market["commission_rate"],
        min_commission=cfg_market["min_commission"],
        stamp_tax_rate=cfg_market["stamp_tax_rate"],
        slippage_rate=cfg_market["slippage_rate"],
        lot_size=cfg_market["lot_size"],
    )

    portfolio = PortfolioTracker(initial_capital=args.capital)
    journal = TradeJournal()

    # ---- 加载数据 ----
    cache = CacheManager(config["cache"]["directory"])
    symbols = factor_panel["symbol"].unique().tolist()
    symbols_in_cache = [s for s in symbols if cache.get_daily(s) is not None]
    logger.info(f"缓存中可用: {len(symbols_in_cache)} / {len(symbols)} 只股票")

    data_dict = {}
    for sym in symbols_in_cache:
        df = cache.get_daily(sym)
        df["日期"] = pd.to_datetime(df["日期"])
        data_dict[sym] = df

    # ---- 构建特征和预测器 ----
    factor_cols = [c for c in factor_panel.columns
                   if c not in ["date", "symbol"]]
    feature_matrix = factor_panel.set_index(["date", "symbol"])[factor_cols]

    predictor = Predictor(
        trainer,
        top_k=cfg_backtest["max_positions"],
        position_sizing=cfg_backtest["position_sizing"],
    )

    # ---- 生成全时段信号 ----
    if trainer is None:
        all_signals = predictor.generate_signals_from_series(predictions)
    else:
        all_signals = predictor.generate_signals(feature_matrix)

    # ---- 按日期模拟真实交易 ----
    all_dates = sorted(factor_panel["date"].unique())
    rebalance_dates = get_month_end_trading_days(
        str(all_dates[0])[:10], str(all_dates[-1])[:10]
    )
    rebalance_dates = [d for d in rebalance_dates
                       if pd.Timestamp(d) in all_dates]

    logger.info(f"模拟盘开始: {args.capital:,.0f} 元, "
                f"{len(rebalance_dates)} 个调仓日, "
                f"{len(all_dates)} 个交易日")

    print(f"\n{'='*55}")
    print(f"  QuantLab 模拟盘 (Walk-Forward 仿真)")
    print(f"  初始资金: {args.capital:,.0f} 元 | "
          f"持仓上限: {cfg_backtest['max_positions']} 只")
    print(f"  佣金: {cfg_market['commission_rate']*10000:.0f}bp | "
          f"印花税: {cfg_market['stamp_tax_rate']*10000:.0f}bp(卖)")
    print(f"  调仓频率: 月末 | 起止: {all_dates[0].strftime('%Y-%m-%d')} ~ "
          f"{all_dates[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*55}\n")

    for i, rebal_dt in enumerate(tqdm(rebalance_dates, desc="模拟盘中")):
        rebal_ts = pd.Timestamp(rebal_dt)

        # 调到下一个交易日执行
        exec_date = None
        for d in all_dates:
            if d > rebal_ts:
                exec_date = d
                break
        if exec_date is None:
            continue

        # ---- 当日盯市 + 解锁T+1 ----
        market_snapshot = {}
        for sym in data_dict:
            df_sym = data_dict[sym].set_index("日期")
            if exec_date in df_sym.index:
                row = df_sym.loc[exec_date]
                market_snapshot[sym] = {
                    "open": float(row["开盘"]),
                    "high": float(row["最高"]),
                    "low": float(row["最低"]),
                    "close": float(row["收盘"]),
                    "volume": float(row["成交量"]),
                    "at_limit_up": float(row.get("涨跌幅", 0)) >= 9.5,
                    "at_limit_down": float(row.get("涨跌幅", 0)) <= -9.5,
                }

        # 执行撮合
        filled = broker.process_daily(exec_date, market_snapshot)

        # 记录成交
        for order in filled:
            journal.log_trade(order)

        # ---- 调仓日：生成信号并下单 ----
        if rebal_ts in all_signals.index.get_level_values("date"):
            try:
                day_signals = all_signals.xs(rebal_ts, level="date")
            except KeyError:
                day_signals = all_signals[all_signals.index.get_level_values("date") == rebal_ts]

            target = day_signals[day_signals["weight"] > 0]
            target_symbols = set(target.index)

            # 卖：不在目标中的持仓
            for sym, pos in list(broker.positions.items()):
                if sym not in target_symbols and pos.available_shares > 0:
                    broker.place_market_order(sym, "sell", pos.available_shares)

            # 买：目标持仓
            cash_per_stock = broker.cash / max(len(target_symbols), 1)
            for sym in target_symbols:
                if sym in market_snapshot:
                    price = market_snapshot[sym]["open"]
                    shares = broker.lot_size * (
                        int(cash_per_stock / price) // broker.lot_size
                    )
                    if shares >= broker.lot_size:
                        broker.place_market_order(sym, "buy", shares)

        # ---- 每日快照（每月记录一次） ----
        portfolio.update(exec_date, broker)
        journal.log_daily_snapshot(exec_date, broker, portfolio)

        # 风控告警
        alerts = portfolio.check_risk_limits(broker)
        for alert in alerts:
            logger.warning(alert)

        # 进度打印（每6个月）
        if i % 6 == 0 and i > 0:
            pnl = broker.get_total_value() - args.capital
            print(f"  [{exec_date.strftime('%Y-%m')}] "
                  f"净值={broker.get_total_value()/args.capital:.3f} | "
                  f"持仓={len(broker.positions)} | "
                  f"累计盈亏={pnl:+,.0f}")

    # ---- 最终报告 ----
    final_value = broker.get_total_value()
    final_pnl = final_value - args.capital
    final_return = final_pnl / args.capital * 100

    print(f"\n{'='*55}")
    print(f"  模拟盘结束")
    print(f"  最终资金: {final_value:,.0f} 元")
    print(f"  累计盈亏: {final_pnl:+,.0f} 元 ({final_return:+.2f}%)")
    print(f"  当前持仓: {len(broker.positions)} 只")
    print(f"{'='*55}")

    # 交易统计
    trade_report = journal.generate_report()
    print(trade_report)

    # 导出
    journal.export_csv("paper_trade_result")

    # 净值图
    try:
        import plotly.graph_objects as go
        equity = portfolio.get_equity_curve()
        if len(equity) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity.index, y=equity.values, mode="lines",
                name="模拟盘净值", line=dict(color="#4ec9b0", width=2)))
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig.update_layout(title="模拟盘净值曲线", yaxis_title="净值",
                              template="plotly_dark", height=400)
            html_path = f"reports/paper_trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            os.makedirs("reports", exist_ok=True)
            fig.write_html(html_path)
            logger.info(f"净值图已保存: {html_path}")
    except Exception as e:
        logger.warning(f"净值图生成失败: {e}")


def cmd_pipeline(args):
    """一键运行全流程。"""
    logger.info("===== QuantLab Pipeline 开始 =====")
    logger.info(f"股票池: {args.universe}")
    logger.info(f"日期范围: {args.start} ~ {args.end}")
    logger.info(f"初始资金: {args.capital:,.0f} 元")

    # Step 1: Download
    args_dl = argparse.Namespace(
        config=args.config, universe=args.universe,
        symbols=None, start=args.start, end=args.end,
    )
    cmd_download(args_dl)

    # Step 2: Factors
    args_factors = argparse.Namespace(
        config=args.config, start=args.start, end=args.end,
    )
    cmd_factors(args_factors)

    # Step 3: Train（Walk-Forward 滚动训练，全样本外预测）
    args_train = argparse.Namespace(
        config=args.config, min_train_months=None,
        retrain_months=None,
    )
    cmd_train(args_train)

    # Step 4: Backtest
    model_dir = "models/saved"
    models = sorted([f for f in os.listdir(model_dir) if f.endswith(".txt")])
    if models:
        model_path = os.path.join(model_dir, models[-1])
    else:
        logger.error("模型训练未生成文件")
        return

    args_bt = argparse.Namespace(
        config=args.config, model_path=model_path,
        capital=args.capital,
    )
    cmd_backtest(args_bt)

    logger.info("===== Pipeline 完成 =====")


def cmd_report(args):
    """生成 HTML 报告（从已有的回测结果）。"""
    logger.info("请使用 backtest 命令生成报告，或从已有数据重新运行回测")


# ==================== CLI 设置 ====================

def main():
    parser = argparse.ArgumentParser(
        description="QuantLab — A股量化研究与回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py download --universe 000300 --start 2023-01-01 --end 2024-12-31
  python main.py factors --start 2023-01-01 --end 2024-12-31
  python main.py train --train-end 2023-12-31 --test-start 2024-01-01
  python main.py backtest --capital 1000000
  python main.py pipeline --universe 000300 --start 2020-01-01 --end 2024-12-31
        """
    )

    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径 (默认: config.yaml)")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # download
    p_dl = subparsers.add_parser("download", help="下载股票数据")
    p_dl.add_argument("--universe", default="000852",
                       help="指数代码，逗号分隔 (默认: 000852=中证1000)")
    p_dl.add_argument("--symbols", default=None,
                       help="指定股票代码，逗号分隔 (覆盖 --universe)")
    p_dl.add_argument("--start", default="2023-01-01", help="起始日期")
    p_dl.add_argument("--end", default="2024-12-31", help="结束日期")

    # factors
    p_factors = subparsers.add_parser("factors", help="计算因子")
    p_factors.add_argument("--start", default="2023-01-01")
    p_factors.add_argument("--end", default="2024-12-31")

    # train
    p_train = subparsers.add_parser("train", help="Walk-Forward 滚动训练")
    p_train.add_argument("--min-train-months", type=int, default=None,
                         help="首个预测窗口前的最短训练期/月 (默认取 config)")
    p_train.add_argument("--retrain-months", type=int, default=None,
                         help="滚动重训间隔/月 (默认取 config)")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="运行回测")
    p_bt.add_argument("--model-path", default="models/saved",
                      help="模型文件路径")
    p_bt.add_argument("--capital", type=float, default=1_000_000,
                      help="初始资金 (默认: 100万)")

    # paper-trade
    p_pt = subparsers.add_parser("paper-trade", help="启动模拟盘")
    p_pt.add_argument("--model-path", default="models/saved")
    p_pt.add_argument("--capital", type=float, default=1_000_000)

    # report
    p_rpt = subparsers.add_parser("report", help="生成报告")
    p_rpt.add_argument("--backtest-id", default="latest")

    # pipeline
    p_pl = subparsers.add_parser("pipeline", help="一键运行全流程")
    p_pl.add_argument("--universe", default="000852")
    p_pl.add_argument("--start", default="2021-01-01")
    p_pl.add_argument("--end", default="2025-12-31")
    p_pl.add_argument("--capital", type=float, default=1_000_000)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 设置日志
    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logger(
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/quantlab.log"),
        rotation=log_cfg.get("rotation", "10 MB"),
        retention=log_cfg.get("retention", "30 days"),
    )

    # 分发
    commands = {
        "download": cmd_download,
        "factors": cmd_factors,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "paper-trade": cmd_paper_trade,
        "report": cmd_report,
        "pipeline": cmd_pipeline,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
