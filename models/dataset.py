"""
数据集构建 — 特征矩阵、标签生成、时序划分。
"""

import pandas as pd
import numpy as np
from loguru import logger


class DatasetBuilder:
    """从因子面板构建 ML 数据集。

    核心职责:
        1. 将因子面板 + 价格数据合并为特征矩阵 X
        2. 计算前向收益标签 y
        3. 严格按时间顺序划分训练/验证/测试集

    防未来函数规则:
        - T 日的因子值由 T-1 日收盘价等数据算出（因子已 lag 1 期）
        - 标签 = T+horizon 相对 T 的收益率
        - 所以训练数据用到的全部信息在预测时刻都是已知的
    """

    def __init__(self, horizon: int = 5, model_type: str = "classifier"):
        """
        Args:
            horizon: 预测未来 N 个交易日的收益率
            model_type: "classifier" (涨跌二分类) / "regressor" (收益率回归)
        """
        self.horizon = horizon
        self.model_type = model_type

    # ==================== 标签构建 ====================

    def build_label(self, price_df: pd.DataFrame) -> pd.Series:
        """计算前向 N 日收益率标签。

        Args:
            price_df: 单只股票的 OHLCV DataFrame（必须含 '日期', '收盘'）

        Returns:
            Series (index=日期) 前向 N 日收益率
        """
        close = price_df.set_index("日期")["收盘"]
        # 前向收益: (T+N日收盘 / T日收盘 - 1)
        forward_return = close.shift(-self.horizon) / close - 1.0
        forward_return.name = f"forward_{self.horizon}d_return"
        return forward_return

    def build_label_classification(self, price_df: pd.DataFrame,
                                   threshold: float = 0.0) -> pd.Series:
        """计算前向 N 日涨跌分类标签。

        Returns:
            Series: 1 = 上涨, 0 = 下跌
        """
        fwd_ret = self.build_label(price_df)
        return (fwd_ret > threshold).astype(int).rename("label")

    # ==================== 特征矩阵 ====================

    def build_feature_matrix(self, factor_panel: pd.DataFrame,
                             symbol_col: str = "symbol",
                             date_col: str = "date") -> pd.DataFrame:
        """将因子面板整理为特征矩阵。

        Args:
            factor_panel: 因子面板 DataFrame
                - 如果含 date_col 和 symbol_col，说明是多股票长表格式
                - 如果是 MultiIndex (date, symbol)，直接使用

        Returns:
            DataFrame 以 (date, symbol) 为 MultiIndex，列为因子
        """
        if isinstance(factor_panel.index, pd.MultiIndex):
            return factor_panel.dropna(how="all")

        # 长表格式转面板
        factor_cols = [c for c in factor_panel.columns
                       if c not in [date_col, symbol_col]]
        panel = factor_panel.set_index([date_col, symbol_col])[factor_cols]
        return panel.dropna(how="all")

    # ==================== 合并特征和标签 ====================

    def merge_features_labels(self, factor_panel: pd.DataFrame,
                              data_dict: dict) -> pd.DataFrame:
        """将因子与标签合并成完整的训练数据集。

        Args:
            factor_panel: 因子面板 (MultiIndex: date x symbol)
            data_dict: {symbol: OHLCV DataFrame} 字典

        Returns:
            DataFrame (index=date x symbol) 含所有因子 + label + forward_return
        """
        all_labels = []
        for sym, df in data_dict.items():
            label = self.build_label_classification(df) \
                if self.model_type == "classifier" else self.build_label(df)
            label_df = label.reset_index()
            label_df.columns = ["日期", "label" if self.model_type == "classifier"
                                else "forward_return"]
            label_df["symbol"] = sym
            all_labels.append(label_df)

        labels_panel = pd.concat(all_labels, ignore_index=True)
        labels_panel["日期"] = pd.to_datetime(labels_panel["日期"])

        # 确保 factor_panel 的 date 列是 datetime
        factor_flat = factor_panel.reset_index()
        if "date" in factor_flat.columns:
            factor_flat["date"] = pd.to_datetime(factor_flat["date"])

        # 用 merge 替代 join，确保按 date + symbol 精确匹配
        dataset = factor_flat.merge(
            labels_panel,
            left_on=["date", "symbol"],
            right_on=["日期", "symbol"],
            how="inner",
        ).drop(columns=["日期"])

        logger.info(f"完整数据集: {len(dataset)} 条 "
                    f"({len(dataset.columns) - 3} 个特征 + 标签)")

        return dataset

    # ==================== 时序划分 ====================

    def split_timeseries(self, dataset: pd.DataFrame,
                         train_end: str, test_start: str) -> tuple:
        """按时序切分训练/测试集（不随机，严格时间先后）。

        Args:
            dataset: DataFrame with 'date' and 'symbol' columns (from merge_features_labels)
            train_end: 训练集截止日期 "YYYY-MM-DD"
            test_start: 测试集起始日期 "YYYY-MM-DD"

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        train_end_dt = pd.Timestamp(train_end)
        test_start_dt = pd.Timestamp(test_start)

        dates = pd.to_datetime(dataset["date"])

        train_mask = dates <= train_end_dt
        test_mask = dates >= test_start_dt

        feature_cols = [c for c in dataset.columns
                        if c not in ["label", "forward_return", "date", "symbol"]]
        label_col = "label" if self.model_type == "classifier" \
            else "forward_return"

        X_train = dataset.loc[train_mask, feature_cols]
        X_test = dataset.loc[test_mask, feature_cols]
        y_train = dataset.loc[train_mask, label_col]
        y_test = dataset.loc[test_mask, label_col]

        logger.info(f"训练集: {len(X_train)} 条 | 测试集: {len(X_test)} 条")

        return X_train, X_test, y_train, y_test

    def walk_forward_splits(self, dataset: pd.DataFrame,
                            train_start: str, train_end: str,
                            test_months: int = 6) -> list[tuple]:
        """生成 Walk-Forward 验证划分序列。

        每次迭代:
            - 训练：train_start → 当前 train_end
            - 验证：train_end 之后 test_months 个月
            - 下一次的 train_end 向后滚动 test_months 个月

        Args:
            dataset: 完整数据集
            train_start: 初始训练起点
            train_end: 首个训练终点
            test_months: 每次测试期长度（月）

        Returns:
            [(train_mask, test_mask), ...]
        """
        dates = sorted(dataset.index.get_level_values("date").unique())
        train_start_dt = pd.Timestamp(train_start)
        train_end_dt = pd.Timestamp(train_end)

        splits = []
        current_train_end = train_end_dt

        # 每次滚动 test_months
        while True:
            current_test_end = current_train_end + \
                pd.DateOffset(months=test_months)

            # 找最近的交易日
            train_dates = [d for d in dates
                           if train_start_dt <= d <= current_train_end]
            test_dates = [d for d in dates
                          if current_train_end < d <= current_test_end]

            if len(test_dates) < 20:  # 测试集太小就停
                break

            train_mask = dataset.index.get_level_values("date").isin(train_dates)
            test_mask = dataset.index.get_level_values("date").isin(test_dates)

            splits.append((train_mask, test_mask))

            current_train_end = current_test_end

            # 检查数据是否用完
            if current_train_end > dates[-1]:
                break

        logger.info(f"Walk-Forward 划分: {len(splits)} 折")
        return splits
