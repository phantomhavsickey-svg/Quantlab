"""
LightGBM 训练器 — 模型训练、超参调优、特征重要性。
"""

import os
import pandas as pd
import numpy as np
from loguru import logger

try:
    import lightgbm as lgb
except ImportError:
    logger.error("请先安装 lightgbm: pip install lightgbm")
    raise

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna 未安装，超参调优不可用")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class LightGBMTrainer:
    """LightGBM 模型训练器。

    支持:
        - 分类器（涨/跌）: objective='binary', metric='auc'
        - 回归器（收益率）: objective='regression', metric='rmse'
        - Optuna 贝叶斯超参优化
        - SHAP 特征重要性
    """

    DEFAULT_PARAMS = {
        "classifier": {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "random_state": 42,
        },
        "regressor": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "random_state": 42,
        },
    }

    def __init__(self, model_type: str = "classifier",
                 params: dict | None = None):
        """
        Args:
            model_type: "classifier" / "regressor"
            params: 覆盖默认参数的配置
        """
        self.model_type = model_type
        self.params = self.DEFAULT_PARAMS[model_type].copy()
        if params:
            self.params.update(params)
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []

    # ==================== 训练 ====================

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_valid: pd.DataFrame | None = None,
              y_valid: pd.Series | None = None,
              num_boost_round: int = 500,
              early_stopping_rounds: int = 50) -> dict:
        """训练 LightGBM 模型。

        Args:
            X_train, y_train: 训练数据
            X_valid, y_valid: 验证数据（可选，用于 early stopping）
            num_boost_round: 最大迭代轮数
            early_stopping_rounds: 提前停止轮数

        Returns:
            训练结果 dict (含 best_iteration, best_score 等)
        """
        self.feature_names = list(X_train.columns)

        # 处理 NaN（LightGBM 原生支持，但保险起见）
        X_train = X_train.fillna(np.nan)
        if X_valid is not None:
            X_valid = X_valid.fillna(np.nan)

        train_data = lgb.Dataset(X_train, label=y_train,
                                  feature_name=self.feature_names)

        valid_sets = None
        valid_names = None
        callbacks = []

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid,
                                      feature_name=self.feature_names,
                                      reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ["train", "valid"]
            callbacks = [
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(50),
            ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # best_score 可能是 defaultdict（多验证集）或单个值
        best_score = self.model.best_score
        if isinstance(best_score, dict) or hasattr(best_score, "get"):
            # 尝试从嵌套字典中取 valid 的 metric
            try:
                valid_scores = best_score.get("valid", {})
                metric = list(valid_scores.keys())[0] if valid_scores else "metric"
                best_score_val = valid_scores.get(metric, 0.0)
            except (AttributeError, IndexError):
                best_score_val = 0.0
        else:
            best_score_val = float(best_score)

        result = {
            "best_iteration": self.model.best_iteration,
            "best_score": best_score_val,
            "feature_names": self.feature_names,
        }

        logger.info(f"训练完成: best_iteration={result['best_iteration']}, "
                     f"best_score={result['best_score']:.4f}")

        return result

    # ==================== 超参调优 ====================

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_valid: pd.DataFrame, y_valid: pd.Series,
             n_trials: int = 50) -> dict:
        """用 Optuna 做贝叶斯超参优化。

        Returns:
            最优参数字典
        """
        if not HAS_OPTUNA:
            logger.warning("optuna 未安装，跳过超参调优")
            return self.params

        X_train_filled = X_train.fillna(np.nan)
        X_valid_filled = X_valid.fillna(np.nan)
        feature_names = list(X_train.columns)

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate",
                                                      0.01, 0.3, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",
                                                       10, 200),
                "feature_fraction": trial.suggest_float("feature_fraction",
                                                         0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction",
                                                         0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1",
                                                  1e-4, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2",
                                                  1e-4, 10.0, log=True),
            }
            params.update({
                k: v for k, v in self.DEFAULT_PARAMS[self.model_type].items()
                if k not in params
            })

            train_data = lgb.Dataset(X_train_filled,
                                      label=y_train,
                                      feature_name=feature_names)
            valid_data = lgb.Dataset(X_valid_filled,
                                      label=y_valid,
                                      feature_name=feature_names,
                                      reference=train_data)

            model = lgb.train(
                params, train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )

            if self.model_type == "classifier":
                return model.best_score["valid"]["auc"]
            else:
                return -model.best_score["valid"]["rmse"]  # Optuna 最小化

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # 合并最优参数
        self.params.update(study.best_params)
        logger.info(f"最优参数: {study.best_params}")
        logger.info(f"最优分数: {study.best_value:.4f}")

        return self.params

    # ==================== 特征重要性 ====================

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """获取特征重要性。

        Args:
            importance_type: "gain" (信息增益) 或 "split" (分裂次数)

        Returns:
            DataFrame (index=特征名, columns=[importance])
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        importance = self.model.feature_importance(importance_type=importance_type)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return df

    def shap_values(self, X: pd.DataFrame) -> np.ndarray | None:
        """计算 SHAP 值。

        Returns:
            SHAP values array, shape (n_samples, n_features)
        """
        if not HAS_SHAP:
            logger.warning("shap 未安装")
            return None
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")

        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(X.fillna(np.nan))
        return shap_vals

    # ==================== 保存/加载 ====================

    def save(self, path: str):
        """保存模型。"""
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用 train()")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        self.model = lgb.Booster(model_file=path)
        self.feature_names = self.model.feature_name()
        logger.info(f"模型已加载: {path}")

    # ==================== Walk-Forward 训练 ====================

    def walk_forward_train(self, dataset: pd.DataFrame,
                           splits: list[tuple],
                           tune: bool = False) -> list[dict]:
        """Walk-Forward 滚动训练。

        Args:
            dataset: 完整数据集 (MultiIndex: date x symbol)
            splits: walk_forward_splits 返回的划分列表
            tune: 是否调参

        Returns:
            [{fold, train_start, train_end, test_start, test_end, metrics}, ...]
        """
        results = []
        feature_cols = [c for c in dataset.columns
                        if c not in ["label", "forward_return"]]
        label_col = "label" if self.model_type == "classifier" \
            else "forward_return"

        dates = sorted(dataset.index.get_level_values("date").unique())

        for i, (train_mask, test_mask) in enumerate(splits):
            train_dates = dataset.loc[train_mask].index.get_level_values("date")
            test_dates = dataset.loc[test_mask].index.get_level_values("date")

            X_train = dataset.loc[train_mask, feature_cols]
            y_train = dataset.loc[train_mask, label_col]
            X_test = dataset.loc[test_mask, feature_cols]
            y_test = dataset.loc[test_mask, label_col]

            if tune and HAS_OPTUNA:
                # 从训练集中拆出最后3个月做验证
                mid = int(len(X_train) * 0.8)
                self.tune(X_train.iloc[:mid], y_train.iloc[:mid],
                          X_train.iloc[mid:], y_train.iloc[mid:],
                          n_trials=30)

            result = self.train(X_train, y_train, X_test, y_test)

            fold_result = {
                "fold": i + 1,
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "test_start": test_dates.min(),
                "test_end": test_dates.max(),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "best_score": result.get("best_score"),
                "best_iteration": result.get("best_iteration"),
            }

            results.append(fold_result)
            logger.info(f"Fold {i+1}: "
                        f"{fold_result['train_start'].strftime('%Y-%m')} → "
                        f"{fold_result['test_end'].strftime('%Y-%m')} | "
                        f"score={fold_result['best_score']:.4f}")

        return results
