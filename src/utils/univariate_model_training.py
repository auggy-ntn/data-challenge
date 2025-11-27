"""Univariate PC-type modeling pipeline backed by Lasso feature selection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import mlflow
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from constants import constants as cst
from constants import intermediate_names, processed_names
from constants.paths import PROCESSED_DATA_DIR
from src.data_pipelines.uni_intermediate_to_processed import build_univariate_dataset
from src.utils.logger import logger

PC_TARGET_COLUMN_MAP = {
    cst.PCType.CRYSTAL: (
        f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_CRYSTAL_BEST_PRICE}"
    ),
    cst.PCType.WHITE: (
        f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_WHITE_BEST_PRICE}"
    ),
    cst.PCType.GF10: (f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_GF10_BEST_PRICE}"),
    cst.PCType.GF20: (f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_GF20_BEST_PRICE}"),
    cst.PCType.RECYCLED_GF10_WHITE: (
        f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE}"
    ),
    cst.PCType.RECYCLED_GF10_GREY: (
        f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE}"
    ),
    cst.PCType.SI: (f"{cst.EU_PREFIX}{intermediate_names.PC_EU_PC_SI_BEST_PRICE}"),
}


@dataclass
class PCModelResult:
    """Container for per-PC model diagnostics."""

    pc_type: cst.PCType
    horizon: int
    n_test_samples: int
    n_features_before: int
    n_features_after_stationarity: int
    n_lasso_features: int
    train_mape: float
    test_mape: float
    selected_features: list[str]


class PandasVarianceThreshold(BaseEstimator, TransformerMixin):
    """Variance threshold that keeps pandas column names intact."""

    def __init__(self, threshold: float = 1e-4):
        """Initialize filter with the desired variance threshold."""
        self.threshold = threshold
        self.selected_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Identify columns whose variance exceeds the threshold."""
        X_df = self._ensure_dataframe(X)
        variances = X_df.var(skipna=True)
        self.selected_features_ = variances[variances > self.threshold].index.tolist()
        if not self.selected_features_:
            logger.warning(
                "Variance threshold removed all features; reverting to original set."
            )
            self.selected_features_ = X_df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        """Return the subset of columns kept during `fit`."""
        X_df = self._ensure_dataframe(X)
        return X_df[self.selected_features_].copy()

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("PandasVarianceThreshold expects a pandas DataFrame input.")


class StationarityFilter(BaseEstimator, TransformerMixin):
    """Filter that keeps only stationary series based on Augmented Dickey-Fuller."""

    def __init__(self, alpha: float = 0.05, min_samples: int = 24):
        """Initialize filter with ADF significance level and min samples."""
        self.alpha = alpha
        self.min_samples = min_samples
        self.selected_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Keep columns that pass the Augmented Dickey-Fuller test."""
        X_df = self._ensure_dataframe(X)
        stationary_features = []
        dropped_features = []
        for col in X_df.columns:
            series = X_df[col].dropna()
            if len(series) < self.min_samples:
                dropped_features.append(col)
                continue
            try:
                p_value = adfuller(series, autolag="AIC")[1]
            except (ValueError, LinAlgError):
                dropped_features.append(col)
                continue

            if p_value <= self.alpha:
                stationary_features.append(col)
            else:
                dropped_features.append(col)

        if not stationary_features:
            logger.warning(
                "Stationarity filter removed all features; retaining pre-filtered set."
            )
            stationary_features = X_df.columns.tolist()

        self.selected_features_ = stationary_features
        logger.info(
            "Stationarity filter kept %d / %d features (alpha=%.3f).",
            len(self.selected_features_),
            X_df.shape[1],
            self.alpha,
        )
        if dropped_features:
            logger.debug("Stationarity filter dropped: %s", dropped_features)
        return self

    def transform(self, X: pd.DataFrame):
        """Return stationary subset of features retained in `fit`."""
        X_df = self._ensure_dataframe(X)
        return X_df[self.selected_features_].copy()

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("StationarityFilter expects a pandas DataFrame input.")


def _dataset_path(horizon: int, grouped: bool) -> Path:
    suffix = "_grouped" if grouped else ""
    filename = f"uni_{horizon}m{suffix}.csv"
    return PROCESSED_DATA_DIR / filename


def load_or_build_univariate_dataset(
    horizon: int,
    group_by_pc_types: bool = True,
    include_exogenous: bool = True,
    include_differencing: bool = True,
) -> pd.DataFrame:
    """Load processed dataset if present, otherwise rebuild from intermediates."""
    path = _dataset_path(horizon, grouped=group_by_pc_types)
    if path.exists():
        logger.info("Loading cached univariate dataset from %s", path)
        df = pd.read_csv(path)
    else:
        logger.info("Processed dataset not found, rebuilding via feature pipeline.")
        path.parent.mkdir(parents=True, exist_ok=True)
        df = build_univariate_dataset(
            horizon=horizon,
            group_by_pc_types=group_by_pc_types,
            include_exogenous=include_exogenous,
            include_differencing=include_differencing,
        )
        df.to_csv(path, index=False)
        logger.success("Wrote processed dataset to %s", path)
    if processed_names.WIDE_DATE in df.columns:
        df[processed_names.WIDE_DATE] = pd.to_datetime(
            df[processed_names.WIDE_DATE], errors="coerce"
        )
    return df.sort_values(by=processed_names.WIDE_DATE).reset_index(drop=True)


def make_supervised_matrix(
    df: pd.DataFrame, target_col: str, horizon: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised learning matrix for direct H-step forecasting."""
    dataset = df.copy()
    target_name = f"{target_col}_target_h{horizon}"
    dataset[target_name] = dataset[target_col].shift(-horizon)
    dataset = dataset.dropna(subset=[target_name])
    numeric_df = dataset.select_dtypes(include=[np.number]).copy()
    if target_name not in numeric_df.columns:
        numeric_df[target_name] = dataset[target_name]
    y = numeric_df[target_name]
    X = numeric_df.drop(columns=[target_name])
    X = X.dropna(axis=1, how="all")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def temporal_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Simple chronological split preserving ordering."""
    n_samples = len(y)
    test_size = max(1, int(np.floor(n_samples * test_ratio)))
    train_size = n_samples - test_size
    if train_size < 10:
        raise ValueError(
            "Not enough samples for the requested split. "
            f"Need at least 10 training rows, received {train_size}."
        )
    X_train = X.iloc[:train_size].copy()
    y_train = y.iloc[:train_size].copy()
    X_test = X.iloc[train_size:].copy()
    y_test = y.iloc[train_size:].copy()
    return X_train, X_test, y_train, y_test


def build_lasso_pipeline(
    variance_threshold: float,
    stationarity_alpha: float,
    n_splits: int,
    random_state: int = 42,
    max_iter: int = 10000,
    n_alphas: int = 100,
) -> Pipeline:
    """Create the end-to-end feature selection + Lasso pipeline."""
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    return Pipeline(
        steps=[
            ("variance", PandasVarianceThreshold(threshold=variance_threshold)),
            ("stationarity", StationarityFilter(alpha=stationarity_alpha)),
            ("scaler", StandardScaler(copy=True)),
            (
                "lasso",
                LassoCV(
                    cv=ts_cv,
                    random_state=random_state,
                    alphas=n_alphas,  # Use alphas instead of deprecated n_alphas
                    max_iter=max_iter,
                    tol=1e-3,  # Relaxed tolerance for better convergence
                ),
            ),
        ]
    )


def _compute_cv_splits(n_train_samples: int, desired_splits: int) -> int:
    """Ensure TimeSeriesSplit has enough folds for the available rows."""
    if n_train_samples < 4:
        return 2
    max_possible = max(2, min(desired_splits, n_train_samples - 1))
    return max_possible


def train_single_pc_model(
    df: pd.DataFrame,
    pc_type: cst.PCType,
    horizon: int,
    variance_threshold: float = 1e-4,
    stationarity_alpha: float = 0.05,
    test_ratio: float = 0.2,
    random_state: int = 42,
    desired_cv_splits: int = 5,
    mlflow_run_name: str | None = None,
) -> PCModelResult:
    """Train a Lasso pipeline for a single PC type and horizon."""
    if pc_type not in PC_TARGET_COLUMN_MAP:
        raise ValueError(f"Unsupported pc_type: {pc_type}")
    target_col = PC_TARGET_COLUMN_MAP[pc_type]
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not available in the dataset. "
            "Rebuild processed data with group_by_pc_types=True."
        )

    X, y = make_supervised_matrix(df=df, target_col=target_col, horizon=horizon)
    if len(X) < 24:
        raise ValueError(
            f"Not enough samples ({len(X)}) for PC type {pc_type.value} "
            f"horizon {horizon}."
        )
    n_features_before = X.shape[1]
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        X=X, y=y, test_ratio=test_ratio
    )
    n_splits = _compute_cv_splits(len(X_train), desired_splits=desired_cv_splits)
    pipeline = build_lasso_pipeline(
        variance_threshold=variance_threshold,
        stationarity_alpha=stationarity_alpha,
        n_splits=n_splits,
        random_state=random_state,
    )

    run_name = mlflow_run_name or f"lasso_{pc_type.value}_{horizon}m"
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.set_tags(
            {
                cst.MLFLOW_MODEL_PHILOSOPHY: "univariate",
                cst.MLFLOW_MODEL_TYPE: "lasso",
                cst.MLFLOW_HORIZON: horizon,
                cst.MLFLOW_FUNCTION: "evaluation",
                "pc_type": pc_type.value,
            }
        )
        mlflow.log_params(
            {
                "variance_threshold": variance_threshold,
                "stationarity_alpha": stationarity_alpha,
                "test_ratio": test_ratio,
                "n_features_before": n_features_before,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "cv_splits": n_splits,
            }
        )

        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        train_mape = mean_absolute_percentage_error(y_train, train_pred)
        test_mape = mean_absolute_percentage_error(y_test, test_pred)

        stationarity_features = pipeline.named_steps["stationarity"].selected_features_
        lasso_model = pipeline.named_steps["lasso"]
        final_coefs = lasso_model.coef_
        selected_alpha = lasso_model.alpha_
        selected_mask = np.abs(final_coefs) > 1e-8
        selected_features = [
            feature
            for feature, keep in zip(stationarity_features, selected_mask, strict=True)
            if keep
        ]

        if len(selected_features) == 0:
            logger.warning(
                "Lasso selected zero features for %s (horizon=%d). "
                "Selected alpha=%.6f, intercept=%.6f. "
                "Model will predict constant (intercept only). "
                "This may indicate over-regularization or weak signal. "
                "Consider relaxing stationarity_alpha or checking data quality.",
                pc_type.value,
                horizon,
                selected_alpha,
                lasso_model.intercept_,
            )
            # Log additional diagnostics
            mlflow.log_param("zero_features_warning", True)
            mlflow.log_metric(
                "n_stationary_features_when_zero", len(stationarity_features)
            )

        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("n_stationary_features", len(stationarity_features))
        mlflow.log_metric("n_lasso_features", len(selected_features))
        mlflow.log_metric("lasso_alpha", selected_alpha)
        mlflow.log_metric("lasso_intercept", lasso_model.intercept_)

        feature_info = {
            "pc_type": pc_type.value,
            "horizon": horizon,
            "target_column": target_col,
            "features_before": n_features_before,
            "features_after_stationarity": len(stationarity_features),
            "selected_by_lasso": selected_features,
            "lasso_alpha": float(selected_alpha),
            "lasso_intercept": float(lasso_model.intercept_),
        }
        mlflow.log_text(
            json.dumps(feature_info, indent=2),
            artifact_file=f"{pc_type.value}_feature_summary.json",
        )
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return PCModelResult(
        pc_type=pc_type,
        horizon=horizon,
        n_test_samples=len(X_test),
        n_features_before=n_features_before,
        n_features_after_stationarity=len(stationarity_features),
        n_lasso_features=len(selected_features),
        train_mape=train_mape,
        test_mape=test_mape,
        selected_features=selected_features,
    )


def train_per_pc_models(
    horizon: int,
    pc_types: Sequence[cst.PCType] | None = None,
    group_by_pc_types: bool = True,
    variance_threshold: float = 1e-4,
    stationarity_alpha: float = 0.05,
    test_ratio: float = 0.2,
    random_state: int = 42,
    desired_cv_splits: int = 5,
    mlflow_experiment: str | None = None,
) -> list[PCModelResult]:
    """Train Lasso models for each requested PC type."""
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
    df = load_or_build_univariate_dataset(
        horizon=horizon,
        group_by_pc_types=group_by_pc_types,
        include_exogenous=True,
        include_differencing=True,
    )
    targets = pc_types or list(PC_TARGET_COLUMN_MAP.keys())
    results: list[PCModelResult] = []
    per_pc_mape: dict[cst.PCType, tuple[float, int]] = {}
    parent_nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"lasso_{horizon}m_global", nested=parent_nested):
        mlflow.set_tags(
            {
                cst.MLFLOW_MODEL_PHILOSOPHY: "univariate",
                cst.MLFLOW_MODEL_TYPE: "lasso",
                cst.MLFLOW_HORIZON: horizon,
                cst.MLFLOW_FUNCTION: "batch_training",
            }
        )
        for pc_type in targets:
            logger.info(
                "Training %s model for horizon %s months",
                pc_type.value,
                horizon,
            )
            try:
                result = train_single_pc_model(
                    df=df,
                    pc_type=pc_type,
                    horizon=horizon,
                    variance_threshold=variance_threshold,
                    stationarity_alpha=stationarity_alpha,
                    test_ratio=test_ratio,
                    random_state=random_state,
                    desired_cv_splits=desired_cv_splits,
                )
            except ValueError as exc:
                if "Not enough samples" in str(exc):
                    logger.warning("Skipping %s: %s", pc_type.value, exc)
                    continue
                raise
            results.append(result)
            per_pc_mape[pc_type] = (result.test_mape, result.n_test_samples)
            logger.success(
                "PC=%s | horizon=%s | test MAPE=%.3f | features=%d",
                pc_type.value,
                horizon,
                result.test_mape,
                result.n_lasso_features,
            )
        if per_pc_mape:
            mapes = np.array([value for value, _ in per_pc_mape.values()])
            weights = np.array([weight for _, weight in per_pc_mape.values()])
            overall_mape = float(np.mean(mapes))
            weighted_mape = float(np.average(mapes, weights=weights))

            regular_types = {
                cst.PCType.CRYSTAL,
                cst.PCType.WHITE,
                cst.PCType.GF10,
                cst.PCType.GF20,
                cst.PCType.SI,
            }
            green_types = {
                cst.PCType.RECYCLED_GF10_WHITE,
                cst.PCType.RECYCLED_GF10_GREY,
            }

            regular_mapes = [
                per_pc_mape[pc][0] for pc in regular_types if pc in per_pc_mape
            ]
            green_mapes = [
                per_pc_mape[pc][0] for pc in green_types if pc in per_pc_mape
            ]

            if regular_mapes:
                mlflow.log_metric("regular_MAPE", float(np.mean(regular_mapes)))
            if green_mapes:
                mlflow.log_metric("green_MAPE", float(np.mean(green_mapes)))

            mlflow.log_metric(cst.GLOBAL_MAPE, overall_mape)
            mlflow.log_metric(cst.WEIGHTED_MAPE, weighted_mape)
    return results


def _parse_pc_types(pc_types_arg: str | None) -> list[cst.PCType] | None:
    if not pc_types_arg:
        return None
    requested = []
    for value in pc_types_arg.split(","):
        value = value.strip().lower()
        try:
            requested.append(cst.PCType(value))
        except ValueError as exc:
            raise ValueError(f"Unknown PC type '{value}'.") from exc
    return requested


def main():
    """CLI entrypoint for ad-hoc training via `python -m` execution."""
    parser = argparse.ArgumentParser(
        description="Train per-PC univariate Lasso models with feature selection."
    )
    parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon.")
    parser.add_argument(
        "--pc-types",
        type=str,
        default=None,
        help="Comma-separated PC types (values from constants.PCType).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-4,
        help="Variance threshold for feature pruning.",
    )
    parser.add_argument(
        "--stationarity-alpha",
        type=float,
        default=0.05,
        help="ADF p-value threshold for stationarity.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for hold-out testing.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Maximum number of TimeSeriesSplit folds.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="Optional MLflow experiment name.",
    )
    parser.add_argument(
        "--no-grouping",
        action="store_true",
        help="Use the base dataset (without PC-type grouping).",
    )
    args = parser.parse_args()

    pc_types = _parse_pc_types(args.pc_types)
    train_per_pc_models(
        horizon=args.horizon,
        pc_types=pc_types,
        group_by_pc_types=not args.no_grouping,
        variance_threshold=args.variance_threshold,
        stationarity_alpha=args.stationarity_alpha,
        test_ratio=args.test_ratio,
        desired_cv_splits=args.cv_splits,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == "__main__":
    main()
