"""Univariate PC-type modeling pipeline with Lasso feature selection.

This module implements per-PC-type forecasting models using a three-stage feature
selection pipeline:
1. Variance threshold filtering to remove low-variance features
2. Stationarity filtering using Augmented Dickey-Fuller test
3. L1-regularized Lasso regression for final feature selection

Models are trained with time series cross-validation and logged to MLflow with
SHAP explainability artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import shap
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

# Constants
DEFAULT_VARIANCE_THRESHOLD = 1e-4
DEFAULT_STATIONARITY_ALPHA = 0.05
DEFAULT_TEST_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_SPLITS = 5
DEFAULT_MAX_ITER = 10000
DEFAULT_N_ALPHAS = 100
DEFAULT_TOL = 1e-3
MIN_SAMPLES_REQUIRED = 24
MIN_TRAIN_SAMPLES = 10
SHAP_BACKGROUND_LIMIT = 500
SHAP_SAMPLE_LIMIT = 200
LASSO_COEF_THRESHOLD = 1e-8

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
    """Variance threshold filter that preserves pandas DataFrame structure.

    Removes features with variance below the specified threshold while maintaining
    column names for downstream processing steps.
    """

    def __init__(self, threshold: float = DEFAULT_VARIANCE_THRESHOLD):
        """Initialize variance threshold filter.

        Args:
            threshold: Minimum variance required to keep a feature.
        """
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
    """Filter that keeps only stationary time series using Augmented Dickey-Fuller test.

    Features are retained if their ADF test p-value is below the specified alpha,
    indicating stationarity. This is important for time series forecasting models.
    """

    def __init__(
        self, alpha: float = DEFAULT_STATIONARITY_ALPHA, min_samples: int = 24
    ):
        """Initialize stationarity filter.

        Args:
            alpha: Significance level for ADF test (features with p-value <= alpha
                are considered stationary).
            min_samples: Minimum number of non-null samples required to test a feature.
        """
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


def _transform_through_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply variance, stationarity, and scaling steps to raw features.

    Transforms input DataFrame through the preprocessing steps of the pipeline
    (excluding the final Lasso estimator) to prepare data for SHAP explainer.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Raw feature DataFrame.

    Returns:
        Transformed feature array (numpy) ready for model input.
    """
    variance_step = pipeline.named_steps["variance"]
    stationarity_step = pipeline.named_steps["stationarity"]
    scaler_step = pipeline.named_steps["scaler"]

    X_var = variance_step.transform(X)
    X_stat = stationarity_step.transform(X_var)
    return scaler_step.transform(X_stat)


def _log_top_shap_metrics(
    feature_names: list[str], mean_abs_values: np.ndarray
) -> None:
    """Log mean absolute SHAP values for top 5 features to MLflow.

    Args:
        feature_names: List of feature names corresponding to SHAP values.
        mean_abs_values: Mean absolute SHAP values per feature.
    """
    if mean_abs_values.size == 0:
        return

    sorted_indices = np.argsort(mean_abs_values)[::-1]
    for rank, idx in enumerate(sorted_indices[:5], start=1):
        metric_name = f"shap_top{rank}_{feature_names[idx]}"
        mlflow.log_metric(metric_name, float(mean_abs_values[idx]))


def log_shap_artifacts(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_sample: pd.DataFrame,
    pc_type: cst.PCType,
    horizon: int,
) -> None:
    """Compute SHAP explanations and log plots/CSVs to MLflow.

    Generates SHAP values using LinearExplainer for the Lasso model and logs:
    - CSV file with SHAP values for all samples
    - Bar plot showing mean absolute SHAP values per feature
    - Top 5 feature importance metrics

    Args:
        pipeline: Fitted sklearn Pipeline containing the Lasso model.
        X_background: Background dataset for SHAP explainer (training data).
        X_sample: Sample dataset to explain (test data).
        pc_type: PC type identifier for artifact naming.
        horizon: Forecast horizon for artifact naming.
    """
    if X_sample is None or X_sample.empty:
        logger.warning(
            "Skipping SHAP logging for %s (horizon=%d): no sample data.",
            pc_type.value,
            horizon,
        )
        return

    feature_names = pipeline.named_steps["stationarity"].selected_features_
    if not feature_names:
        logger.warning(
            "Skipping SHAP logging for %s (horizon=%d): no stationary features.",
            pc_type.value,
            horizon,
        )
        return

    background_limit = min(len(X_background), SHAP_BACKGROUND_LIMIT)
    sample_limit = min(len(X_sample), SHAP_SAMPLE_LIMIT)
    if background_limit == 0 or sample_limit == 0:
        logger.warning(
            "Skipping SHAP logging for %s (horizon=%d): insufficient samples "
            "(background=%d, sample=%d).",
            pc_type.value,
            horizon,
            background_limit,
            sample_limit,
        )
        return

    background_df = X_background.tail(background_limit)
    sample_df = X_sample.tail(sample_limit)

    try:
        processed_background = _transform_through_pipeline(pipeline, background_df)
        processed_sample = _transform_through_pipeline(pipeline, sample_df)
        lasso_model = pipeline.named_steps["lasso"]
        explainer = shap.LinearExplainer(
            lasso_model,
            processed_background,
            feature_names=feature_names,
        )
        shap_values = explainer(processed_sample)
    except Exception as exc:
        logger.warning(
            "Failed to compute SHAP values for %s (horizon=%d): %s",
            pc_type.value,
            horizon,
            exc,
        )
        return

    shap_array = np.asarray(shap_values.values)
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    _log_top_shap_metrics(feature_names, mean_abs)

    sample_feature_df = pd.DataFrame(processed_sample, columns=feature_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        shap_csv = tmpdir_path / f"{pc_type.value}_{horizon}m_shap_values.csv"
        pd.DataFrame(shap_array, columns=feature_names).to_csv(shap_csv, index=False)
        mlflow.log_artifact(shap_csv, artifact_path="shap")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_array,
            features=sample_feature_df,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        summary_path = tmpdir_path / f"{pc_type.value}_{horizon}m_shap_summary.png"
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(summary_path, artifact_path="shap")


def load_or_build_univariate_dataset(
    horizon: int,
    group_by_pc_types: bool = True,
    include_exogenous: bool = True,
    include_differencing: bool = True,
) -> pd.DataFrame:
    """Load processed dataset from cache or rebuild via feature pipeline.

    Checks for cached processed dataset first. If not found, triggers the
    full feature engineering pipeline to build it from intermediate data.

    Args:
        horizon: Forecast horizon in months (affects dataset filename).
        group_by_pc_types: Whether to use grouped dataset (best prices per type).
        include_exogenous: Whether to include exogenous features.
        include_differencing: Whether to include differenced features.

    Returns:
        Processed DataFrame sorted by date with date column converted to datetime.
    """
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
    """Create supervised learning matrix for direct H-step ahead forecasting.

    Shifts the target column forward by the forecast horizon to create the
    prediction target, then prepares feature matrix with proper handling of
    missing values and infinite values.

    Args:
        df: Input DataFrame with time series data.
        target_col: Name of the target column to forecast.
        horizon: Number of steps ahead to forecast.

    Returns:
        Tuple of (feature matrix X, target vector y) with all-null columns
        removed and missing values forward/backward filled.
    """
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
    X: pd.DataFrame, y: pd.Series, test_ratio: float = DEFAULT_TEST_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform chronological train-test split preserving temporal ordering.

    Splits data such that all training samples precede all test samples in time,
    which is critical for time series forecasting evaluation.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_ratio: Fraction of samples to reserve for testing (default: 0.2).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If resulting training set has fewer than MIN_TRAIN_SAMPLES rows.
    """
    n_samples = len(y)
    test_size = max(1, int(np.floor(n_samples * test_ratio)))
    train_size = n_samples - test_size
    if train_size < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Not enough samples for the requested split. "
            f"Need at least {MIN_TRAIN_SAMPLES} training rows, received {train_size}."
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
    tol: float = 1e-3,
) -> Pipeline:
    """Create the end-to-end feature selection + Lasso pipeline.

    Args:
        variance_threshold: Minimum variance threshold for feature filtering.
        stationarity_alpha: Significance level for Augmented Dickey-Fuller test.
        n_splits: Number of time series cross-validation splits.
        random_state: Random seed for reproducibility.
        max_iter: Maximum iterations for Lasso optimization.
        n_alphas: Number of alpha values to test in LassoCV.
        tol: Convergence tolerance for Lasso optimization.

    Returns:
        Configured sklearn Pipeline with variance threshold, stationarity filter,
        standard scaler, and LassoCV estimator.
    """
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
                    alphas=n_alphas,
                    max_iter=max_iter,
                    tol=tol,
                ),
            ),
        ]
    )


def _compute_cv_splits(n_train_samples: int, desired_splits: int) -> int:
    """Compute appropriate number of CV splits given available training samples.

    Ensures TimeSeriesSplit has enough data points per fold while respecting
    the desired number of splits.

    Args:
        n_train_samples: Number of training samples available.
        desired_splits: Desired number of cross-validation splits.

    Returns:
        Actual number of splits to use (adjusted based on sample size).
    """
    if n_train_samples < 4:
        return 2
    max_possible = max(2, min(desired_splits, n_train_samples - 1))
    return max_possible


def train_single_pc_model(
    df: pd.DataFrame,
    pc_type: cst.PCType,
    horizon: int,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    stationarity_alpha: float = DEFAULT_STATIONARITY_ALPHA,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    desired_cv_splits: int = DEFAULT_CV_SPLITS,
    mlflow_run_name: str | None = None,
) -> PCModelResult:
    """Train a Lasso pipeline for a single PC type and horizon.

    Applies three-stage feature selection (variance threshold, stationarity filter,
    Lasso regularization) and logs model artifacts, metrics, and SHAP explanations
    to MLflow.

    Args:
        df: Processed dataset with features and target column.
        pc_type: PC type to train model for.
        horizon: Forecast horizon in months.
        variance_threshold: Minimum variance for feature filtering.
        stationarity_alpha: ADF test significance level.
        test_ratio: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        desired_cv_splits: Maximum number of CV splits for Lasso.
        mlflow_run_name: Optional custom name for MLflow run.

    Returns:
        PCModelResult containing training diagnostics and selected features.

    Raises:
        ValueError: If PC type is unsupported, target column missing, or
            insufficient samples available.
    """
    if pc_type not in PC_TARGET_COLUMN_MAP:
        raise ValueError(f"Unsupported pc_type: {pc_type}")
    target_col = PC_TARGET_COLUMN_MAP[pc_type]
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not available in the dataset. "
            "Rebuild processed data with group_by_pc_types=True."
        )

    X, y = make_supervised_matrix(df=df, target_col=target_col, horizon=horizon)
    if len(X) < MIN_SAMPLES_REQUIRED:
        raise ValueError(
            f"Not enough samples ({len(X)}) for PC type {pc_type.value} "
            f"horizon {horizon}. Minimum required: {MIN_SAMPLES_REQUIRED}."
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
        selected_mask = np.abs(final_coefs) > LASSO_COEF_THRESHOLD
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
        log_shap_artifacts(
            pipeline=pipeline,
            X_background=X_train,
            X_sample=X_test,
            pc_type=pc_type,
            horizon=horizon,
        )
        # Prepare input example for model signature inference
        if len(X_test) > 0:
            input_example = X_test.iloc[:1].copy()
        else:
            input_example = X_train.iloc[:1].copy()
        mlflow.sklearn.log_model(pipeline, name="model", input_example=input_example)

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
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    stationarity_alpha: float = DEFAULT_STATIONARITY_ALPHA,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    desired_cv_splits: int = DEFAULT_CV_SPLITS,
    mlflow_experiment: str | None = None,
) -> list[PCModelResult]:
    """Train Lasso models for each requested PC type with MLflow logging.

    Trains individual models for each PC type and logs them as nested child runs
    under a parent run. Aggregates metrics (global MAPE, weighted MAPE, regular
    MAPE, green MAPE) in the parent run.

    Args:
        horizon: Forecast horizon in months.
        pc_types: List of PC types to train. If None, trains all available types.
        group_by_pc_types: Whether to use grouped dataset (best prices per type).
        variance_threshold: Minimum variance for feature filtering.
        stationarity_alpha: ADF test significance level.
        test_ratio: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        desired_cv_splits: Maximum number of CV splits for Lasso.
        mlflow_experiment: MLflow experiment name/path.

    Returns:
        List of PCModelResult objects, one per successfully trained PC type.
    """
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
        default=DEFAULT_VARIANCE_THRESHOLD,
        help="Variance threshold for feature pruning.",
    )
    parser.add_argument(
        "--stationarity-alpha",
        type=float,
        default=DEFAULT_STATIONARITY_ALPHA,
        help="ADF p-value threshold for stationarity.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of samples reserved for hold-out testing.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=DEFAULT_CV_SPLITS,
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
