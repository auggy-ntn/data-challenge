"""Base multivariate model training functions."""

from typing import Literal

import mlflow
from mlflow.models import infer_signature
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from constants import processed_names
import constants.constants as cst
from src.utils.evaluation import compute_performance_metrics


def compute_sample_weights(
    df: pd.DataFrame,
    group_col: str,
    region_col: str,
    target_region: str,
    method: Literal["inverse_frequency", "sqrt_inverse", "balanced"] = "balanced",
    region_weight_multiplier: float = 1.5,
) -> pd.Series:
    """Compute sample weights based on group frequency and region.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        group_col (str): Column name for grouping (e.g., pc_type).
        region_col (str): Column name for region.
        target_region (str): Region to prioritize.
        method (Literal["inverse_frequency", "sqrt_inverse", "balanced"], optional):
            Method to compute weights. Defaults to "balanced".
        region_weight_multiplier (float, optional): Multiplier for weights of the
            target region. Defaults to 1.5.

    Returns:
        pd.Series: Sample weights for each row in the dataframe.
    """
    # Validate method
    if method not in ["inverse_frequency", "sqrt_inverse", "balanced"]:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'inverse_frequency', "
            "'sqrt_inverse', 'balanced'."
        )

    group_counts = df[group_col].value_counts()
    n_samples = len(df)
    n_groups = len(group_counts)

    # Simple inverse frequency weights (1/count)
    if method == "inverse_frequency":
        weights_map = {group: 1.0 / count for group, count in group_counts.items()}
        weights = df[group_col].map(weights_map).values

    # Square root of inverse frequency weights
    # Less aggressive than inverse_frequency
    elif method == "sqrt_inverse":
        weights_map = {
            group: 1.0 / np.sqrt(count) for group, count in group_counts.items()
        }
        weights = df[group_col].map(weights_map).values

    elif method == "balanced":
        # Sklearn-style balanced weights: n_samples / (n_groups * count)
        # Normalized so sum(weights) ≈ n_samples (maintains effective sample size)
        weights_map = {
            group: n_samples / (n_groups * count)
            for group, count in group_counts.items()
        }
        weights = df[group_col].map(weights_map).values

        # Apply region multiplier
    region_multiplier = np.where(
        df[region_col] == target_region,
        region_weight_multiplier,  # 1.5× weight for Europe
        1.0,  # 1.0× weight for Asia
    )
    weights = weights * region_multiplier

    # Renormalize to maintain effective sample size
    weights = weights * len(df) / weights.sum()

    return weights


def optimize_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights: pd.Series,
    hyperparameter_grid: dict | None,
    pc_types: pd.Series,
    n_trials: int = 50,
    **kwargs,
) -> dict:
    """Optimize an XGBoost model with Optuna.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_test (np.ndarray): Testing feature matrix.
        y_test (np.ndarray): Testing target vector.
        weights (pd.Series): Sample weights for the test set.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
            If None, a default search space is used.
        pc_types (pd.Series): PC types corresponding to the test set.
        n_trials (int, optional): Number of Optuna trials. Defaults to 50.
        **kwargs: Additional keyword arguments for XGBRegressor.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    def objective(trial):
        """Internal objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): Optuna trial object for hyperparameter suggestions.

        Returns:
            float: Weighted MAPE metric for the trial.
        """
        # Use provided grid or create default search space
        params = (
            hyperparameter_grid
            if hyperparameter_grid is not None
            else {
                # Tree structure
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                # Learning rate
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                # Regularization
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                # Sampling
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                # Tree method and other settings
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                # Fixed settings for stability
                "tree_method": "hist",  # Faster and often better than 'auto'
            }
        )
        model = XGBRegressor(**params, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        weighted_mape = compute_performance_metrics(
            y_test, y_pred, pc_types=pc_types, weights=weights
        ).get(cst.WEIGHTED_MAPE, None)
        return weighted_mape

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params


def log_model(
    model_type: Literal["xgboost", "random_forest", "lightgbm", "catboost", "tft"],
    model_role: Literal["evaluation", "prediction"],
    model: object,
    signature: mlflow.models.signature.ModelSignature,
) -> None:
    """Log a trained model to MLflow with the appropriate method.

    Args:
        model_type (Literal["xgboost", "random_forest", "lightgbm", "catboost", "tft"]):
            Type of the model.
        model_role (Literal["evaluation", "prediction"]): Role of the model.
        model (object): Trained model to log.
        signature (mlflow.models.signature.ModelSignature): Model signature for logging.
    """
    if model_type == "xgboost":
        mlflow.xgboost.log_model(model, f"{model_role}_model", signature=signature)
    elif model_type == "lightgbm":
        mlflow.lightgbm.log_model(model, f"{model_role}_model", signature=signature)
    elif model_type == "random_forest":
        mlflow.sklearn.log_model(model, f"{model_role}_model", signature=signature)
    elif model_type == "catboost":
        mlflow.catboost.log_model(model, f"{model_role}_model", signature=signature)
    elif model_type == "tft":
        mlflow.pytorch.log_model(model, f"{model_role}_model", signature=signature)
    else:
        raise ValueError(f"Unsupported model type for logging: {model_type}")


def evaluate_and_log_model(
    eval_model,
    pred_model,
    best_params: dict,
    mlflow_run_name: str,
    model_type: Literal["xgboost", "random_forest", "lightgbm", "catboost", "tft"],
    horizon: int,
    group_by_pc_types: bool,
    weighting_method: Literal["inverse_frequency", "sqrt_inverse", "balanced"],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_df_aligned: pd.DataFrame,
    validation_df_aligned: pd.DataFrame | None,
    test_df_aligned: pd.DataFrame,
    train_sample_weights: pd.Series,
    test_sample_weights: pd.Series,
):
    """Evaluate and log the performance of a trained model.

    Args:
        eval_model: Trained model to evaluate.
        pred_model: Trained model for prediction. (Retrained on full data)
        best_params (dict): Best hyperparameters for the model.
        mlflow_run_name (str): Name for the MLflow run.
        model_type (Literal["xgboost", "random_forest", "lightgbm", "tft"]): Type of the
            model.
        horizon (int): Forecast horizon in months.
        group_by_pc_types (bool): Whether data is grouped by PC types.
        weighting_method (Literal["inverse_frequency", "sqrt_inverse", "balanced"]):
            Method used for sample weighting.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_validation (np.ndarray | None): Validation feature matrix.
        y_validation (np.ndarray | None): Validation target vector.
        X_test (np.ndarray): Testing feature matrix.
        y_test (np.ndarray): Testing target vector.
        train_df_aligned (pd.DataFrame): Aligned training dataframe.
        validation_df_aligned (pd.DataFrame | None): Aligned validation dataframe.
        test_df_aligned (pd.DataFrame): Aligned testing dataframe.
        train_sample_weights (pd.Series): Sample weights for training data.
        test_sample_weights (pd.Series): Sample weights for testing data.
    """
    with mlflow.start_run(run_name=f"{mlflow_run_name}_eval"):
        mlflow.set_tags(
            {
                cst.MLFLOW_MODEL_PHILOSOPHY: "multivariate",
                cst.MLFLOW_MODEL_TYPE: model_type,
                cst.MLFLOW_HORIZON: horizon,
                cst.MLFLOW_FUNCTION: "evaluation",
            }
        )
        mlflow.log_params(best_params)
        # Train evaluation model with TRAIN weights
        if group_by_pc_types:
            eval_model.fit(X_train, y_train, sample_weight=train_sample_weights)
        else:
            # Combine TRAIN and VALIDATION for evaluating model on TEST set
            X_train = np.vstack([X_train, X_validation])
            y_train = np.concatenate([y_train, y_validation])
            train_df_aligned = pd.concat(
                [train_df_aligned, validation_df_aligned], ignore_index=True
            )
            train_sample_weights = compute_sample_weights(
                df=train_df_aligned,
                group_col=processed_names.LONG_PC_TYPE,
                region_col=processed_names.LONG_REGION,
                target_region=cst.EUROPE,
                method=weighting_method,
            )
            eval_model.fit(X_train, y_train, sample_weight=train_sample_weights)

        # Evaluate model with TEST weights
        performance_metrics = compute_performance_metrics(
            y_true=y_test,
            y_pred=eval_model.predict(X_test),
            pc_types=test_df_aligned[processed_names.LONG_PC_TYPE],
            weights=test_sample_weights,
        )
        for metric_name, metric_value in performance_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model with signature
        signature = infer_signature(X_train, eval_model.predict(X_train))
        log_model(
            model_type=model_type,
            model_role="evaluation",
            model=eval_model,
            signature=signature,
        )

        # For prediction, retrain on full training data
        X_full_train = np.vstack([X_train, X_test])
        y_full_train = np.concatenate([y_train, y_test])
        full_df_aligned = pd.concat(
            [train_df_aligned, test_df_aligned], ignore_index=True
        )
        sample_weights_full = compute_sample_weights(
            df=full_df_aligned,
            group_col=processed_names.LONG_PC_TYPE,
            region_col=processed_names.LONG_REGION,
            target_region=cst.EUROPE,
            method=weighting_method,
        )
        with mlflow.start_run(run_name=f"{mlflow_run_name}_predict", nested=True):
            mlflow.set_tags(
                {
                    cst.MLFLOW_MODEL_PHILOSOPHY: "multivariate",
                    cst.MLFLOW_MODEL_TYPE: model_type,
                    cst.MLFLOW_HORIZON: horizon,
                    cst.MLFLOW_FUNCTION: "prediction",
                }
            )
            mlflow.log_params(best_params)
            pred_model.fit(
                X_full_train, y_full_train, sample_weight=sample_weights_full
            )

            # Log model with signature
            signature = infer_signature(X_full_train, pred_model.predict(X_full_train))
            log_model(
                model_type=model_type,
                model_role="prediction",
                model=pred_model,
                signature=signature,
            )
