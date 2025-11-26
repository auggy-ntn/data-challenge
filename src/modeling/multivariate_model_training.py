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
from src.modeling.evaluation import multi_compute_performance_metrics
from src.modeling.multivariate_data_prep import (
    adaptive_train_test_split,
    adaptive_train_validation_test_split,
    compute_sample_weights,
    load_and_prepare_data,
    prepare_training_data,
)


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
        weighted_mape = multi_compute_performance_metrics(
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
        performance_metrics = multi_compute_performance_metrics(
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


def train_global_model(
    group_by_pc_types: bool,
    horizon: int,
    target_test_ratio: float,
    target_validation_ratio: float,
    min_train_samples: int,
    min_validation_samples: int,
    min_test_samples: int,
    weighting_method: Literal["inverse_frequency", "sqrt_inverse", "balanced"],
    model_type: Literal["xgboost", "random_forest", "lightgbm", "catboost", "tft"],
    hyperparameter_grid: dict | None,
    mlflow_run_name: str,
    n_trials: int,
) -> None:
    """Train a global model on all PC types and log it to MLflow.

    Args:
        group_by_pc_types (bool): Whether PC prices are grouped by type.
        horizon (int): Forecast horizon in months.
        target_test_ratio (float): Desired test set size.
        target_validation_ratio (float): Desired validation set size.
        min_train_samples (int): Minimum training samples per group.
        min_validation_samples (int): Minimum validation samples per group.
        min_test_samples (int): Minimum test samples per group.
        weighting_method (Literal): Method to compute sample weights.
        model_type (Literal): Type of model to train.
        hyperparameter_grid (dict | None): Hyperparameter grid for optimization.
        mlflow_run_name (str): Name for the MLflow run.
        n_trials (int): Number of trials for hyperparameter optimization.
    """
    # 1. Load and prepare data
    df, target_col, feature_cols = load_and_prepare_data(
        group_by_pc_types=group_by_pc_types, horizon=horizon
    )

    # 2. Split data into training and testing sets
    if group_by_pc_types:
        train_df, test_df = adaptive_train_test_split(
            df=df,
            group_col=processed_names.LONG_PC_TYPE,
            target_test_ratio=target_test_ratio,
            min_train_samples=min_train_samples,
            min_test_samples=min_test_samples,
        )
    else:
        train_df, validation_df, test_df = adaptive_train_validation_test_split(
            df=df,
            group_col=processed_names.LONG_PC_TYPE,
            target_test_ratio=target_test_ratio,
            target_validation_ratio=target_validation_ratio,
            min_train_samples=min_train_samples,
            min_validation_samples=min_validation_samples,
            min_test_samples=min_test_samples,
        )

    # 3. Prepare training and testing data
    if group_by_pc_types:
        X_train, y_train, X_test, y_test, train_df_aligned, test_df_aligned = (
            prepare_training_data(
                train_df=train_df,
                validation_df=None,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col=target_col,
                horizon=horizon,
            )
        )
    else:
        (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            train_df_aligned,
            validation_df_aligned,
            test_df_aligned,
        ) = prepare_training_data(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=horizon,
        )

    # 4. Compute sample weights for train and test
    train_sample_weights = compute_sample_weights(
        df=train_df_aligned,
        group_col=processed_names.LONG_PC_TYPE,
        region_col=processed_names.LONG_REGION,
        target_region=cst.EUROPE,
        method=weighting_method,
    )

    if not group_by_pc_types:
        validation_sample_weights = compute_sample_weights(
            df=validation_df_aligned,
            group_col=processed_names.LONG_PC_TYPE,
            region_col=processed_names.LONG_REGION,
            target_region=cst.EUROPE,
            method=weighting_method,
        )

    test_sample_weights = compute_sample_weights(
        df=test_df_aligned,
        group_col=processed_names.LONG_PC_TYPE,
        region_col=processed_names.LONG_REGION,
        target_region=cst.EUROPE,
        method=weighting_method,
    )

    # 5. Train model
    if model_type == "xgboost":
        if group_by_pc_types:
            best_params = optimize_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                weights=test_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=test_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        else:
            best_params = optimize_xgboost_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_validation,
                y_test=y_validation,
                weights=validation_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=validation_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        eval_model = XGBRegressor(**best_params)
        pred_model = XGBRegressor(**best_params)

        # 6. Evaluate and log model
        evaluate_and_log_model(
            eval_model=eval_model,
            pred_model=pred_model,
            best_params=best_params,
            mlflow_run_name=mlflow_run_name,
            model_type=model_type,
            horizon=horizon,
            group_by_pc_types=group_by_pc_types,
            weighting_method=weighting_method,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation if not group_by_pc_types else None,
            y_validation=y_validation if not group_by_pc_types else None,
            X_test=X_test,
            y_test=y_test,
            train_df_aligned=train_df_aligned,
            validation_df_aligned=validation_df_aligned
            if not group_by_pc_types
            else None,
            test_df_aligned=test_df_aligned,
            train_sample_weights=train_sample_weights,
            test_sample_weights=test_sample_weights,
        )

    elif model_type == "random_forest":
        # Placeholder for random forest implementation
        pass

    elif model_type == "lightgbm":
        # Placeholder for LightGBM implementation
        pass

    elif model_type == "catboost":
        # Placeholder for CatBoost implementation
        pass

    elif model_type == "tft":
        # Placeholder for TFT implementation
        pass
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")
