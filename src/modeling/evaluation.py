"""Evaluation utilities for model performance assessment."""

import os
from typing import Literal

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_percentage_error

import constants.constants as cst
import constants.processed_names as processed_names
from src.modeling.multivariate_data_prep import (
    compute_sample_weights,
)
from src.utils.logger import logger


def multi_compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pc_types: pd.Series,
    weights: np.ndarray = None,
) -> dict[str, float]:
    """Compute performance metrics for the model.

    Three performance metrics are computed:
    - Global MAPE: Mean Absolute Percentage Error across all samples.
    - Weighted MAPE: MAPE computed with sample weights.
    - Per pc type MAPE: MAPE computed for each pc type separately.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        pc_types (pd.Series): PC types of the values.
        weights (np.ndarray, optional): Sample weights. Defaults to None.

    Returns:
        dict[str, float]: Dictionary containing the computed performance metrics.
    """
    performance_metrics = {}

    # Global MAPE
    global_mape = mean_absolute_percentage_error(y_true, y_pred)
    performance_metrics[cst.GLOBAL_MAPE] = global_mape

    # Weighted MAPE
    if weights is not None:
        weighted_mape = mean_absolute_percentage_error(
            y_true, y_pred, sample_weight=weights
        )
        performance_metrics[cst.WEIGHTED_MAPE] = weighted_mape

    # Per PC type MAPE
    for pc_type in pc_types.unique():
        pc_type_mask = pc_types == pc_type
        y_pc_specific_true = y_true[pc_type_mask]
        y_pc_specific_pred = y_pred[pc_type_mask]
        pc_specific_mape = mean_absolute_percentage_error(
            y_pc_specific_true, y_pc_specific_pred
        )
        performance_metrics[f"{pc_type}_MAPE"] = pc_specific_mape

    return performance_metrics


def analyze_model_with_shap(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_type: str,
    max_display: int = 20,
) -> dict:
    """Generate SHAP analysis for model interpretation.

    Args:
        model: Trained model (XGBoost, LightGBM, CatBoost, or RandomForest).
        X_train (np.ndarray): Training features for background data.
        X_test (np.ndarray): Test features to explain.
        feature_names (list[str]): List of feature names.
        model_type (str): Type of model ('xgboost', 'lightgbm', 'catboost',
            'random_forest').
        max_display (int, optional): Maximum features to display in plots.
            Defaults to 20.

    Returns:
        dict: Dictionary with SHAP values, feature importance, and plots.
    """
    logger.info(f"Generating SHAP analysis for {model_type} model...")

    # Create appropriate explainer based on model type
    if model_type in ["xgboost", "lightgbm", "catboost", "random_forest"]:
        # TreeExplainer works for all tree-based models (fast)
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer (slower but works for any model)
        explainer = shap.KernelExplainer(
            model.predict,
            shap.sample(X_train, 100),  # Use sample for speed
        )

    # Calculate SHAP values for test set
    logger.info(f"Computing SHAP values for {X_test.shape[0]} test samples...")
    shap_values = explainer.shap_values(X_test)

    # Get feature importance (mean absolute SHAP values)
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    logger.info(f"Top 5 features: {importance_df['feature'].head(5).tolist()}")

    # Generate SHAP summary plot (beeswarm)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    # Simplify x-axis label to prevent cutoff
    plt.xlabel("mean(|SHAP value|)")
    summary_plot_path = "shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Generate SHAP bar plot (mean absolute values)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    # Simplify x-axis label to prevent cutoff
    plt.xlabel("mean(|SHAP value|)")
    bar_plot_path = "shap_importance_bar.png"
    plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP analysis completed")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "summary_plot_path": summary_plot_path,
        "bar_plot_path": bar_plot_path,
    }


# Model Evaluation and Logging Functions
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
        mlflow.xgboost.log_model(
            model,
            name=f"{model_role}_model",
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )
    elif model_type == "lightgbm":
        mlflow.lightgbm.log_model(
            model,
            name=f"{model_role}_model",
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )
    elif model_type == "random_forest":
        mlflow.sklearn.log_model(
            model,
            name=f"{model_role}_model",
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )
    elif model_type == "catboost":
        mlflow.catboost.log_model(
            model,
            name=f"{model_role}_model",
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )
    elif model_type == "tft":
        mlflow.pytorch.log_model(
            model,
            name=f"{model_role}_model",
            signature=signature,
            pip_requirements=None,
            conda_env=None,
        )
    else:
        raise ValueError(f"Unsupported model type for logging: {model_type}")


def multi_evaluate_and_log_model(
    eval_model,
    pred_model,
    best_params: dict,
    mlflow_run_name: str,
    model_type: Literal["xgboost", "random_forest", "lightgbm", "catboost", "tft"],
    horizon: int,
    use_validation_set: bool,
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
    feature_cols: list[str],
    shap_max_display: int,
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
        use_validation_set (bool): Whether to use a validation set.
        group_by_pc_types (bool): Whether PC types are grouped.
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
        feature_cols (list[str]): List of feature column names for SHAP analysis.
        shap_max_display (int): Maximum number of features to display in SHAP plots.
    """
    with mlflow.start_run(run_name=f"{mlflow_run_name}_eval"):
        mlflow.set_tags(
            {
                cst.MLFLOW_MODEL_PHILOSOPHY: "multivariate",
                cst.MLFLOW_MODEL_TYPE: model_type,
                cst.MLFLOW_HORIZON: horizon,
                cst.MLFLOW_FUNCTION: "evaluation",
                cst.MLFLOW_USE_VALIDATION_SET: str(use_validation_set),
                cst.MLFLOW_GROUP_BY_PC_TYPES: str(group_by_pc_types),
            }
        )
        mlflow.log_params(best_params)
        # Train evaluation model with TRAIN weights
        if use_validation_set:
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
        else:
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

        # SHAP Analysis for model interpretation
        logger.info("Generating SHAP analysis for model interpretation...")
        shap_results = analyze_model_with_shap(
            model=eval_model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_cols,
            model_type=model_type,
            max_display=shap_max_display,
        )

        # Log SHAP plots to MLflow as artifacts
        mlflow.log_artifact(shap_results["summary_plot_path"])
        mlflow.log_artifact(shap_results["bar_plot_path"])

        # Clean up plot files after logging
        os.remove(shap_results["summary_plot_path"])
        os.remove(shap_results["bar_plot_path"])

        # Save feature importance CSV as artifact
        importance_path = "shap_feature_importance.csv"
        shap_results["feature_importance"].to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Clean up CSV file after logging
        os.remove(importance_path)

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
                    cst.MLFLOW_USE_VALIDATION_SET: str(use_validation_set),
                    cst.MLFLOW_GROUP_BY_PC_TYPES: str(group_by_pc_types),
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
