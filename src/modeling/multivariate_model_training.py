"""Base multivariate model training functions."""

from typing import Literal

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from constants import processed_names
import constants.constants as cst
from src.modeling.evaluation import (
    multi_compute_performance_metrics,
    multi_evaluate_and_log_model,
)
from src.modeling.multivariate_data_prep import (
    adaptive_train_test_split,
    adaptive_train_validation_test_split,
    compute_sample_weights,
    load_and_prepare_data,
    prepare_training_data,
)


# Model Optimization Functions
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


def optimize_lightgbm_model(
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
    """Optimize a LightGBM model with Optuna.

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
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                # Learning rate
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                # Regularization
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-3, 10.0, log=True
                ),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                # Sampling
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                # Additional LightGBM-specific parameters
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", 1e-8, 1.0, log=True
                ),
            }
        )

        model = LGBMRegressor(**params, **kwargs)
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


def optimize_catboost_model(
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
    """Optimize a CatBoost model with Optuna.

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
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 3, 10),
                # Learning rate
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                # Regularization
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "max_leaves": trial.suggest_int("max_leaves", 20, 64),
                # Sampling
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                # CatBoost-specific parameters
                "border_count": trial.suggest_int("border_count", 32, 255),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-8, 10.0, log=True
                ),
            }
        )

        model = CatBoostRegressor(**params, **kwargs)
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


def optimize_random_forest_model(
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
    """Optimize a Random Forest model with Optuna.

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
                # Number of trees
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                # Tree structure
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                # Feature sampling
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                # Sampling
                "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
                "bootstrap": True,  # Keep bootstrap enabled for bagging
                # Regularization
                "min_impurity_decrease": trial.suggest_float(
                    "min_impurity_decrease", 0.0, 0.1
                ),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 500),
                # Fixed settings
                "n_jobs": -1,  # Use all available cores
            }
        )

        model = RandomForestRegressor(**params, **kwargs)
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


# Global Model Training Function
def train_global_model(
    group_by_pc_types: bool,
    horizon: int,
    use_validation_set: bool,
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
    shap_max_display: int,
) -> None:
    """Train a global model on all PC types and log it to MLflow.

    Args:
        group_by_pc_types (bool): Whether PC prices are grouped by type.
        horizon (int): Forecast horizon in months.
        use_validation_set (bool): Whether to use a validation set.
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
        shap_max_display (int): Maximum number of features to display in SHAP plots.
    """
    # 1. Load and prepare data (categorical features are label encoded)
    df, target_col, feature_cols = load_and_prepare_data(
        group_by_pc_types=group_by_pc_types, horizon=horizon
    )

    # 2. Split data into training and testing sets
    if use_validation_set:
        train_df, validation_df, test_df = adaptive_train_validation_test_split(
            df=df,
            group_col=processed_names.LONG_PC_TYPE,
            target_test_ratio=target_test_ratio,
            target_validation_ratio=target_validation_ratio,
            min_train_samples=min_train_samples,
            min_validation_samples=min_validation_samples,
            min_test_samples=min_test_samples,
        )
    else:
        train_df, test_df = adaptive_train_test_split(
            df=df,
            group_col=processed_names.LONG_PC_TYPE,
            target_test_ratio=target_test_ratio,
            min_train_samples=min_train_samples,
            min_test_samples=min_test_samples,
        )

    # 3. Prepare training and testing data
    if use_validation_set:
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
    else:
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

    # 4. Compute sample weights for train and test
    train_sample_weights = compute_sample_weights(
        df=train_df_aligned,
        group_col=processed_names.LONG_PC_TYPE,
        region_col=processed_names.LONG_REGION,
        target_region=cst.EUROPE,
        method=weighting_method,
    )

    if use_validation_set:
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
        if use_validation_set:
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
        else:
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
        eval_model = XGBRegressor(**best_params)
        pred_model = XGBRegressor(**best_params)

    elif model_type == "lightgbm":
        if use_validation_set:
            best_params = optimize_lightgbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_validation,
                y_test=y_validation,
                weights=validation_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=validation_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        else:
            best_params = optimize_lightgbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                weights=test_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=test_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        eval_model = LGBMRegressor(**best_params)
        pred_model = LGBMRegressor(**best_params)

    elif model_type == "catboost":
        if use_validation_set:
            best_params = optimize_catboost_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_validation,
                y_test=y_validation,
                weights=validation_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=validation_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        else:
            best_params = optimize_catboost_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                weights=test_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=test_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        eval_model = CatBoostRegressor(**best_params)
        pred_model = CatBoostRegressor(**best_params)

    elif model_type == "random_forest":
        if use_validation_set:
            best_params = optimize_random_forest_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_validation,
                y_test=y_validation,
                weights=validation_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=validation_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        else:
            best_params = optimize_random_forest_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                weights=test_sample_weights,
                hyperparameter_grid=hyperparameter_grid,
                pc_types=test_df_aligned[processed_names.LONG_PC_TYPE],
                n_trials=n_trials,
            )
        eval_model = RandomForestRegressor(**best_params)
        pred_model = RandomForestRegressor(**best_params)

    elif model_type == "tft":
        # Placeholder for TFT implementation
        pass
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented.")

    # 6. Evaluate and log model (for all model types)
    if model_type != "tft":  # Skip TFT as it's not implemented yet
        multi_evaluate_and_log_model(
            eval_model=eval_model,
            pred_model=pred_model,
            best_params=best_params,
            mlflow_run_name=mlflow_run_name,
            model_type=model_type,
            horizon=horizon,
            use_validation_set=use_validation_set,
            weighting_method=weighting_method,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation if use_validation_set else None,
            y_validation=y_validation if use_validation_set else None,
            X_test=X_test,
            y_test=y_test,
            train_df_aligned=train_df_aligned,
            validation_df_aligned=validation_df_aligned if use_validation_set else None,
            test_df_aligned=test_df_aligned,
            train_sample_weights=train_sample_weights,
            test_sample_weights=test_sample_weights,
            feature_cols=feature_cols,
            shap_max_display=shap_max_display,
        )
