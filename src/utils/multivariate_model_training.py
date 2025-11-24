"""Base multivariate model training functions."""

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

import constants.constants as cst
from src.utils.evaluation import compute_performance_metrics


def optimize_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights: pd.Series,
    hyperparameter_grid: dict | None,
    pc_types: pd.Series,
    **kwargs,
):
    """Optimize an XGBoost model with Optuna."""

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
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
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
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    return best_params
