"""Evaluation utilities for model performance assessment."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

import constants.constants as cst


def compute_performance_metrics(
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
