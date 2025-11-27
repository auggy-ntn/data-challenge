"""Data loading and preparation for multivariate time series forecasting."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from constants import processed_names
import constants.constants as cst
from constants.paths import PROCESSED_DATA_DIR
from src.utils.logger import logger


def load_and_prepare_data(
    group_by_pc_types: bool,
    horizon: int = 3,
) -> tuple[pd.DataFrame, str, list[str], dict]:
    """Load processed data and separate features and target variable.

    Categorical features (region, pc_type) are always label encoded for compatibility
    with numpy arrays used in model training.

    Args:
        group_by_pc_types (bool): Whether PC prices are grouped by type.
        horizon (int, optional): Forecast horizon in months. Defaults to 3.

    Returns:
        tuple[pd.DataFrame, str, list[str], dict]: DataFrame, target column name,
        feature column names (including encoded categoricals), and encoding mappings.
    """
    # Load processed data
    if group_by_pc_types:
        df = pd.read_csv(PROCESSED_DATA_DIR / f"multi_{horizon}m_grouped.csv")
    else:
        df = pd.read_csv(PROCESSED_DATA_DIR / f"multi_{horizon}m.csv")

    # Separate features and target
    target_col = processed_names.LONG_PC_PRICE
    meta_cols = [
        processed_names.LONG_DATE,
        processed_names.LONG_REGION,
        processed_names.LONG_PC_TYPE,
    ]
    categorical_cols = [
        processed_names.LONG_REGION,
        processed_names.LONG_PC_TYPE,
    ]

    # Get base numerical features (exclude meta and target)
    base_feature_cols = [
        col for col in df.columns if col not in meta_cols + [target_col]
    ]

    # Label encode categorical features (works for all tree-based models)
    logger.info(f"Label encoding categorical features: {', '.join(categorical_cols)}")
    encoding_mappings = {}

    for col in categorical_cols:
        encoded_col = f"{col}_encoded"
        le = LabelEncoder()
        df[encoded_col] = le.fit_transform(df[col])

        # Create mapping dictionary
        mapping = {
            str(k): int(v)
            for k, v in zip(le.classes_, le.transform(le.classes_), strict=True)
        }
        encoding_mappings[col] = mapping

        # Log the mapping
        logger.info(f"  {col} -> {encoded_col}:")
        for original, encoded in mapping.items():
            logger.info(f"    {original} = {encoded}")

    # Combine encoded categoricals with base features
    feature_cols = [
        f"{processed_names.LONG_REGION}_encoded",
        f"{processed_names.LONG_PC_TYPE}_encoded",
    ] + base_feature_cols

    logger.info(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    logger.info(f"Target: {target_col}. Features: {len(feature_cols)} columns.")

    return df, target_col, feature_cols, encoding_mappings


# Use for dataset not grouped by PC types (more data points per group)
def adaptive_train_validation_test_split(
    df: pd.DataFrame,
    group_col: str = processed_names.LONG_PC_TYPE,
    group_by_pc_types: bool = False,
    target_test_ratio: float = 0.1,
    target_validation_ratio: float = 0.1,
    min_train_samples: int = 20,
    min_validation_samples: int = 5,
    min_test_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adaptive temporal split that ensures minimum samples per group.

    If standard split results in insufficient test samples for any group,
    adjusts split date to ensure minimum requirements.

    Args:
        df: Long format dataframe with 'date' column
        group_col: Column defining groups
        group_by_pc_types: Whether PC prices are grouped by type
        target_test_ratio: Desired test set size (default 0.1 = 10% of total data)
        target_validation_ratio: Desired validation set size (default 0.1 = 10%
                                 of total data)
        min_train_samples: Minimum training samples per group
        min_test_samples: Minimum test samples per group
        min_validation_samples: Minimum validation samples per group

    Returns:
        (train_df, validation_df,test_df) with sufficient samples per group
    """
    df = df.sort_values(processed_names.LONG_DATE).reset_index(drop=True)

    # Find the latest date that satisfies constraints
    dates = sorted(df[processed_names.LONG_DATE].unique())

    # Target PC types to check
    if group_by_pc_types:
        target_pc_types = [
            pc_type
            for pc_type in df[group_col].unique()
            if pc_type in cst.PCType._value2member_map_
        ]
    else:
        target_pc_types = [cst.REGULAR_PC_TYPE, cst.GREEN_PC_TYPE]

    for validation_split_date in reversed(dates):
        possible_test_dates = [date for date in dates if date > validation_split_date]
        for test_split_date in reversed(possible_test_dates):
            train = df[df[processed_names.LONG_DATE] < validation_split_date]
            validation = df[
                (df[processed_names.LONG_DATE] >= validation_split_date)
                & (df[processed_names.LONG_DATE] < test_split_date)
            ]
            test = df[df[processed_names.LONG_DATE] >= test_split_date]
            # Check if all groups meet minimum requirements
            valid = True
            for target_pc in target_pc_types:
                train_count = (train[group_col] == target_pc).sum()
                validation_count = (validation[group_col] == target_pc).sum()
                test_count = (test[group_col] == target_pc).sum()

                if (
                    train_count < min_train_samples
                    or validation_count < min_validation_samples
                    or test_count < min_test_samples
                ):
                    valid = False
                    break

            if valid:
                logger.info(
                    f"Found valid split at dates: "
                    f"validation {validation_split_date}, test {test_split_date}"
                )
                actual_test_ratio = len(test) / len(df)
                actual_validation_ratio = len(validation) / len(df)
                logger.info(f"Actual test set ratio: {actual_test_ratio:.2%}")
                logger.info(f"Desired test set ratio: {target_test_ratio:.2%}")
                logger.info(
                    f"Actual validation set ratio: {actual_validation_ratio:.2%}"
                )
                logger.info(
                    f"Desired validation set ratio: {target_validation_ratio:.2%}"
                )

                return train, validation, test

    # If no valid split found, fall back to target ratios
    logger.warning(
        f"No valid adaptive split found satisfying minimum sample constraints "
        f"(train≥{min_train_samples}, validation≥{min_validation_samples}, "
        f"test≥{min_test_samples}). Falling back to "
        f"{1 - target_test_ratio - target_validation_ratio:.0%}/"
        f"{target_validation_ratio:.0%}/{target_test_ratio:.0%} split."
    )

    # Use index-based splitting on unique dates (more reliable than quantile)
    dates = sorted(df[processed_names.LONG_DATE].unique())
    n_dates = len(dates)

    val_idx = int(n_dates * (1 - target_test_ratio - target_validation_ratio))
    test_idx = int(n_dates * (1 - target_test_ratio))

    split_date_validation = dates[val_idx]
    split_date_test = dates[test_idx]

    train = df[df[processed_names.LONG_DATE] < split_date_validation]
    validation = df[
        (df[processed_names.LONG_DATE] >= split_date_validation)
        & (df[processed_names.LONG_DATE] < split_date_test)
    ]
    test = df[df[processed_names.LONG_DATE] >= split_date_test]

    # Log warning about groups that don't meet requirements
    for target_pc in target_pc_types:
        train_count = (train[group_col] == target_pc).sum()
        val_count = (validation[group_col] == target_pc).sum()
        test_count = (test[group_col] == target_pc).sum()

        if (
            train_count < min_train_samples
            or val_count < min_validation_samples
            or test_count < min_test_samples
        ):
            logger.warning(
                f"PC type '{target_pc}' has insufficient samples in fallback split: "
                f"train={train_count} (min={min_train_samples}), "
                f"val={val_count} (min={min_validation_samples}), "
                f"test={test_count} (min={min_test_samples})"
            )

    return train, validation, test


# Use for data grouped by PC types (fewer data points per group)
def adaptive_train_test_split(
    df: pd.DataFrame,
    group_col: str = processed_names.LONG_PC_TYPE,
    group_by_pc_types: bool = False,
    target_test_ratio: float = 0.2,
    min_train_samples: int = 20,
    min_test_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adaptive temporal split that ensures minimum samples per group.

    If standard split results in insufficient test samples for any group,
    adjusts split date to ensure minimum requirements.

    Args:
        df: Long format dataframe with 'date' column
        group_col: Column defining groups
        group_by_pc_types: Whether PC prices are grouped by type
        target_test_ratio: Desired test set size (default 0.2 = 20%)
        min_train_samples: Minimum training samples per group
        min_test_samples: Minimum test samples per group

    Returns:
        (train_df, test_df) with sufficient samples per group
    """
    df = df.sort_values(processed_names.LONG_DATE).reset_index(drop=True)

    # Find the latest date that satisfies constraints
    dates = sorted(df[processed_names.LONG_DATE].unique())

    # Target PC types to check
    if group_by_pc_types:
        target_pc_types = [
            pc_type
            for pc_type in df[group_col].unique()
            if pc_type in cst.PCType._value2member_map_
        ]
    else:
        target_pc_types = [cst.REGULAR_PC_TYPE, cst.GREEN_PC_TYPE]

    for split_date in reversed(dates):
        train = df[df[processed_names.LONG_DATE] < split_date]
        test = df[df[processed_names.LONG_DATE] >= split_date]
        # Check if all groups meet minimum requirements
        valid = True
        for target_pc in target_pc_types:
            train_count = (train[group_col] == target_pc).sum()
            test_count = (test[group_col] == target_pc).sum()

            if train_count < min_train_samples or test_count < min_test_samples:
                valid = False
                break

        if valid:
            logger.info(f"Found valid split at date: {split_date}")
            actual_test_ratio = len(test) / len(df)
            logger.info(f"Actual test set ratio: {actual_test_ratio:.2%}")
            logger.info(f"Desired test set ratio: {target_test_ratio:.2%}")

            return train, test

    # If no valid split found, warn user
    logger.warning(
        "Could not find a split date satisfying minimum sample requirements "
        "for all groups. Falling back to target test ratio split."
    )

    # Fallback: use target ratio
    n_test = int(len(dates) * target_test_ratio)
    split_date = dates[-n_test]
    train = df[df[processed_names.LONG_DATE] < split_date]
    test = df[df[processed_names.LONG_DATE] >= split_date]

    # Log warning about groups that don't meet requirements
    for target_pc in target_pc_types:
        train_count = (train[group_col] == target_pc).sum()
        test_count = (test[group_col] == target_pc).sum()

        if train_count < min_train_samples or test_count < min_test_samples:
            logger.warning(
                f"PC type '{target_pc}' has insufficient samples in fallback split: "
                f"train={train_count} (min={min_train_samples}), "
                f"test={test_count} (min={min_test_samples})"
            )

    return train, test


# Prepare training data
def prepare_training_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame | None,
    pd.DataFrame,
]:
    """Prepare features and target variable for training.

    To align the target variable with the features, the target is shifted
    by -horizon months within each group defined by region and pc_type.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        validation_df (pd.DataFrame | None): Validation dataframe.
        test_df (pd.DataFrame): Testing dataframe.
        feature_cols (list[str]): List of feature column names.
        target_col (str): Target column name.
        horizon (int): Forecast horizon in months.

    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None,
        np.ndarray, pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
            X_train, y_train, X_validation, y_validation, X_test, y_test,
            aligned training, validation and testing dataframes
    """
    # Target (shift by -horizon to align with features)
    # Group by region and pc_type to shift correctly
    train_df["target"] = train_df.groupby(
        [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]
    )[target_col].shift(-horizon)
    if validation_df is not None:
        validation_df["target"] = validation_df.groupby(
            [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]
        )[target_col].shift(-horizon)
    test_df["target"] = test_df.groupby(
        [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]
    )[target_col].shift(-horizon)

    # Drop rows with NaN in target (due to shifting)
    train_mask = ~train_df["target"].isna()
    if validation_df is not None:
        validation_mask = ~validation_df["target"].isna()
    test_mask = ~test_df["target"].isna()

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, "target"].values
    train_df_aligned = train_df[train_mask].copy()
    logger.info(f"Training data prepared with {X_train.shape[0]} samples.")

    if validation_df is not None:
        X_validation = validation_df.loc[validation_mask, feature_cols].values
        y_validation = validation_df.loc[validation_mask, "target"].values
        validation_df_aligned = validation_df[validation_mask].copy()
        logger.info(f"Validation data prepared with {X_validation.shape[0]} samples.")

    X_test = test_df.loc[test_mask, feature_cols].values
    y_test = test_df.loc[test_mask, "target"].values
    test_df_aligned = test_df[test_mask].copy()
    logger.info(f"Testing data prepared with {X_test.shape[0]} samples.")

    if validation_df is not None:
        return (
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            train_df_aligned,
            validation_df_aligned,
            test_df_aligned,
        )

    return X_train, y_train, X_test, y_test, train_df_aligned, test_df_aligned


# Compute sample weights
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
