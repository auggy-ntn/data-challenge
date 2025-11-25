"""Data splitting utilities for time series with imbalanced groups."""

import pandas as pd

from constants import processed_names
from constants.constants import PCType
from src.utils.logger import logger


def adaptive_train_test_split(
    df: pd.DataFrame,
    group_col: str = processed_names.LONG_PC_TYPE,
    target_test_ratio: float = 0.2,
    min_train_samples: int = 20,
    min_test_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adaptive temporal split that ensures minimum samples per group.

    If standard 80/20 split results in insufficient test samples for any group,
    adjusts split date to ensure minimum requirements.

    Args:
        df: Long format dataframe with 'date' column
        group_col: Column defining groups
        target_test_ratio: Desired test set size (default 0.2 = 20%)
        min_train_samples: Minimum training samples per group
        min_test_samples: Minimum test samples per group

    Returns:
        (train_df, test_df) with sufficient samples per group
    """
    df = df.sort_values(processed_names.LONG_DATE).reset_index(drop=True)

    # Find the latest date that satisfies constraints
    dates = sorted(df[processed_names.LONG_DATE].unique())

    for split_date in reversed(dates):
        train = df[df[processed_names.LONG_DATE] < split_date]
        test = df[df[processed_names.LONG_DATE] >= split_date]
        # Check if all groups meet minimum requirements
        valid = True
        target_pc_types = [
            pc_type
            for pc_type in df[group_col].unique()
            if pc_type in PCType._value2member_map_
        ]
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

    return train, test
