# Feature Engineering Utilities

import numpy as np
import pandas as pd

from constants import intermediate_names, processed_names
import constants.constants as cst
from constants.paths import (
    INTERMEDIATE_AUTOMOBILE_INDUSTRY_DIR,
    INTERMEDIATE_ELECTRICITY_PRICE_DIR,
    INTERMEDIATE_PC_PRICE_DIR,
    INTERMEDIATE_PHENOL_ACETONE_DIR,
    INTERMEDIATE_SHUTDOWN_DIR,
)

############################ Univariate Feature Engineering ############################


def create_wide_format() -> pd.DataFrame:
    """Creates a wide format dataset with all the features for each PC type.

    In this dataset, there is one column per PC type and one column per feature
    engineered for that PC type.

    Returns:
        pd.DataFrame: The wide format dataset.
    """
    # Merge EU and Asia PC price datasets
    eu_renamed = pd.read_csv(INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_eu.csv")
    asia_renamed = pd.read_csv(
        INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_asia.csv"
    )

    eu_renamed[intermediate_names.PC_EU_DATE] = pd.to_datetime(
        eu_renamed[intermediate_names.PC_EU_DATE], format=cst.DATE_FORMAT
    )
    asia_renamed[intermediate_names.PC_ASIA_DATE] = pd.to_datetime(
        asia_renamed[intermediate_names.PC_ASIA_DATE], format=cst.DATE_FORMAT
    )

    # Prefix columns with region
    eu_renamed_cols = []
    for col in eu_renamed.columns:
        if col != intermediate_names.PC_EU_DATE and "best_price" in col:
            # "best price" comes from preprocessing step
            eu_renamed.rename(columns={col: f"eu_{col}"}, inplace=True)
            eu_renamed_cols.append(f"eu_{col}")
    # Keep only date and renamed columns
    eu_renamed = eu_renamed[[intermediate_names.PC_EU_DATE] + eu_renamed_cols]

    asia_renamed_cols = []
    for col in asia_renamed.columns:
        if col != intermediate_names.PC_ASIA_DATE and "best_price" in col:
            asia_renamed.rename(columns={col: f"asia_{col}"}, inplace=True)
            asia_renamed_cols.append(f"asia_{col}")
    # Keep only date and renamed columns
    asia_renamed = asia_renamed[[intermediate_names.PC_ASIA_DATE] + asia_renamed_cols]

    wide_df = pd.merge(
        eu_renamed,
        asia_renamed,
        left_on=intermediate_names.PC_EU_DATE,
        right_on=intermediate_names.PC_ASIA_DATE,
        how="outer",
        validate="1:1",
    )

    # BPA capacity loss
    bpa_capacity_loss = pd.read_csv(
        INTERMEDIATE_PHENOL_ACETONE_DIR / "intermediate_bpa_capacity_loss.csv"
    )
    bpa_capacity_loss[intermediate_names.BPA_DATE] = pd.to_datetime(
        bpa_capacity_loss[intermediate_names.BPA_DATE], format=cst.DATE_FORMAT
    )

    wide_df = pd.merge(
        wide_df,
        bpa_capacity_loss,
        left_on=processed_names.WIDE_DATE,
        right_on=intermediate_names.BPA_DATE,
        how="left",
        validate="1:1",
    )

    # Electricity prices
    electricity_prices = pd.read_csv(
        INTERMEDIATE_ELECTRICITY_PRICE_DIR
        / "intermediate_european_wholesale_electricity_price.csv"
    )
    electricity_prices[intermediate_names.ELECTRICITY_DATE] = pd.to_datetime(
        electricity_prices[intermediate_names.ELECTRICITY_DATE], format=cst.DATE_FORMAT
    )
    wide_df = pd.merge(
        wide_df,
        electricity_prices,
        left_on=processed_names.WIDE_DATE,
        right_on=intermediate_names.ELECTRICITY_DATE,
        how="left",
        validate="1:1",
    )

    # Automobile industry - new passenger car registrations
    auto_industry = pd.read_csv(
        INTERMEDIATE_AUTOMOBILE_INDUSTRY_DIR / "intermediate_automobile_industry.csv"
    )
    auto_industry[intermediate_names.AI_DATE] = pd.to_datetime(
        auto_industry[intermediate_names.AI_DATE], format=cst.DATE_FORMAT
    )
    wide_df = pd.merge(
        wide_df,
        auto_industry,
        left_on=processed_names.WIDE_DATE,
        right_on=intermediate_names.AI_DATE,
        how="left",
        validate="1:1",
    )

    # Shutdown capacity loss
    shutdown_capacity_loss = pd.read_csv(
        INTERMEDIATE_SHUTDOWN_DIR / "intermediate_shutdown_capacity_loss.csv"
    )
    shutdown_capacity_loss[intermediate_names.SHUTDOWN_DATE] = pd.to_datetime(
        shutdown_capacity_loss[intermediate_names.SHUTDOWN_DATE], format=cst.DATE_FORMAT
    )
    wide_df = pd.merge(
        wide_df,
        shutdown_capacity_loss,
        left_on=processed_names.WIDE_DATE,
        right_on=intermediate_names.SHUTDOWN_DATE,
        how="left",
        validate="1:1",
    )

    # TODO: Add other datasets features here (Commodities, etc.)

    return wide_df


def uni_add_lag_features(
    df: pd.DataFrame, lags: list[int], target_cols: list[str]
) -> pd.DataFrame:
    """Add lag features for wide format.

    Args:
        df: Wide format dataframe
        lags: List of lag periods (e.g., [1, 3, 6, 12])
        target_cols: Column names to create lag features for

    Returns:
        Dataframe with lag features added
    """
    new_cols = {}
    for target_col in target_cols:
        for lag in lags:
            new_cols[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def uni_add_rolling_features(
    df: pd.DataFrame, window_sizes: list[int], target_cols: list[str]
) -> pd.DataFrame:
    """Add rolling mean and std features for specified window sizes.

    Args:
        df (pd.DataFrame): Input dataframe.
        window_sizes (list[int]): List of window sizes for rolling calculations.
        target_cols (list[str]): Column names to compute rolling features on.

        Returns: pd.DataFrame with new rolling features added.
    """
    new_cols = {}
    for target_col in target_cols:
        for window in window_sizes:
            new_cols[f"{target_col}_roll_mean_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            new_cols[f"{target_col}_roll_std_{window}"] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def uni_add_expanding_features(
    df: pd.DataFrame, target_cols: list[str]
) -> pd.DataFrame:
    """Add expanding window statistics.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_cols (list[str]): Column names to compute expanding features on.

    Returns:
        pd.DataFrame with new expanding features added.
    """
    new_cols = {}
    for target_col in target_cols:
        new_cols[f"{target_col}_expanding_mean"] = (
            df[target_col].expanding(min_periods=1).mean()
        )
        new_cols[f"{target_col}_expanding_std"] = (
            df[target_col].expanding(min_periods=1).std()
        )
        new_cols[f"{target_col}_expanding_min"] = (
            df[target_col].expanding(min_periods=1).min()
        )
        new_cols[f"{target_col}_expanding_max"] = (
            df[target_col].expanding(min_periods=1).max()
        )

        # Distance from historical extremes
        new_cols[f"{target_col}_dist_from_max"] = (
            new_cols[f"{target_col}_expanding_max"] - df[target_col]
        )
        new_cols[f"{target_col}_dist_from_min"] = (
            df[target_col] - new_cols[f"{target_col}_expanding_min"]
        )

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def uni_add_rate_of_change_features(
    df: pd.DataFrame, periods: list[int], target_cols: list[str]
) -> pd.DataFrame:
    """Add rate of change features for specified periods.

    Args:
        df (pd.DataFrame): Input dataframe.
        periods (list[int]): List of periods for rate of change calculations.
        target_cols (list[str]): Column names to compute rate of change features on.

    Returns:
        pd.DataFrame with new rate of change features added.
    """
    new_cols = {}
    for target_col in target_cols:
        for period in periods:
            new_cols[f"{target_col}_roc_{period}"] = df[target_col].pct_change(
                periods=period, fill_method=None
            )
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def uni_add_differencing_features(
    df: pd.DataFrame, periods: list[int], target_cols: list[str]
) -> pd.DataFrame:
    """Add differencing features for stationarity.

    First-order differencing removes trends.
    Seasonal differencing removes seasonality.

    Args:
        df (pd.DataFrame): Input dataframe.
        periods (list[int]): List of periods for differencing.
        target_cols (list[str]): Column names to compute differencing features on.

    Returns:
        pd.DataFrame with new differencing features added.
    """
    new_cols = {}
    for target_col in target_cols:
        for period in periods:
            # First-order difference
            new_cols[f"{target_col}_diff_{period}"] = df[target_col].diff(
                periods=period
            )
            # Log difference (percentage change in log space)
            new_cols[f"{target_col}_log_diff_{period}"] = np.log(df[target_col]).diff(
                periods=period
            )

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)


def uni_add_time_features(
    df: pd.DataFrame, date_col: str = processed_names.WIDE_DATE
) -> pd.DataFrame:
    """Add temporal features to wide format.

    Args:
        df: Wide format dataframe
        date_col: Name of date column

    Returns:
        Dataframe with time features added
    """
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], format=cst.DATE_FORMAT)

    # Cyclical features - seasonality
    df[processed_names.WIDE_MONTH] = df[date_col].dt.month
    df[processed_names.WIDE_QUARTER] = df[date_col].dt.quarter
    df[processed_names.WIDE_MONTH_SIN] = np.sin(
        2 * np.pi * df[processed_names.WIDE_MONTH] / 12
    )
    df[processed_names.WIDE_MONTH_COS] = np.cos(
        2 * np.pi * df[processed_names.WIDE_MONTH] / 12
    )
    df[processed_names.WIDE_QUARTER_SIN] = np.sin(
        2 * np.pi * df[processed_names.WIDE_QUARTER] / 4
    )
    df[processed_names.WIDE_QUARTER_COS] = np.cos(
        2 * np.pi * df[processed_names.WIDE_QUARTER] / 4
    )

    # Trend
    df[processed_names.WIDE_YEAR] = df[date_col].dt.year

    # Time index (months since start)
    df[processed_names.WIDE_TIME_IDX] = (
        df[date_col] - df[date_col].min()
    ).dt.days / 30.44  # Approximate month length

    return df


########################### Multivariate Feature Engineering ###########################


def create_long_format() -> pd.DataFrame:
    """Creates a long format dataset suitable for multivariate global modeling.

    In this dataset, each row corresponds to a specific date and PC type,with associated
    features. Thus one date will have multiple rows, one for each PC type.

    Returns:
        pd.DataFrame: The long format dataset.
    """
    ## PC prices
    eu_df = pd.read_csv(INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_eu.csv")
    asia_df = pd.read_csv(INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_asia.csv")

    eu_df[intermediate_names.PC_EU_DATE] = pd.to_datetime(
        eu_df[intermediate_names.PC_EU_DATE], format=cst.DATE_FORMAT
    )
    asia_df[intermediate_names.PC_ASIA_DATE] = pd.to_datetime(
        asia_df[intermediate_names.PC_ASIA_DATE], format=cst.DATE_FORMAT
    )

    eu_long_dfs = []
    for col in eu_df.columns:
        if "best_price" in col:  # "best_price" comes from preprocessing step
            pc_type = col.replace("_best_price", "").replace("pc_", "")
            temp_df = eu_df[[intermediate_names.PC_EU_DATE, col]].copy()
            temp_df[processed_names.LONG_REGION] = processed_names.EUROPE
            temp_df[processed_names.LONG_PC_TYPE] = pc_type
            temp_df.rename(columns={col: processed_names.LONG_PC_PRICE}, inplace=True)
            eu_long_dfs.append(temp_df)

    asia_long_dfs = []
    for col in asia_df.columns:
        if "best_price" in col:
            pc_type = col.replace("_best_price", "").replace("pc_", "")
            temp_df = asia_df[[intermediate_names.PC_ASIA_DATE, col]].copy()
            temp_df[processed_names.LONG_REGION] = processed_names.ASIA
            temp_df[processed_names.LONG_PC_TYPE] = pc_type
            temp_df.rename(columns={col: processed_names.LONG_PC_PRICE}, inplace=True)
            asia_long_dfs.append(temp_df)

    # Combine
    long_df = pd.concat(eu_long_dfs + asia_long_dfs, ignore_index=True)
    long_df = long_df.sort_values(
        [
            processed_names.LONG_DATE,
            processed_names.LONG_REGION,
            processed_names.LONG_PC_TYPE,
        ]
    )

    # Drop rows where price is NaN (no observation available)
    long_df = long_df.dropna(subset=[processed_names.LONG_PC_PRICE]).reset_index(
        drop=True
    )

    # BPA capacity loss

    bpa_capacity_loss = pd.read_csv(
        INTERMEDIATE_PHENOL_ACETONE_DIR / "intermediate_bpa_capacity_loss.csv"
    )

    bpa_capacity_loss[intermediate_names.BPA_DATE] = pd.to_datetime(
        bpa_capacity_loss[intermediate_names.BPA_DATE], format=cst.DATE_FORMAT
    )

    long_df = pd.merge(
        long_df,
        bpa_capacity_loss,
        left_on=processed_names.LONG_DATE,
        right_on=intermediate_names.BPA_DATE,
        how="left",
        validate="m:1",
    )

    # Electricity prices
    electricity_prices = pd.read_csv(
        INTERMEDIATE_ELECTRICITY_PRICE_DIR
        / "intermediate_european_wholesale_electricity_price.csv"
    )
    electricity_prices[intermediate_names.ELECTRICITY_DATE] = pd.to_datetime(
        electricity_prices[intermediate_names.ELECTRICITY_DATE], format=cst.DATE_FORMAT
    )
    long_df = pd.merge(
        long_df,
        electricity_prices,
        left_on=processed_names.LONG_DATE,
        right_on=intermediate_names.ELECTRICITY_DATE,
        how="left",
        validate="m:1",
    )

    # Automobile industry - new passenger car registrations
    auto_industry = pd.read_csv(
        INTERMEDIATE_AUTOMOBILE_INDUSTRY_DIR / "intermediate_automobile_industry.csv"
    )
    auto_industry[intermediate_names.AI_DATE] = pd.to_datetime(
        auto_industry[intermediate_names.AI_DATE], format=cst.DATE_FORMAT
    )
    long_df = pd.merge(
        long_df,
        auto_industry,
        left_on=processed_names.LONG_DATE,
        right_on=intermediate_names.AI_DATE,
        how="left",
        validate="m:1",
    )

    # Shutdown capacity loss
    shutdown_capacity_loss = pd.read_csv(
        INTERMEDIATE_SHUTDOWN_DIR / "intermediate_shutdown_capacity_loss.csv"
    )
    shutdown_capacity_loss[intermediate_names.SHUTDOWN_DATE] = pd.to_datetime(
        shutdown_capacity_loss[intermediate_names.SHUTDOWN_DATE], format=cst.DATE_FORMAT
    )
    long_df = pd.merge(
        long_df,
        shutdown_capacity_loss,
        left_on=processed_names.LONG_DATE,
        right_on=intermediate_names.SHUTDOWN_DATE,
        how="left",
        validate="m:1",
    )

    # TODO: Add other datasets features here (Commodities, etc.)

    return long_df


def multi_add_time_features(
    df: pd.DataFrame, date_col: str = processed_names.LONG_DATE
) -> pd.DataFrame:
    """Add temporal features.

    Args:
        df: Long format dataframe
        date_col: Name of date column

    Returns:
        Dataframe with time features added
    """
    df = df.copy()

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], format=cst.DATE_FORMAT)

    # Cyclical features - seasonality
    df[processed_names.LONG_MONTH] = df[date_col].dt.month
    df[processed_names.LONG_QUARTER] = df[date_col].dt.quarter
    df[processed_names.LONG_MONTH_SIN] = np.sin(
        2 * np.pi * df[processed_names.LONG_MONTH] / 12
    )
    df[processed_names.LONG_MONTH_COS] = np.cos(
        2 * np.pi * df[processed_names.LONG_MONTH] / 12
    )
    df[processed_names.LONG_QUARTER_SIN] = np.sin(
        2 * np.pi * df[processed_names.LONG_QUARTER] / 4
    )
    df[processed_names.LONG_QUARTER_COS] = np.cos(
        2 * np.pi * df[processed_names.LONG_QUARTER] / 4
    )

    # Trend
    df[processed_names.LONG_YEAR] = df["date"].dt.year

    # Time index (months since start)
    df[processed_names.LONG_TIME_IDX] = (
        df[processed_names.LONG_DATE] - df[processed_names.LONG_DATE].min()
    ).dt.days / 30.44  # Approximate month length

    return df


def multi_add_lag_features(
    df: pd.DataFrame,
    lags: list,
    target_cols: list[str],
    group_cols: list = None,
) -> pd.DataFrame:
    """Add lag features for the specified target column.

    Since the data is in long format, lag features are created within each group
    defined by `group_cols`.

    Args:
        df (pd.DataFrame): Input dataframe. Must be a long format dataframe.
        lags (list): List of lag periods to create features for.
        target_cols (list[str]): Target columns to create lag features for.
        group_cols (list, optional): Columns to group by when creating lag features.

    Returns:
        pd.DataFrame: Dataframe with lag features added.
    """
    if group_cols is None:
        # Even for exogenous features, grouping by PC works fine
        # as the features are the same across PCs at each timestamp
        group_cols = [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]

    df = df.copy()

    for target_col in target_cols:
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(
                lag
            )

    return df


def multi_add_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str],
    group_cols: list = None,
    windows: list = None,
) -> pd.DataFrame:
    """Add rolling statistics, grouped by PC type.

    Args:
        df (pd.DataFrame): Input dataframe. Must be a long format dataframe.
        target_cols (list[str]): Target columns to create rolling features for.
        group_cols (list, optional): Columns to group by when creating rolling features.
        windows (list): List of rolling window sizes to create features for.

    Returns:
        pd.DataFrame: Dataframe with rolling features added.
    """
    if group_cols is None:
        group_cols = [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]

    df = df.copy()

    for target_col in target_cols:
        for window in windows:
            # Rolling mean
            df[f"{target_col}_rolling_mean_{window}"] = df.groupby(group_cols)[
                target_col
            ].transform(lambda x, w=window: x.rolling(window=w, min_periods=1).mean())

        # Rolling std
        df[f"{target_col}_rolling_std_{window}"] = df.groupby(group_cols)[
            target_col
        ].transform(lambda x, w=window: x.rolling(window=w, min_periods=1).std())

    return df


def multi_add_cross_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features capturing relationships between PC types.

    Args:
        df (pd.DataFrame): Input long format dataframe.

    Returns:
        pd.DataFrame with cross-series features added.
    """
    df = df.copy()

    # Compute regional average price at each timestamp
    regional_avg = df.groupby([processed_names.LONG_DATE, processed_names.LONG_REGION])[
        processed_names.LONG_PC_PRICE
    ].transform("mean")
    df[processed_names.LONG_REGIONAL_AVG_PRICE] = regional_avg

    # Compute regional price volatility (std across PC types)
    regional_std = df.groupby([processed_names.LONG_DATE, processed_names.LONG_REGION])[
        processed_names.LONG_PC_PRICE
    ].transform("std")
    df[processed_names.LONG_REGIONAL_PRICE_VOLATILITY] = regional_std

    # Deviation from regional average
    df[processed_names.LONG_PRICE_DEVIATION_FROM_REGIONAL_AVG] = (
        df[processed_names.LONG_PC_PRICE] - df[processed_names.LONG_REGIONAL_AVG_PRICE]
    )
    return df


def multi_add_pc_type_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical features describing PC type properties.

    Args:
        df (pd.DataFrame): Input long format dataframe.

    Returns:
        pd.DataFrame with PC type characteristics features added.
    """
    df = df.copy()

    # Is recycled?
    df[processed_names.LONG_PC_RECYCLED] = (
        df[processed_names.LONG_PC_TYPE].str.contains("recycled").astype(int)
    )

    # Is glass-filled? (gf, gf10, gf20)
    df[processed_names.LONG_PC_GLASS_FILLED] = (
        df[processed_names.LONG_PC_TYPE].str.contains("gf").astype(int)
    )

    # Is flame retardant?
    df[processed_names.LONG_PC_FLAME_RETARDANT] = (
        df[processed_names.LONG_PC_TYPE].str.contains("fr").astype(int)
    )

    return df


def multi_add_momentum_features(
    df: pd.DataFrame,
    target_col: str = processed_names.LONG_PC_PRICE,
    group_cols: list = None,
) -> pd.DataFrame:
    """Add momentum and acceleration features.

    Momentum = rate of change of price
    Acceleration = rate of change of momentum

    Args:
        df (pd.DataFrame): Input long format dataframe.
        target_col (str): Target column to compute momentum features on.
        group_cols (list, optional): Columns to group by when computing features.

    Returns:
        pd.DataFrame: Dataframe with momentum features added.
    """
    if group_cols is None:
        group_cols = [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]

    df = df.copy()

    # Price momentum (1st derivative)
    df[processed_names.LONG_PRICE_MOMENTUM_3M] = df.groupby(group_cols)[
        target_col
    ].transform(
        lambda x: x.diff(3)  # Change over 3 months
    )
    df[processed_names.LONG_PRICE_MOMENTUM_6M] = df.groupby(group_cols)[
        target_col
    ].transform(
        lambda x: x.diff(6)  # Change over 6 months
    )

    # Price acceleration (2nd derivative)
    df[processed_names.LONG_PRICE_ACCELERATION_3M] = df.groupby(group_cols)[
        processed_names.LONG_PRICE_MOMENTUM_3M
    ].transform(
        lambda x: x.diff(1)  # Change in momentum
    )

    return df
