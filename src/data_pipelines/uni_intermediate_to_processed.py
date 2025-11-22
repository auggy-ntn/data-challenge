# Intermediate to Processed Data Pipeline (for univariate modeling)

import pandas as pd

from constants import intermediate_names, processed_names
import src.utils.feature_engineering as fe_utils


def build_univariate_dataset(
    horizon: int,
    include_exogenous: bool = True,
    include_differencing: bool = False,  # Only needed for ARIMA
) -> pd.DataFrame:
    """Complete pipeline for univariate modeling.

    Creates wide format dataset with all features needed for training
    separate models for each (pc_type, region, horizon) combination.

    Args:
        horizon: Forecast horizon in months (3, 6, or 9)
        include_exogenous: Whether to add exogenous variables
        include_differencing: Whether to add differencing features (for ARIMA)

    Returns:
        Wide format dataframe ready for univariate modeling with columns:
        - date: Standardized date column
        - eu_pc_*_best_price: EU PC type prices
        - asia_pc_*_best_price: Asia PC type prices
        - *_lag_*: Lag features for each PC type
        - *_roll_mean_*: Rolling mean features
        - *_roll_std_*: Rolling std features
        - *_expanding_*: Expanding window features
        - *_roc_*: Rate of change features
        - *_diff_* (optional): Differencing features
        - Exogenous features: electricity, capacity loss, shutdowns, exchange rates
        - Calendar features: month_sin, month_cos, quarter_sin, quarter_cos, year,
                             time_idx
    """
    # 1. Create base wide format with exogenous variables
    wide_df = fe_utils.create_wide_format()

    # 2. Date column
    date_col = processed_names.WIDE_DATE

    # 3. Add calendar features
    wide_df = fe_utils.uni_add_time_features(wide_df, date_col=date_col)

    # 4. Get list of PC price columns
    pc_price_cols = intermediate_names.ENDOGENOUS_COLUMNS

    # 5. Horizon-specific configuration
    # Avoid lookahead bias (i.e. lags/windows should be >= horizon)
    if horizon == 3:
        lags = [3, 6, 9]
        windows = [3, 6]
        roc_periods = [3, 6]
    elif horizon == 6:
        lags = [6, 12, 18]
        windows = [6, 12]
        roc_periods = [6, 9]
    elif horizon == 9:
        lags = [9, 12, 18, 24]
        windows = [12, 24]
        roc_periods = [9, 12]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be 3, 6, or 9.")

    # 6. Add temporal features for PC prices
    wide_df = fe_utils.uni_add_lag_features(
        wide_df, lags=lags, target_cols=pc_price_cols
    )
    wide_df = fe_utils.uni_add_rolling_features(
        wide_df, window_sizes=windows, target_cols=pc_price_cols
    )
    wide_df = fe_utils.uni_add_expanding_features(wide_df, target_cols=pc_price_cols)
    wide_df = fe_utils.uni_add_rate_of_change_features(
        wide_df, periods=roc_periods, target_cols=pc_price_cols
    )

    if include_differencing:
        wide_df = fe_utils.uni_add_differencing_features(
            wide_df, periods=[1, 12], target_cols=pc_price_cols
        )

    # 7. Add temporal features for exogenous variables
    if include_exogenous:
        # Add lags for exogenous variables
        wide_df = fe_utils.uni_add_lag_features(
            wide_df, lags=lags[:2], target_cols=intermediate_names.EXOGENOUS_COLUMNS
        )  # Fewer lags for exog
        wide_df = fe_utils.uni_add_rolling_features(
            wide_df,
            window_sizes=windows,
            target_cols=intermediate_names.EXOGENOUS_COLUMNS,
        )

    return wide_df
