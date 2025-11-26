# Intermediate to Processed Data Pipeline (for univariate modeling)


import pandas as pd

from constants import intermediate_names, processed_names
import constants.constants as cst
import constants.paths as pth
import src.utils.feature_engineering as fe_utils


def build_univariate_dataset(
    horizon: int,
    group_by_pc_types: bool,
    include_exogenous: bool = True,
    include_differencing: bool = True,
) -> pd.DataFrame:
    """Complete pipeline for univariate modeling.

    Creates wide format dataset with all features needed for training
    separate models for each (pc_type, region, horizon) combination.

    Args:
        horizon: Forecast horizon in months (3, 6, or 9)
        group_by_pc_types: Whether PC prices are grouped by pc_types
        include_exogenous: Whether to add exogenous variables
        include_differencing: Whether to add differencing features (

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
    wide_df = fe_utils.create_wide_format(group_by_pc_types=group_by_pc_types)
    future_preds = wide_df[wide_df[processed_names.WIDE_DATE] >= cst.CUTOFF_DATE].copy()
    wide_df = wide_df[wide_df[processed_names.WIDE_DATE] < cst.CUTOFF_DATE].copy()
    if group_by_pc_types:
        future_preds = future_preds[
            [processed_names.WIDE_DATE] + intermediate_names.GROUPED_ENDOGENOUS_COLUMNS
        ]
        # Save future predictions for later use
        future_preds.to_csv(
            pth.SE_PREDICTIONS_DATA_DIR / "se_predictions_uni_grouped.csv", index=False
        )
    else:
        future_preds = future_preds[
            [processed_names.WIDE_DATE] + intermediate_names.BASE_ENDOGENOUS_COLUMNS
        ]
        # Save future predictions for later use
        future_preds.to_csv(
            pth.SE_PREDICTIONS_DATA_DIR / "se_predictions_uni.csv", index=False
        )

    # 2. Date column
    date_col = processed_names.WIDE_DATE

    # 3. Add calendar features
    wide_df = fe_utils.uni_add_time_features(wide_df, date_col=date_col)

    # 4. Get list of PC price columns
    if group_by_pc_types:
        pc_price_cols = intermediate_names.GROUPED_ENDOGENOUS_COLUMNS
    else:
        pc_price_cols = intermediate_names.BASE_ENDOGENOUS_COLUMNS

    # 5. Horizon-specific configuration
    # For direct forecasting, we can use all historical lags including recent ones
    # Recent lags (1, 2, 3) provide the most predictive power
    if horizon == cst.HORIZON_3_MONTHS:
        lags = [
            cst.LAG_1_MONTH,
            cst.LAG_2_MONTHS,
            cst.LAG_3_MONTHS,
            cst.LAG_6_MONTHS,
            cst.LAG_9_MONTHS,
            cst.LAG_12_MONTHS,
        ]
        windows = [
            cst.ROLLING_WINDOW_3_MONTHS,
            cst.ROLLING_WINDOW_6_MONTHS,
            cst.ROLLING_WINDOW_12_MONTHS,
        ]
        roc_periods = [cst.ROC_PERIOD_3_MONTHS, cst.ROC_PERIOD_6_MONTHS]
    elif horizon == cst.HORIZON_6_MONTHS:
        lags = [
            cst.LAG_1_MONTH,
            cst.LAG_2_MONTHS,
            cst.LAG_3_MONTHS,
            cst.LAG_6_MONTHS,
            cst.LAG_12_MONTHS,
            cst.LAG_18_MONTHS,
        ]
        windows = [
            cst.ROLLING_WINDOW_3_MONTHS,
            cst.ROLLING_WINDOW_6_MONTHS,
            cst.ROLLING_WINDOW_12_MONTHS,
        ]
        roc_periods = [cst.ROC_PERIOD_6_MONTHS, cst.ROC_PERIOD_9_MONTHS]
    elif horizon == cst.HORIZON_9_MONTHS:
        lags = [
            cst.LAG_1_MONTH,
            cst.LAG_2_MONTHS,
            cst.LAG_3_MONTHS,
            cst.LAG_6_MONTHS,
            cst.LAG_9_MONTHS,
            cst.LAG_12_MONTHS,
            cst.LAG_18_MONTHS,
            cst.LAG_24_MONTHS,
        ]
        windows = [
            cst.ROLLING_WINDOW_6_MONTHS,
            cst.ROLLING_WINDOW_12_MONTHS,
            cst.ROLLING_WINDOW_24_MONTHS,
        ]
        roc_periods = [cst.ROC_PERIOD_9_MONTHS, cst.ROC_PERIOD_12_MONTHS]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be 3, 6, or 9.")

    # 6. Add temporal features for PC prices
    wide_df = wide_df.sort_values(by=date_col).reset_index(drop=True)
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
        wide_df = wide_df.sort_values(by=date_col).reset_index(drop=True)
        wide_df = fe_utils.uni_add_lag_features(
            wide_df, lags=lags[:2], target_cols=intermediate_names.EXOGENOUS_COLUMNS
        )  # Fewer lags for exog
        wide_df = fe_utils.uni_add_rolling_features(
            wide_df,
            window_sizes=windows,
            target_cols=intermediate_names.EXOGENOUS_COLUMNS,
        )

    return wide_df.sort_values(by=date_col).reset_index(drop=True)


if __name__ == "__main__":
    # Univariate datasets without grouping by pc_types
    df_3m = build_univariate_dataset(horizon=3, group_by_pc_types=False)
    df_6m = build_univariate_dataset(horizon=6, group_by_pc_types=False)
    df_9m = build_univariate_dataset(horizon=9, group_by_pc_types=False)

    df_3m.to_csv(pth.PROCESSED_DATA_DIR / "uni_3m.csv", index=False)
    df_6m.to_csv(pth.PROCESSED_DATA_DIR / "uni_6m.csv", index=False)
    df_9m.to_csv(pth.PROCESSED_DATA_DIR / "uni_9m.csv", index=False)

    # Univariate datasets with PC prices grouped by pc_types
    df_3m_grouped = build_univariate_dataset(horizon=3, group_by_pc_types=True)
    df_6m_grouped = build_univariate_dataset(horizon=6, group_by_pc_types=True)
    df_9m_grouped = build_univariate_dataset(horizon=9, group_by_pc_types=True)

    df_3m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "uni_3m_grouped.csv", index=False)
    df_6m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "uni_6m_grouped.csv", index=False)
    df_9m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "uni_9m_grouped.csv", index=False)
