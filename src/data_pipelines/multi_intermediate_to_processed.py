# Intermediate to Processed Data Pipeline (for multivariate modeling)

import pandas as pd

from constants import processed_names
import constants.constants as cst
import constants.intermediate_names as intermediate_names
import constants.paths as pth
import src.utils.feature_engineering as fe_utils


def build_multivariate_dataset(
    horizon: int,
    group_by_pc_types: bool,
    include_exogenous: bool = True,
) -> pd.DataFrame:
    """Complete pipeline for multivariate modeling.

    Creates long format dataset with all features needed for training
    multivariate models across all (pc_type, region) combinations.

    Args:
        horizon: Forecast horizon in months (3, 6, or 9)
        group_by_pc_types: Whether PC prices are grouped by pc_types
        include_exogenous: Whether to add exogenous variables

    Returns:
        Long format dataframe ready for multivariate modeling with columns:
        - date: Standardized date column
        - region: Region identifier
        - pc_type: PC type identifier
        - pc_price: PC type prices
        - Cross PC type features: regional_avg_price, regional_price_volatility,
                                  price_deviation_from_regional_avg
        - PC characteristics: is_recycled, is_glass_filled, is_flame_retardant
        - Calendar features: month_sin, month_cos, quarter_sin, quarter_cos, year,
                             time_idx
        - Exogenous features: electricity, capacity loss, shutdowns, exchange rates
    """
    # 1. Create base long format dataset
    long_df = fe_utils.create_long_format(group_by_pc_types=group_by_pc_types)
    future_preds = long_df[long_df[processed_names.LONG_DATE] >= cst.CUTOFF_DATE].copy()
    long_df = long_df[long_df[processed_names.LONG_DATE] < cst.CUTOFF_DATE].copy()
    future_preds = future_preds[
        [
            processed_names.LONG_DATE,
            processed_names.LONG_REGION,
            processed_names.LONG_PC_TYPE,
            processed_names.LONG_PC_PRICE,
        ]
    ]
    # Save future predictions for later use
    if group_by_pc_types:
        future_preds.to_csv(
            pth.SE_PREDICTIONS_DATA_DIR / "se_predictions_multi_grouped.csv",
            index=False,
        )
    else:
        future_preds.to_csv(
            pth.SE_PREDICTIONS_DATA_DIR / "se_predictions_multi.csv", index=False
        )

    # 2. Date column
    date_col = processed_names.LONG_DATE
    long_df = long_df.sort_values(
        [
            processed_names.LONG_REGION,
            processed_names.LONG_PC_TYPE,
            processed_names.LONG_DATE,
        ]
    ).reset_index(drop=True)

    # Validate no duplicate dates within groups
    duplicates = long_df.groupby(
        [
            processed_names.LONG_REGION,
            processed_names.LONG_PC_TYPE,
            processed_names.LONG_DATE,
        ]
    ).size()
    if (duplicates > 1).any():
        duplicate_groups = duplicates[duplicates > 1]
        raise ValueError(
            "Duplicate dates found within (region, pc_type) groups:"
            f"\n{duplicate_groups.head()}"
        )

    # 3. Add calendar features
    long_df = fe_utils.multi_add_time_features(long_df, date_col=date_col)

    # 4. Horizon-specific configuration
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
    else:
        raise ValueError(f"Unsupported horizon: {horizon}. Must be 3, 6, or 9.")

    # 5. Add temporal features for PC prices
    long_df = long_df.sort_values(
        [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE, date_col]
    ).reset_index(drop=True)
    long_df = fe_utils.multi_add_lag_features(
        df=long_df, target_cols=[processed_names.LONG_PC_PRICE], lags=lags
    )
    long_df = fe_utils.multi_add_rolling_features(
        df=long_df, target_cols=[processed_names.LONG_PC_PRICE], windows=windows
    )

    # 6. Add cross PC type features
    long_df = fe_utils.multi_add_cross_series_features(long_df)

    # 7. Add PC characteristics
    long_df = fe_utils.multi_add_pc_type_characteristics(long_df)

    # 8. Add temporal features for exogenous variables
    if include_exogenous:
        # Add lags for exogenous variables
        long_df = long_df.sort_values(
            [processed_names.LONG_REGION, processed_names.LONG_PC_TYPE, date_col]
        ).reset_index(drop=True)
        long_df = fe_utils.multi_add_lag_features(
            df=long_df,
            target_cols=intermediate_names.EXOGENOUS_COLUMNS,
            group_cols=[processed_names.LONG_REGION, processed_names.LONG_PC_TYPE],
            lags=lags[:3],
        )

    return long_df.sort_values(
        by=[date_col, processed_names.LONG_REGION, processed_names.LONG_PC_TYPE]
    ).reset_index(drop=True)


if __name__ == "__main__":
    # Multivariate datasets without grouping by pc_types
    df_3m = build_multivariate_dataset(horizon=3, group_by_pc_types=False)
    df_6m = build_multivariate_dataset(horizon=6, group_by_pc_types=False)
    df_9m = build_multivariate_dataset(horizon=9, group_by_pc_types=False)

    df_3m.to_csv(pth.PROCESSED_DATA_DIR / "multi_3m.csv", index=False)
    df_6m.to_csv(pth.PROCESSED_DATA_DIR / "multi_6m.csv", index=False)
    df_9m.to_csv(pth.PROCESSED_DATA_DIR / "multi_9m.csv", index=False)

    # Multivariate datasets with PC prices grouped by pc_types
    df_3m_grouped = build_multivariate_dataset(horizon=3, group_by_pc_types=True)
    df_6m_grouped = build_multivariate_dataset(horizon=6, group_by_pc_types=True)
    df_9m_grouped = build_multivariate_dataset(horizon=9, group_by_pc_types=True)

    df_3m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "multi_3m_grouped.csv", index=False)
    df_6m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "multi_6m_grouped.csv", index=False)
    df_9m_grouped.to_csv(pth.PROCESSED_DATA_DIR / "multi_9m_grouped.csv", index=False)
