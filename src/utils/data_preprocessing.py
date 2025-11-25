import pandas as pd

import constants.constants as cst
import constants.raw_names as raw_names
from src.utils.logger import logger


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the names of a column for a dataframe.

    This cleaning involves stripping whitespace and replacing newlines with spaces.
    Column names are also put in lower case.

    Args:
        df (pd.DataFrame): Raw dataframe with uncleaned column names.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=False).str.lower()
    return df


def _get_supplier_pc_info(col_name: str) -> tuple[str | None, str | None]:
    """Extract supplier name and PC type from column name.

    This function is specific to the Asia dataset column naming conventions.
    Should be used in the data preprocessing step for Asia dataset.

    Args:
        col_name (str): Column name to extract info from (case-insensitive).

    Returns:
        tuple[str | None, str | None]: Supplier name and PC type.
    """
    col_name_lower = col_name.lower()

    # Extract supplier name
    supplier = None
    for i in range(1, 10):
        if f"asia_supplier_{i}" in col_name_lower:
            supplier = f"asia_supplier_{i}"
            break

    # Extract PC type from the part after "pc " or between "pc " and a unit
    pc_type = None

    # Check if column has "pc " in it
    if " pc " in col_name_lower:
        # Get the part after "pc "
        after_pc = col_name_lower.split(" pc ", 1)[1]

        # Extract PC type
        for pt in [
            "gp recycled",
            "gf10fr",
            "gf recycled",
            "si recycled",
            "gp",
            "gf",
            "fr",
            "nat",
            "si",
            "gf fr",
        ]:
            # Check if the PC type appears at the start of after_pc
            if after_pc.startswith(pt + " ") or after_pc == pt:
                pc_type = pt
                break
            # Also check if it's followed by a unit in parentheses
            if after_pc.startswith(pt + "("):
                pc_type = pt
                break

    return supplier, pc_type


def _normalize_mixed_unit_column(
    series: pd.Series,
    expected_unit: str,
    threshold_rmb: float = 100.0,
    threshold_usd: float = 10.0,
) -> pd.Series:
    """Normalize columns with mixed T (ton) and KG values.

    Some columns in the raw data have inconsistent units within the same column:
    - RMB: Most values in RMB/T (10,000-30,000) but some in RMB/KG (10-100)
    - USD: Most values in USD/T (1,000-5,000) but some in USD/KG (1-10)

    Detection logic (thresholds are based on typical price ranges):
    - RMB_T: values < threshold_rmb (100) are likely RMB/KG → multiply by 1000
    - RMB_KG: values > threshold_rmb (100) are likely RMB/T → divide by 1000
    - USD_T: values < threshold_usd (10) are likely USD/KG → multiply by 1000
    - USD_KG: values > threshold_usd * 100 (1000) are likely USD/T → divide by 1000

    Args:
        series (pd.Series): Price series with potential mixed units.
        expected_unit (str): The expected unit (from constants).
        threshold_rmb (float): Threshold for RMB T/KG distinction. Default 100.
        threshold_usd (float): Threshold for USD T/KG distinction. Default 10.

    Returns:
        pd.Series: Normalized series with consistent units.
    """
    normalized = series.copy()

    if expected_unit == cst.RMB_T:
        # Values < threshold are likely RMB/KG, convert to RMB/T
        mask_wrong_unit = (series < threshold_rmb) & series.notna()
        if mask_wrong_unit.any():
            count = mask_wrong_unit.sum()
            logger.warning(
                f"Detected {count} values < {threshold_rmb} in RMB/T column - "
                f"normalizing as RMB/KG (multiplying by 1000)"
            )
            normalized.loc[mask_wrong_unit] = series.loc[mask_wrong_unit] * 1000

    elif expected_unit == cst.RMB_KG:
        # Values > threshold are likely RMB/T, convert to RMB/KG
        mask_wrong_unit = (series > threshold_rmb) & series.notna()
        if mask_wrong_unit.any():
            count = mask_wrong_unit.sum()
            logger.warning(
                f"Detected {count} values > {threshold_rmb} in RMB/KG column - "
                f"normalizing as RMB/T (dividing by 1000)"
            )
            normalized.loc[mask_wrong_unit] = series.loc[mask_wrong_unit] / 1000

    elif expected_unit == cst.USD_T:
        # Values < threshold_usd are likely USD/KG, convert to USD/T
        mask_wrong_unit = (series < threshold_usd) & series.notna()
        if mask_wrong_unit.any():
            count = mask_wrong_unit.sum()
            logger.warning(
                f"Detected {count} values < {threshold_usd} in USD/T column - "
                f"normalizing as USD/KG (multiplying by 1000)"
            )
            normalized.loc[mask_wrong_unit] = series.loc[mask_wrong_unit] * 1000

    elif expected_unit == cst.USD_KG:
        # Values > threshold_usd * 100 are likely USD/T, convert to USD/KG
        threshold_high = threshold_usd * 100
        mask_wrong_unit = (series > threshold_high) & series.notna()
        if mask_wrong_unit.any():
            count = mask_wrong_unit.sum()
            logger.warning(
                f"Detected {count} values > {threshold_high} in USD/KG column - "
                f"normalizing as USD/T (dividing by 1000)"
            )
            normalized.loc[mask_wrong_unit] = series.loc[mask_wrong_unit] / 1000

    return normalized


def convert_pc_asia_prices(df_pc_asia: pd.DataFrame) -> pd.DataFrame:
    """Convert all Asia PC price columns to USD/Kg.

    This function processes the Asia dataset dataframe and converts all PC price
    columns to USD/Kg. It handles conversions from RMB/T, RMB/Kg, USD/T, and INR/Kg
    to USD/Kg using the provided exchange rates in the dataframe and the unit
    mappings defined in constants.raw_names.PC_ASIA_PRICE_COLUMNS_UNITS.

    NOTE: Some columns have mixed units (RMB/T and RMB/KG values in the same column).
    These are automatically detected and normalized before conversion.

    When multiple columns convert to the same standardized name, the minimum value
    across those columns is kept for each row.

    Args:
        df_pc_asia (pd.DataFrame): DataFrame containing Asia PC price data.

    Returns:
        pd.DataFrame: DataFrame with all PC price columns converted to USD/Kg.
    """
    # Dictionary to store converted columns by standardized name
    # Key: standardized column name, Value: list of converted series
    converted_cols_by_name = {}
    unparsed_cols = []

    for col in raw_names.PC_ASIA_PRICE_COLUMNS:
        # Get unit from the dictionary
        if col not in raw_names.PC_ASIA_PRICE_COLUMNS_UNITS:
            logger.warning(f"Column '{col}' not found in unit dictionary, skipping")
            unparsed_cols.append(col)
            continue

        unit = raw_names.PC_ASIA_PRICE_COLUMNS_UNITS[col]

        # Extract supplier and PC type for standardized naming
        supplier, pc_type = _get_supplier_pc_info(col)

        # Handle columns without supplier name or PC type
        if supplier is None or pc_type is None:
            # Try to extract just the PC type for columns without supplier
            col_lower = col.lower()
            if " si recycled " in col_lower or col_lower.startswith("pc si recycled"):
                pc_type_only = "si recycled"
                new_col_name = f"pc {pc_type_only} (usd/kg)"
            elif " si " in col_lower or col_lower.startswith("pc si"):
                pc_type_only = "si"
                new_col_name = f"pc {pc_type_only} (usd/kg)"
            elif supplier is not None and pc_type is None:
                # Column has supplier but no PC type
                logger.warning(f"Dropping '{col}' - has supplier but no PC type")
                unparsed_cols.append(col)
                continue
            else:
                # No supplier and couldn't parse PC type
                logger.warning(f"Dropping '{col}' - couldn't parse supplier/type")
                unparsed_cols.append(col)
                continue
        else:
            # Create standardized column name with supplier
            new_col_name = f"{supplier} pc {pc_type} (usd/kg)"

        # Normalize mixed units first (for RMB and USD columns)
        series = df_pc_asia[col].copy()
        if unit in [cst.RMB_T, cst.RMB_KG, cst.USD_T, cst.USD_KG]:
            series = _normalize_mixed_unit_column(series, unit)

        # Convert based on unit
        if unit == cst.USD_KG:
            # Already in USD/Kg, just copy
            converted_series = series
        elif unit == cst.USD_T:
            # Convert from USD/T to USD/Kg
            converted_series = series / 1000
        elif unit == cst.RMB_KG:
            # Convert from RMB/Kg to USD/Kg
            converted_series = series / df_pc_asia[raw_names.PC_ASIA_USD_RMB]
        elif unit == cst.INR_KG:
            # Convert from INR/Kg to USD/Kg
            converted_series = series / df_pc_asia[raw_names.PC_ASIA_USD_INR]
        elif unit == cst.RMB_T:
            # Convert from RMB/T to USD/Kg
            converted_series = series / df_pc_asia[raw_names.PC_ASIA_USD_RMB] / 1000
        else:
            logger.warning(f"Unknown unit '{unit}' for column '{col}', skipping")
            unparsed_cols.append(col)
            continue

        # Store converted series
        if new_col_name not in converted_cols_by_name:
            converted_cols_by_name[new_col_name] = []
        converted_cols_by_name[new_col_name].append(converted_series)

    # For duplicate columns, keep the minimum value across columns
    final_cols = {}
    for col_name, series_list in converted_cols_by_name.items():
        if len(series_list) == 1:
            # Only one column, use it directly
            final_cols[col_name] = series_list[0]
        else:
            # Multiple columns, take minimum across them
            temp_df = pd.concat(series_list, axis=1)
            final_cols[col_name] = temp_df.min(axis=1)

    # Create new dataframe starting with original columns
    df_pc_asia_converted = df_pc_asia.copy()

    # Remove all original price columns that were converted
    cols_to_drop = [
        col for col in raw_names.PC_ASIA_PRICE_COLUMNS if col not in unparsed_cols
    ]
    df_pc_asia_converted = df_pc_asia_converted.drop(
        columns=cols_to_drop, errors="ignore"
    )

    # Add all converted price columns (sorted for consistent column order)
    for col_name, values in sorted(final_cols.items()):
        df_pc_asia_converted[col_name] = values

    logger.info(
        f"Converted {len(final_cols)} price columns to USD/Kg "
        f"from {len(raw_names.PC_ASIA_PRICE_COLUMNS) - len(unparsed_cols)} "
        "original columns"
    )

    if unparsed_cols:
        logger.warning(
            f"Dropped {len(unparsed_cols)} unparsed columns: {', '.join(unparsed_cols)}"
        )
        df_pc_asia_converted = df_pc_asia_converted.drop(
            columns=unparsed_cols, errors="ignore"
        )

    return df_pc_asia_converted


def compute_best_price_asia(df: pd.DataFrame, pc_type: str) -> pd.Series:
    """Compute best (minimum) price for a given PC type across all suppliers.

    This function finds all columns matching the specified PC type and returns
    the minimum price across all suppliers for each row.

    Args:
        df (pd.DataFrame): DataFrame containing Asia PC price data with standardized
            column names in format "asia_supplier_X pc TYPE (usd/kg)".
        pc_type (str): The PC type to compute best price for (e.g., "gp", "fr", "gf").

    Returns:
        pd.Series: Series containing the best (minimum) price for the PC type.
    """
    # Find all columns matching this PC type (case-insensitive)
    # Need to be careful with substring matching
    # (e.g., "gp" shouldn't match "gp recycled")
    pc_type_lower = pc_type.lower()
    matching_cols = []

    for col in df.columns:
        col_lower = col.lower()
        if "(usd/kg)" not in col_lower:
            continue

        # Check if this column matches the PC type
        # Extract the PC type part from column name
        # like "asia_supplier_1 pc gp (usd/kg)"
        if " pc " not in col_lower and not col_lower.startswith("pc "):
            continue

        # Get the part after "pc " and before "(usd/kg)"
        if " pc " in col_lower:
            pc_part = col_lower.split(" pc ")[1].split("(usd/kg)")[0].strip()
        else:
            pc_part = col_lower.split("pc ")[1].split("(usd/kg)")[0].strip()

        # Exact match for the PC type
        if pc_part == pc_type_lower:
            matching_cols.append(col)

    if not matching_cols:
        logger.warning(f"No columns found for PC type: {pc_type}")
        return pd.Series([pd.NA] * len(df), index=df.index)

    logger.info(
        f"Computing best price for '{pc_type}' from {len(matching_cols)} columns"
    )
    return df[matching_cols].min(axis=1)


def extract_bpa_capacity_loss(
    phenol_df: pd.DataFrame, derivative_col: str = "Derivative"
) -> pd.DataFrame:
    """Extract and transform BPA capacity loss data from phenol dataset.

    Transforms data from wide format (derivatives as rows, dates as columns) to
    long format (time series). Filters for BPA derivative only, removes non-date
    columns, and converts to a clean time series format.

    Args:
        phenol_df (pd.DataFrame): Raw phenol capacity loss dataframe in wide format.
        derivative_col (str): Name of the column containing derivative names.

    Returns:
        pd.DataFrame: Transformed BPA capacity loss data in long format with columns:
            - date: datetime column
            - bpa_capacity_loss_kt: capacity loss values in kilotons
    """
    import constants.intermediate_names as intermediate_names

    logger.info("Extracting BPA capacity loss data from phenol dataset")

    # Extract BPA row from phenol dataset
    bpa_row = phenol_df[phenol_df[derivative_col] == "BISPHENOL A"].copy()

    if bpa_row.empty:
        logger.error("BPA derivative not found in phenol dataset")
        raise ValueError("BISPHENOL A not found in phenol dataset")

    # Keep only date columns (filter out Derivative, conversion factors, etc.)
    date_columns = [
        col
        for col in bpa_row.columns
        if col != derivative_col
        and not col.startswith("Avg.")
        and not col.lower().startswith("unnamed")
    ]

    logger.info(f"Found {len(date_columns)} date columns")
    bpa_row = bpa_row[date_columns]

    # Reshape from wide to long format (transpose)
    bpa_long = bpa_row.T.reset_index()
    bpa_long.columns = [
        intermediate_names.BPA_DATE,
        intermediate_names.BPA_CAPACITY_LOSS,
    ]

    # Convert date column to datetime (dates are in format "Mon YYYY" like "Jul 2017")
    bpa_long[intermediate_names.BPA_DATE] = pd.to_datetime(
        bpa_long[intermediate_names.BPA_DATE], format="%b %Y"
    )

    # Sort by date
    bpa_long = bpa_long.sort_values(by=intermediate_names.BPA_DATE).reset_index(
        drop=True
    )

    # Remove rows with missing capacity loss values
    initial_count = len(bpa_long)
    bpa_long = bpa_long.dropna(subset=[intermediate_names.BPA_CAPACITY_LOSS])
    removed_count = initial_count - len(bpa_long)

    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with missing capacity loss values")

    logger.info(
        f"Processed BPA data: {len(bpa_long)} monthly observations from "
        f"{bpa_long[intermediate_names.BPA_DATE].min().strftime('%Y-%m')} to "
        f"{bpa_long[intermediate_names.BPA_DATE].max().strftime('%Y-%m')}"
    )

    return bpa_long


def aggregate_shutdowns_by_month(
    df_shutdown: pd.DataFrame, chemical_name: str
) -> pd.DataFrame:
    """Aggregate shutdown events by month for a given chemical.

    For each shutdown event, distribute the capacity loss across the months
    it spans, proportional to the number of days in each month.

    Args:
        df_shutdown: Shutdown dataframe with outage events.
        chemical_name: Name of the chemical (e.g., "acetone", "phenol").

    Returns:
        DataFrame with monthly aggregated capacity loss with columns:
            - date: datetime column
            - chemical: chemical name
            - capacity_loss_kt: capacity loss values in kilotons
    """
    import constants.intermediate_names as intermediate_names

    monthly_losses = []

    for _, row in df_shutdown.iterrows():
        start_date = row[raw_names.SHUTDOWN_OUTAGE_START_DATE]
        end_date = row[raw_names.SHUTDOWN_OUTAGE_END_DATE]
        total_loss = row[raw_names.SHUTDOWN_TOTAL_CAPACITY_LOSS]

        # Skip if dates or loss are invalid
        if pd.isna(start_date) or pd.isna(end_date) or pd.isna(total_loss):
            continue

        # Convert to datetime if not already
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date, errors="coerce")
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date, errors="coerce")

        if pd.isna(start_date) or pd.isna(end_date):
            continue

        # Calculate total days
        total_days = (
            end_date - start_date
        ).days + 1  # +1 to include both start and end

        if total_days <= 0:
            continue

        # Generate all dates in the outage period
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Distribute capacity loss across days
        daily_loss = total_loss / total_days

        # Group by year-month and aggregate
        for date in date_range:
            year_month = date.to_period("M")
            monthly_losses.append(
                {
                    "date": year_month.to_timestamp(),
                    "capacity_loss_kt": daily_loss,
                }
            )

    # Convert to DataFrame and aggregate by month
    if not monthly_losses:
        return pd.DataFrame(
            columns=[
                intermediate_names.SHUTDOWN_DATE,
                intermediate_names.SHUTDOWN_CHEMICAL,
                intermediate_names.SHUTDOWN_CAPACITY_LOSS,
            ]
        )

    df_monthly = pd.DataFrame(monthly_losses)

    # Aggregate by month (sum daily losses)
    monthly_agg = (
        df_monthly.groupby(intermediate_names.SHUTDOWN_DATE)[
            intermediate_names.SHUTDOWN_CAPACITY_LOSS
        ]
        .sum()
        .reset_index()
    )

    # Sort by date
    monthly_agg = monthly_agg.sort_values(intermediate_names.SHUTDOWN_DATE).reset_index(
        drop=True
    )

    # Add chemical column
    monthly_agg[intermediate_names.SHUTDOWN_CHEMICAL] = chemical_name

    return monthly_agg
