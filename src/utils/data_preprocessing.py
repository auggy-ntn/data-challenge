import pandas as pd

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


def convert_pc_asia_prices(df_pc_asia: pd.DataFrame) -> pd.DataFrame:
    """Convert all Asia PC price columns to USD/Kg..

    This function processes the Asia dataset dataframe and converts all PC price
    columns to USD/Kg. It handles conversions from RMB/T, RMB/Kg, and INR/Kg to
    USD/Kg using the provided exchange rates in the dataframe.
    Specific column naming knowledge comes from an EDA of the Asia dataset and is
    hardcoded in this function. Subject to change if the raw data format changes.

    Args:
        df_pc_asia (pd.DataFrame): DataFrame containing Asia PC price data.

    Returns:
        pd.DataFrame: DataFrame with all PC price columns converted to USD/Kg.
    """
    # Dictionary to store converted columns by standardized name
    # Key: standardized column name, Value: list of (series, priority) tuples
    # Priority: 0 = USD/Kg (highest), 1 = RMB/Kg or INR/Kg, 2 = RMB/T (lowest)
    converted_cols_prioritized = {}

    for col in raw_names.PC_ASIA_PRICE_COLUMNS:
        supplier, pc_type = _get_supplier_pc_info(col)

        # Determine unit from column name
        col_lower = col.lower()
        if "rmb/t" in col_lower or "cnyperton" in col_lower:
            unit = "rmb/t"
        elif "rmb/kg" in col_lower:
            unit = "rmb/kg"
        elif "usd/kg" in col_lower:
            unit = "usd/kg"
        elif "inr/kg" in col_lower:
            unit = "inr/kg"
        else:
            # Default assumption: USD/Kg if no unit specified
            unit = "usd/kg"
            logger.info(f"Column '{col}' has no explicit unit, treating as USD/kg")

        # Handle columns without supplier name or PC type
        if supplier is None or pc_type is None:
            # Try to extract just the PC type for columns without supplier
            if " si recycled " in col_lower or col_lower.startswith("pc si recycled"):
                pc_type_only = "si recycled"
                new_col_name = f"pc {pc_type_only} (usd/kg)"
            elif " si " in col_lower or col_lower.startswith("pc si"):
                pc_type_only = "si"
                new_col_name = f"pc {pc_type_only} (usd/kg)"
            elif supplier is not None and pc_type is None:
                # Column has supplier but no PC type (e.g., "asia_supplier_4  (usd/kg)")
                # This might be a general PC price without specific type
                logger.warning(f"Column '{col}' has supplier but no PC type, skipping")
                continue
            else:
                # No supplier and couldn't parse PC type
                logger.warning(f"Skipping {col} - couldn't parse supplier/type")
                continue
        else:
            # Create standardized column name with supplier
            new_col_name = f"{supplier} pc {pc_type} (usd/kg)"

        # Convert based on unit and assign priority
        # Priority: 0 = USD/Kg (use as-is), 1 = RMB/Kg or INR/Kg, 2 = RMB/T
        if unit == "usd/kg":
            # Already in USD/Kg, just copy (highest priority)
            converted_series = df_pc_asia[col].copy()
            priority = 0
        elif unit == "rmb/t":
            # Convert from RMB/T to USD/Kg
            converted_series = (
                df_pc_asia[col] / df_pc_asia[raw_names.PC_ASIA_USD_RMB] / 1000
            )
            priority = 2  # Lowest priority
        elif unit == "rmb/kg":
            # Convert from RMB/Kg to USD/Kg
            converted_series = df_pc_asia[col] / df_pc_asia[raw_names.PC_ASIA_USD_RMB]
            priority = 1  # Medium priority
        elif unit == "inr/kg":
            # Convert from INR/Kg to USD/Kg
            converted_series = df_pc_asia[col] / df_pc_asia[raw_names.PC_ASIA_USD_INR]
            priority = 1  # Medium priority
        else:
            # Skip unknown units
            continue

        # Store with priority for later merging
        if new_col_name not in converted_cols_prioritized:
            converted_cols_prioritized[new_col_name] = []
        converted_cols_prioritized[new_col_name].append((converted_series, priority))

    # For duplicate columns (same supplier, same PC type), merge with priority
    # Priority order: USD/Kg (0) > RMB/Kg or INR/Kg (1) > RMB/T (2)
    final_cols = {}
    for col_name, series_list in converted_cols_prioritized.items():
        # Sort by priority
        series_list.sort(key=lambda x: x[1])

        # Start with the highest priority series
        result_series = series_list[0][0].copy()

        # Fill missing values with lower priority series
        for series, _priority in series_list[1:]:
            result_series = result_series.fillna(series)

        final_cols[col_name] = result_series

    # Create new dataframe starting with original columns
    df_pc_asia_converted = df_pc_asia.copy()

    # Track which original columns were converted
    converted_original_cols = []
    for col in raw_names.PC_ASIA_PRICE_COLUMNS:
        supplier, pc_type = _get_supplier_pc_info(col)

        # Try both with and without supplier names
        possible_new_names = []

        if supplier is not None and pc_type is not None:
            possible_new_names.append(f"{supplier} pc {pc_type} (usd/kg)")

        # Also check for columns without supplier (like "pc si (usd/kg)")
        if " si recycled " in col or col.startswith("pc si recycled"):
            possible_new_names.append("pc si recycled (usd/kg)")
        elif " si " in col or col.startswith("pc si"):
            possible_new_names.append("pc si (usd/kg)")

        # Check if any of the possible names exists in final_cols
        for new_col_name in possible_new_names:
            if new_col_name in final_cols:
                converted_original_cols.append(col)
                break

    # Remove only the columns that were successfully converted
    df_pc_asia_converted = df_pc_asia_converted.drop(columns=converted_original_cols)

    # Add all converted price columns (sorted for consistent column order)
    for col_name, values in sorted(final_cols.items()):
        df_pc_asia_converted[col_name] = values

    logger.info(
        f"Converted {len(final_cols)} price columns to USD/Kg "
        f"from {len(converted_original_cols)} original columns"
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
        logger.warning(f"<orange>No columns found for PC type: {pc_type}</orange>")
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
