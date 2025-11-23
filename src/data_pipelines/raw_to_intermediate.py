# Contains the functions defining the raw to intermediate data pipelines

from pathlib import Path

import pandas as pd

import constants.constants as cst
import constants.intermediate_names as intermediate_names
import constants.paths as pth
import constants.raw_names as raw_names
from src.utils.data_preprocessing import (
    clean_column_names,
    compute_best_price_asia,
    convert_pc_asia_prices,
    extract_bpa_capacity_loss,
)
from src.utils.logger import logger


def raw_to_intermediate_phenol_acetone_bpa(
    phenol_df: pd.DataFrame, acetone_df: pd.DataFrame
) -> None:
    """Transforms raw phenol and acetone capacity loss data to intermediate BPA format.

    This function extracts BPA (Bisphenol A) capacity loss data from both phenol and
    acetone datasets. Since BPA production uses both chemicals with 0.999 correlation,
    we extract only phenol BPA data to avoid redundancy. The data is transformed from
    wide format (derivatives as rows, dates as columns) to long format (time series).

    Args:
        phenol_df (pd.DataFrame): Raw phenol capacity loss dataframe in wide format.
        acetone_df (pd.DataFrame): Raw acetone capacity loss dataframe in wide format.
    """
    # Extract and transform BPA data using utility function
    bpa_long = extract_bpa_capacity_loss(
        phenol_df, derivative_col=raw_names.PHENOL_DERIVATIVE
    )

    # Save intermediate dataframe to CSV
    logger.info(
        "Saving intermediate BPA capacity loss dataset at "
        f"{pth.INTERMEDIATE_PHENOL_ACETONE_DIR / 'intermediate_bpa_capacity_loss.csv'}"
    )
    bpa_long.to_csv(
        pth.INTERMEDIATE_PHENOL_ACETONE_DIR / "intermediate_bpa_capacity_loss.csv",
        index=False,
    )


def raw_to_intermediate_pc_price_eu(pc_price_eu: pd.DataFrame) -> None:
    """Transforms the raw pc_price Europe dataset to intermediate format.

    This function cleans the column names of the raw dataframe, formats the date column
    to datetime, drops unnamed columns and empty columns, columns ending with "%" and
    groups the PC by type to compute the best price for each PC type across suppliers.
    The resulting dataframe is saved to a CSV file.

    Args:
        pc_price_eu (pd.DataFrame): Raw pc_price Europe dataframe.
    """
    # There are 2 date columns ("Date" and "date"), keep only the first one
    pc_price_eu = pc_price_eu.drop(columns=["date"])

    # Clean column names
    logger.info("Cleaning column names")
    intermediate_df = clean_column_names(pc_price_eu).copy()

    # Format date column to datetime
    intermediate_df[raw_names.PC_EU_DATE] = pd.to_datetime(
        intermediate_df[raw_names.PC_EU_DATE], format=cst.DATE_FORMAT
    )

    # Drop columns that are unnamed
    unamed_columns = [
        raw_names.PC_EU_UNNAMED_11,
        raw_names.PC_EU_UNNAMED_12,
        raw_names.PC_EU_UNNAMED_49,
    ]
    logger.info(f"Dropping unnamed columns: {unamed_columns}")
    intermediate_df = intermediate_df.drop(columns=unamed_columns, errors="ignore")

    # Drop columns that contain only NaN values
    logger.info("Dropping columns with all NaN values")
    intermediate_df = intermediate_df.dropna(axis=1, how="all")

    # Drop columns ending with "%" (as recommended by client)
    pct_columns = [
        raw_names.PC_EU_SUPPLIER_1_CRYSTAL_PCT,
        raw_names.PC_EU_SUPPLIER_2_CRYSTAL_PCT,
        raw_names.PC_EU_SUPPLIER_3_CRYSTAL_PCT,
        raw_names.PC_EU_SUPPLIER_4_CRYSTAL_PCT,
        raw_names.PC_EU_SUPPLIER_5_CRYSTAL_PCT,
        raw_names.PC_EU_SUPPLIER_3_GF10_PCT,
        raw_names.PC_EU_SUPPLIER_4_GF10_PCT,
    ]
    logger.info(f"Dropping percentage columns: {pct_columns}")
    intermediate_df = intermediate_df.drop(columns=pct_columns, errors="ignore")

    # Group by PC type and compute best price (minimum across suppliers) for each date
    # and PC type
    logger.info("Computing best prices for each PC type")
    intermediate_df[intermediate_names.PC_EU_PC_CRYSTAL_BEST_PRICE] = intermediate_df[
        raw_names.PC_EU_CRYSTAL_COLUMNS
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_WHITE_BEST_PRICE] = intermediate_df[
        raw_names.PC_EU_WHITE_COLUMNS
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_GF10_BEST_PRICE] = intermediate_df[
        raw_names.PC_EU_GF_10_COLUMNS
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_GF20_BEST_PRICE] = intermediate_df[
        raw_names.PC_EU_GF20_COLUMNS
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE] = (
        intermediate_df[raw_names.PC_EU_RECYCLED_WHITE_COLUMNS].min(axis=1)
    )

    intermediate_df[intermediate_names.PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE] = (
        intermediate_df[raw_names.PC_EU_RECYCLED_GREY_COLUMNS].min(axis=1)
    )

    intermediate_df[intermediate_names.PC_EU_PC_SI_BEST_PRICE] = intermediate_df[
        raw_names.PC_EU_SI_COLUMNS
    ].min(axis=1)

    # Drop individual supplier columns to keep only best price columns
    supplier_columns = [
        col
        for col in intermediate_df.columns
        if col
        not in [
            raw_names.PC_EU_DATE,
            intermediate_names.PC_EU_PC_CRYSTAL_BEST_PRICE,
            intermediate_names.PC_EU_PC_WHITE_BEST_PRICE,
            intermediate_names.PC_EU_PC_GF10_BEST_PRICE,
            intermediate_names.PC_EU_PC_GF20_BEST_PRICE,
            intermediate_names.PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE,
            intermediate_names.PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE,
            intermediate_names.PC_EU_PC_SI_BEST_PRICE,
        ]
        + raw_names.PC_EU_REFERENCE_COLUMNS
        + raw_names.PC_EU_SPREAD_COLUMNS
    ]
    logger.info(f"Dropping individual supplier columns: {supplier_columns}")
    intermediate_df = intermediate_df.drop(columns=supplier_columns, errors="ignore")

    # Save intermediate dataframe to CSV
    logger.info(
        "Saving intermediate pc_price Europe dataset at "
        f"{pth.INTERMEDIATE_PC_PRICE_DIR / 'intermediate_pc_price_eu.csv'}"
    )
    intermediate_df.to_csv(
        pth.INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_eu.csv", index=False
    )


def raw_to_intermediate_pc_price_asia(
    pc_price_asia: pd.DataFrame, conversion_rates: pd.DataFrame
) -> None:
    """Transforms the raw pc_price Asia dataset to intermediate format.

    This function cleans the column names of the raw dataframe and saves the
    resulting intermediate dataframe to a CSV file.

    Args:
        pc_price_asia (pd.DataFrame): Raw pc_price Asia dataframe.
        conversion_rates (pd.DataFrame): Currency conversion rates dataframe.
    """
    # There are 2 date columns ("Date" and "date"), drop the lowercase one
    # before cleaning to avoid duplicate column names
    if "date" in pc_price_asia.columns:
        pc_price_asia = pc_price_asia.drop(columns=["date"])

    # Clean column names
    logger.info("Cleaning column names")
    intermediate_df = clean_column_names(pc_price_asia).copy()

    # Also drop year and month columns if they exist (since we have a date column)
    year_month_columns = [raw_names.PC_ASIA_YEAR, raw_names.PC_ASIA_MONTH]
    intermediate_df = intermediate_df.drop(columns=year_month_columns, errors="ignore")

    # Format date column to datetime
    intermediate_df[raw_names.PC_ASIA_DATE] = pd.to_datetime(
        intermediate_df[raw_names.PC_ASIA_DATE], format=cst.DATE_FORMAT
    )

    # Drop columns that are fully empty
    logger.info("Dropping columns with all NaN values")
    intermediate_df = intermediate_df.dropna(axis=1, how="all")

    # Complete missing conversion rates using the provided conversion rates dataframe
    logger.info("Completing missing conversion rates")
    # RMB to USD conversion
    conversion_map_rmb = conversion_rates.set_index(raw_names.DEXCHUS_OBSERVATION_DATE)[
        raw_names.DEXCHUS_VALUE
    ].to_dict()
    mask = intermediate_df[raw_names.PC_ASIA_USD_RMB].isna() | (
        intermediate_df[raw_names.PC_ASIA_USD_RMB] == 0
    )
    intermediate_df.loc[mask, raw_names.PC_ASIA_USD_RMB] = intermediate_df.loc[
        mask, raw_names.PC_ASIA_DATE
    ].map(conversion_map_rmb)

    # For remaining missing values, we use forward fill method
    # and backward fill as a last resort
    intermediate_df = intermediate_df.sort_values(
        by=raw_names.PC_ASIA_DATE
    ).reset_index(drop=True)
    intermediate_df[raw_names.PC_ASIA_USD_RMB] = (
        intermediate_df[raw_names.PC_ASIA_USD_RMB]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # INR to USD conversion
    conversion_map_inr = conversion_rates.set_index(raw_names.DEXINUS_OBSERVATION_DATE)[
        raw_names.DEXINUS_VALUE
    ].to_dict()
    intermediate_df[raw_names.PC_ASIA_USD_INR] = intermediate_df[
        raw_names.PC_ASIA_DATE
    ].map(conversion_map_inr)
    intermediate_df = intermediate_df.sort_values(
        by=raw_names.PC_ASIA_DATE
    ).reset_index(drop=True)
    intermediate_df[raw_names.PC_ASIA_USD_INR] = (
        intermediate_df[raw_names.PC_ASIA_USD_INR]
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # Apply currency conversion to prices in RMB and INR to get all prices in USD
    logger.info("Applying currency conversion to supplier prices")
    intermediate_df = convert_pc_asia_prices(intermediate_df)

    # At this point, all supplier price columns should be in USD/Kg with standardized
    # names.
    # Group by PC type, compute best price (minimum across suppliers) for each PC type
    logger.info("Computing best prices for each PC type")

    # Compute best prices for all PC types
    intermediate_df[intermediate_names.PC_ASIA_GP_BEST_PRICE] = compute_best_price_asia(
        intermediate_df, "gp"
    )
    intermediate_df[intermediate_names.PC_ASIA_GP_RECYCLED_BEST_PRICE] = (
        compute_best_price_asia(intermediate_df, "gp recycled")
    )
    intermediate_df[intermediate_names.PC_ASIA_FR_BEST_PRICE] = compute_best_price_asia(
        intermediate_df, "fr"
    )
    intermediate_df[intermediate_names.PC_ASIA_GF_BEST_PRICE] = compute_best_price_asia(
        intermediate_df, "gf"
    )
    intermediate_df[intermediate_names.PC_ASIA_GF_RECYCLED_BEST_PRICE] = (
        compute_best_price_asia(intermediate_df, "gf recycled")
    )
    intermediate_df[intermediate_names.PC_ASIA_NAT_BEST_PRICE] = (
        compute_best_price_asia(intermediate_df, "nat")
    )
    intermediate_df[intermediate_names.PC_ASIA_SI_BEST_PRICE] = compute_best_price_asia(
        intermediate_df, "si"
    )
    intermediate_df[intermediate_names.PC_ASIA_SI_RECYCLED_BEST_PRICE] = (
        compute_best_price_asia(intermediate_df, "si recycled")
    )

    # Drop individual supplier columns to keep only best price columns
    supplier_columns = [
        col for col in intermediate_df.columns if "(usd/kg)" in col.lower()
    ]
    logger.info(f"Dropping {len(supplier_columns)} individual supplier columns")
    intermediate_df = intermediate_df.drop(columns=supplier_columns, errors="ignore")

    # Save intermediate dataframe to CSV
    logger.info(
        "Saving intermediate pc_price Asia dataset at "
        f"{pth.INTERMEDIATE_PC_PRICE_DIR / 'intermediate_pc_price_asia.csv'}"
    )
    intermediate_df.to_csv(
        pth.INTERMEDIATE_PC_PRICE_DIR / "intermediate_pc_price_asia.csv", index=False
    )


def raw_to_intermediate_electricity_price(electricity_price: pd.DataFrame) -> None:
    """Transforms the raw electricity dataset to intermediate format.

    Standardizes column names, parses dates, coerces prices to numeric, drops invalid
    rows, and writes the cleaned dataset.

    Args:
        electricity_price (pd.DataFrame): Raw electricity price dataframe.
    """
    logger.info("Cleaning column names")
    intermediate_df = clean_column_names(electricity_price).copy()

    # Standardize column names to snake_case conventions used elsewhere
    rename_map = {
        raw_names.ELECTRICITY_COUNTRY: intermediate_names.ELECTRICITY_COUNTRY,
        raw_names.ELECTRICITY_ISO3_CODE: intermediate_names.ELECTRICITY_ISO3_CODE,
        raw_names.ELECTRICITY_DATE: intermediate_names.ELECTRICITY_DATE,
        raw_names.ELECTRICITY_PRICE_EUR_MWHE: (
            intermediate_names.ELECTRICITY_PRICE_EUR_MWHE
        ),
    }
    intermediate_df = intermediate_df.rename(columns=rename_map)

    # Parse date column
    intermediate_df[intermediate_names.ELECTRICITY_DATE] = pd.to_datetime(
        intermediate_df[intermediate_names.ELECTRICITY_DATE],
        format=cst.DATE_FORMAT,
        errors="coerce",
    )

    # Ensure price is numeric
    intermediate_df[intermediate_names.ELECTRICITY_PRICE_EUR_MWHE] = pd.to_numeric(
        intermediate_df[intermediate_names.ELECTRICITY_PRICE_EUR_MWHE], errors="coerce"
    )

    # Drop rows with invalid date or price
    intermediate_df = intermediate_df.dropna(
        subset=[
            intermediate_names.ELECTRICITY_DATE,
            intermediate_names.ELECTRICITY_PRICE_EUR_MWHE,
        ]
    )

    # Sort for deterministic output
    intermediate_df = intermediate_df.sort_values(
        by=[intermediate_names.ELECTRICITY_COUNTRY, intermediate_names.ELECTRICITY_DATE]
    ).reset_index(drop=True)

    # Save intermediate dataframe to CSV
    output_path: Path = (
        pth.INTERMEDIATE_ELECTRICITY_PRICE_DIR
        / "intermediate_european_wholesale_electricity_price.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving intermediate electricity dataset at {output_path}")
    intermediate_df.to_csv(output_path, index=False)


def raw_to_intermediate_automobile_industry(auto_df: pd.DataFrame) -> None:
    """Transforms ECB automobile industry data to intermediate format.

    Keeps date (normalized to first day of month) and new passenger car registration,
    drops the time period column, and writes the cleaned dataset.

    Args:
        auto_df (pd.DataFrame): Raw automobile industry dataframe.
    """
    logger.info("Cleaning column names")
    intermediate_df = clean_column_names(auto_df).copy()

    rename_map = {
        raw_names.AI_DATE: intermediate_names.AI_DATE,
        raw_names.AI_NEW_PASSENGER_REG: intermediate_names.AI_NEW_PASSENGER_REG,
    }
    intermediate_df = intermediate_df.rename(columns=rename_map)

    # Parse date and set to first day of month
    intermediate_df[intermediate_names.AI_DATE] = (
        pd.to_datetime(
            intermediate_df[intermediate_names.AI_DATE],
            format=cst.DATE_FORMAT,
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Ensure registrations are numeric
    intermediate_df[intermediate_names.AI_NEW_PASSENGER_REG] = pd.to_numeric(
        intermediate_df[intermediate_names.AI_NEW_PASSENGER_REG],
        errors="coerce",
    )

    # Drop unneeded column if present
    intermediate_df = intermediate_df.drop(
        columns=[raw_names.AI_TIME_PERIOD], errors="ignore"
    )

    # Drop invalid rows
    intermediate_df = intermediate_df.dropna(
        subset=[intermediate_names.AI_DATE, intermediate_names.AI_NEW_PASSENGER_REG]
    )

    # Sort for deterministic output
    intermediate_df = intermediate_df.sort_values(
        by=[intermediate_names.AI_DATE]
    ).reset_index(drop=True)

    output_path: Path = (
        pth.INTERMEDIATE_AUTOMOBILE_INDUSTRY_DIR
        / "intermediate_automobile_industry.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving intermediate automobile dataset at %s", output_path)
    intermediate_df.to_csv(output_path, index=False)


def raw_to_intermediate() -> None:
    """Runs all raw to intermediate data pipelines."""
    # phenol_acetone_capacity_loss datasets
    logger.info(
        "Reading raw phenol capacity loss dataset at "
        f"{pth.RAW_PHENOL_ACETONE_DIR / 'phenol_consumption_capacity_loss_kt.pq'}"
    )
    raw_phenol = pd.read_parquet(
        pth.RAW_PHENOL_ACETONE_DIR / "phenol_consumption_capacity_loss_kt.pq"
    )

    logger.info(
        "Reading raw acetone capacity loss dataset at "
        f"{pth.RAW_PHENOL_ACETONE_DIR / 'acetone_consumption_capacity_loss_kt.pq'}"
    )
    raw_acetone = pd.read_parquet(
        pth.RAW_PHENOL_ACETONE_DIR / "acetone_consumption_capacity_loss_kt.pq"
    )

    raw_to_intermediate_phenol_acetone_bpa(raw_phenol, raw_acetone)
    logger.info("Completed processing raw to intermediate phenol/acetone BPA dataset")

    # pc_price datasets
    logger.info(
        "Reading raw pc_price Europe dataset at "
        f"{pth.RAW_PC_PRICE_DIR / 'pc_price_eu.csv'}"
    )
    raw_pc_price_eu = pd.read_csv(pth.RAW_PC_PRICE_DIR / "pc_price_eu.csv")
    raw_to_intermediate_pc_price_eu(raw_pc_price_eu)
    logger.info("Completed processing raw to intermediate pc_price Europe dataset")

    # Create conversion rates dataframe for Asia dataset
    logger.info(
        "Reading raw currency conversion rates datasets at "
        f"{pth.RAW_DATA_DIR / 'DEXCHUS.csv'} and "
        f"{pth.RAW_DATA_DIR / 'DEXINUS.csv'}"
    )
    df_conversion_rates_usd_rmb = pd.read_csv(pth.RAW_DATA_DIR / "DEXCHUS.csv")
    df_conversion_rates_usd_inr = pd.read_csv(pth.RAW_DATA_DIR / "DEXINUS.csv")

    df_conversion_rates_usd_rmb[raw_names.DEXCHUS_OBSERVATION_DATE] = pd.to_datetime(
        df_conversion_rates_usd_rmb[raw_names.DEXCHUS_OBSERVATION_DATE],
        format=cst.DATE_FORMAT,
    )
    df_conversion_rates_usd_inr[raw_names.DEXINUS_OBSERVATION_DATE] = pd.to_datetime(
        df_conversion_rates_usd_inr[raw_names.DEXINUS_OBSERVATION_DATE],
        format=cst.DATE_FORMAT,
    )

    df_conversion_rates = pd.merge(
        df_conversion_rates_usd_rmb,
        df_conversion_rates_usd_inr,
        left_on=raw_names.DEXCHUS_OBSERVATION_DATE,
        right_on=raw_names.DEXINUS_OBSERVATION_DATE,
        how="outer",
        validate="1:1",
    )

    logger.info(
        "Reading raw pc_price Asia dataset at "
        f"{pth.RAW_PC_PRICE_DIR / 'pc_price_asia.csv'}"
    )
    raw_pc_price_asia = pd.read_csv(pth.RAW_PC_PRICE_DIR / "pc_price_asia.csv", sep=";")
    raw_to_intermediate_pc_price_asia(raw_pc_price_asia, df_conversion_rates)
    logger.info("Completed processing raw to intermediate pc_price Asia dataset")

    # electricity_price dataset
    electricity_path = (
        pth.RAW_ELECTRICITY_PRICE_DIR
        / "european_wholesale_electricity_price_data_monthly.csv"
    )
    logger.info("Reading raw electricity dataset at %s", electricity_path)
    raw_electricity_prices = pd.read_csv(electricity_path)
    raw_to_intermediate_electricity_price(raw_electricity_prices)
    logger.info("Completed processing raw to intermediate electricity dataset")

    # automobile_industry dataset
    auto_files = sorted(pth.RAW_AUTOMOBILE_INDUSTRY_DIR.glob("ECB Data Portal_*.csv"))
    if not auto_files:
        raise FileNotFoundError(
            f"No ECB automobile industry CSV found in {pth.RAW_AUTOMOBILE_INDUSTRY_DIR}"
        )
    auto_path = auto_files[0]
    logger.info("Reading raw automobile industry dataset at %s", auto_path)
    raw_auto = pd.read_csv(auto_path)
    raw_to_intermediate_automobile_industry(raw_auto)
    logger.info("Completed processing raw to intermediate automobile dataset")


if __name__ == "__main__":
    logger.info("Starting raw to intermediate data pipelines")
    raw_to_intermediate()
    logger.info("Completed all raw to intermediate data pipelines")
