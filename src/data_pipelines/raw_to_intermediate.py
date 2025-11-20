# Contains the functions defining the raw to intermediate data pipelines


import pandas as pd

import constants.constants as cst
import constants.intermediate_names as intermediate_names
import constants.paths as pth
import constants.raw_names as raw_names
from src.utils.clean_column_names import clean_column_names
from src.utils.logger import logger


def raw_to_intermediate_pc_price_eu(pc_price_eu: pd.DataFrame) -> None:
    """Transforms the raw pc_price Europe dataset to intermediate format.

    This function cleans the column names of the raw dataframe and saves the
    resulting intermediate dataframe to a CSV file.

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
        [
            raw_names.PC_EU_SUPPLIER_1_CRYSTAL,
            raw_names.PC_EU_SUPPLIER_2_CRYSTAL,
            raw_names.PC_EU_SUPPLIER_3_CRYSTAL,
            raw_names.PC_EU_SUPPLIER_4_CRYSTAL,
            raw_names.PC_EU_SUPPLIER_5_CRYSTAL,
        ]
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_WHITE_BEST_PRICE] = intermediate_df[
        [
            raw_names.PC_EU_SUPPLIER_1_WHITE,
            raw_names.PC_EU_SUPPLIER_1_WHITE_ALT,
            raw_names.PC_EU_SUPPLIER_2_WHITE,
            raw_names.PC_EU_SUPPLIER_3_WHITE,
            raw_names.PC_EU_SUPPLIER_5_WHITE,
            raw_names.PC_EU_SUPPLIER_6_WHITE,
        ]
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_GF10_BEST_PRICE] = intermediate_df[
        [
            raw_names.PC_EU_SUPPLIER_1_GF10_FR,
            raw_names.PC_EU_SUPPLIER_1_GF10_FR_ALT,
            raw_names.PC_EU_SUPPLIER_2_GF10FR,
            raw_names.PC_EU_SUPPLIER_3_GF10_FR,
            raw_names.PC_EU_SUPPLIER_4_GF10_FR,
            raw_names.PC_EU_SUPPLIER_7_GF10FR,
            raw_names.PC_EU_SUPPLIER_9_GF10_FR,
        ]
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_GF20_BEST_PRICE] = intermediate_df[
        [
            raw_names.PC_EU_SUPPLIER_2_GF20,
            raw_names.PC_EU_SUPPLIER_7_GF20,
        ]
    ].min(axis=1)

    intermediate_df[intermediate_names.PC_EU_PC_RECYCLED_GF10_WHITE_BEST_PRICE] = (
        intermediate_df[
            [
                raw_names.PC_EU_SUPPLIER_5_RECYCLED_GF10_WHITE,
                raw_names.PC_EU_SUPPLIER_8_RECYCLED_GF10_WHITE,
            ]
        ].min(axis=1)
    )

    intermediate_df[intermediate_names.PC_EU_PC_RECYCLED_GF10_GREY_BEST_PRICE] = (
        intermediate_df[
            [
                raw_names.PC_EU_SUPPLIER_5_RECYCLED_GF10_GREY,
                raw_names.PC_EU_SUPPLIER_8_RECYCLED_GF10_GREY,
            ]
        ].min(axis=1)
    )

    intermediate_df[intermediate_names.PC_EU_PC_SI_BEST_PRICE] = intermediate_df[
        [
            raw_names.PC_EU_SUPPLIER_2_SILOXANE,
            raw_names.PC_EU_SUPPLIER_2_SILOXANE_ALT,
            raw_names.PC_EU_SUPPLIER_4_SI,
            raw_names.PC_EU_SUPPLIER_4_SI_ALT,
            raw_names.PC_EU_SUPPLIER_7_SI,
        ]
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


if __name__ == "__main__":
    # pc_price datasets
    logger.info(
        "Reading raw pc_price Europe dataset at "
        f"{pth.RAW_PC_PRICE_DIR / 'pc_price_eu.csv'}"
    )
    raw_pc_price_eu = pd.read_csv(pth.RAW_PC_PRICE_DIR / "pc_price_eu.csv")
    raw_to_intermediate_pc_price_eu(raw_pc_price_eu)
    logger.info("Completed processing raw to intermediate pc_price Europe dataset")
