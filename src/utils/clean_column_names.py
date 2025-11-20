import pandas as pd


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
