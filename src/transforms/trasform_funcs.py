from typing import Callable, Any, Optional
import pandas as pd
from pandas import DataFrame


def filter_by_predicate(
    df: DataFrame,
    column_name: str,
    predicate: Callable[[Any], bool]
) -> DataFrame:
    """
    Filter rows in a DataFrame based on a predicate function applied to a given column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    column_name : str
        The name of the column on which to apply the predicate.
    predicate : Callable[[Any], bool]
        A function that takes a single value from the specified column and returns a boolean.

    Returns
    -------
    DataFrame
        A new DataFrame filtered to only include rows where the predicate is True.
    """
    return df[df[column_name].apply(predicate)]


def select_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Select a subset of columns from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    columns : list[str]
        A list of column names to select.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the specified columns.
    """
    return df[columns]


def select_rows_by_index(df: DataFrame, indexes: list[int]) -> DataFrame:
    """
    Select a subset of rows from a DataFrame by their integer position (index).

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    indexes : list[int]
        A list of integer positions of the rows to select.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the selected rows.
    """
    return df.iloc[indexes]


def drop_missing_values(df: DataFrame, columns: Optional[list[str]] = None) -> DataFrame:
    """
    Drop rows with missing values from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    columns : list[str], optional
        A list of columns to check for missing values. If None, drop rows with missing
        values in any column.

    Returns
    -------
    DataFrame
        A new DataFrame with rows containing missing values removed.
    """
    return df.dropna(subset=columns) if columns else df.dropna()


def drop_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Drop specific columns from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    columns : list[str]
        A list of column names to drop.

    Returns
    -------
    DataFrame
        A new DataFrame without the specified columns.
    """
    return df.drop(columns=columns, axis=1)


def rename_columns(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
    """
    Rename columns in a DataFrame according to a given mapping.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    mapping : dict[str, str]
        A dictionary mapping old column names to new column names.

    Returns
    -------
    DataFrame
        A new DataFrame with renamed columns.
    """
    return df.rename(columns=mapping)


def fill_missing_values(
    df: DataFrame,
    value: Any,
    column: Optional[str] = None
) -> DataFrame:
    """
    Fill missing values in a DataFrame or in a specific column with a provided value.

    If `column` is None, missing values in all columns are filled. This operation
    modifies the DataFrame in place.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame (modified in place if `column` is None).
    value : Any
        The value to use for filling missing entries.
    column : str, optional
        The name of the column to fill. If None, all columns will be filled.

    Returns
    -------
    DataFrame
        The same DataFrame (modified in place if `column` is None) with missing
        values filled.
    """
    if column is None:
        df.fillna(value, inplace=True)
    else:
        df[column] = df[column].fillna(value)
    return df


def calculate_summary(df: DataFrame, column: str) -> dict[str, float]:
    """
    Calculate summary statistics for a specific column in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame.
    column : str
        The name of the column for which to calculate statistics.

    Returns
    -------
    dict[str, float]
        A dictionary containing the mean, median, standard deviation, minimum,
        and maximum of the specified column.
    """
    return {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "std": df[column].std(),
        "min": df[column].min(),
        "max": df[column].max(),
    }


def save_to_csv(df: DataFrame, path: str, index: bool = False) -> None:
    """
    Save a DataFrame to a CSV file.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be saved.
    path : str
        The file path or file name where the CSV should be saved.
    index : bool, optional
        Whether to write row names (index). Defaults to False.

    Returns
    -------
    None
        This function does not return anything. It writes the DataFrame to the specified file path.
    """
    df.to_csv(path, index=index)


def read_from_csv(path: str, **kwargs) -> DataFrame:
    """
    Read a CSV file into a DataFrame.

    Parameters
    ----------
    path : str
        The file path or file name of the CSV to read.
    **kwargs
        Additional keyword arguments to pass to pandas.read_csv (e.g., 'sep', 'header').

    Returns
    -------
    DataFrame
        A DataFrame created by reading in the CSV file.
    """
    return pd.read_csv(path, **kwargs)