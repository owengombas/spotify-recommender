import pandas as pd
import numpy as np
from typing import List


def remove_outliers_iqr(
    df: pd.DataFrame,
    numeric_features: List[str],
    quantile: float = 0.25,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from a dataframe using the interquartile range.
    :param df: The dataframe.
    :param numeric_features: The numeric features to remove outliers from.
    :param quantile: The quantile to use for the interquartile range.
    :param threshold: The threshold to use for the interquartile range.
    :return: The dataframe with outliers removed.
    """
    df = df.copy()
    for feature in numeric_features:
        q1 = df[feature].quantile(quantile)
        q3 = df[feature].quantile(1 - quantile)
        iqr = q3 - q1
        df = df[
            (df[feature] >= q1 - threshold * iqr)
            & (df[feature] <= q3 + threshold * iqr)
        ]
    return df


def count_list_column_per_user(
    df: pd.DataFrame,
    column: str,
    normalize: bool = True,
    username_column: str = "username",
    destination_column: str = "count",
    id_column: str = "id",
) -> pd.DataFrame:
    """
    Count the number of times each value in a list column appears per user.
    :param df: The dataframe.
    :param username_column: The username column.
    :param column: The column to count.
    :id_column: The id column (unique identifier).
    :param destination_column: The destination column.
    :return: The dataframe with the counts.
    """

    df_count = df.explode(column)
    df_count = df_count.groupby([column, username_column]).count().reset_index()
    df_count = df_count.rename(columns={id_column: destination_column})
    df_count = df_count[[column, username_column, destination_column]]
    if normalize:
        df_count[destination_column] = df_count[destination_column] / df_count.groupby(
            username_column
        )[destination_column].transform(destination_column)
    df_count = df_count.sort_values(destination_column, ascending=False)

    return df_count


def get_smallest_sub_dataframe(df: pd.DataFrame, column: str):
    """
    Get the smallest sub dataframe for each value in a column.
    :param df: The dataframe.
    :param column: The column to group by.
    :return: The smallest sub dataframe for each value in a column.
    """
    return min(df.value_counts(column).to_dict().items(), key=lambda x: x[1])[0]
