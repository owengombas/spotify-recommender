import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from sklearn.model_selection import train_test_split
import torch


def remove_outliers_iqr(
    df: pd.DataFrame,
    numeric_features: List[str],
    quantile: float = 0.25,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from a dataframe using the interquartile range.

    Args:
        df (pd.DataFrame): The dataframe.
        numeric_features (List[str]): The numeric features to remove outliers from.
        quantile (float): The quantile to use for the interquartile range.
        threshold (float): The threshold to use for the interquartile range.

    Returns:
        pd.DataFrame: The dataframe with outliers removed.
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

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column to count.
        normalize (bool): Whether to normalize the counts.
        username_column (str): The username column.
        destination_column (str): The destination column.
        id_column (str): The id column (unique identifier).

    Returns:
        pd.DataFrame: The dataframe with the counts.
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

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column to group by.
    
    Returns:
        pd.DataFrame: The smallest sub dataframe for each value in a column.
    """
    return min(df.value_counts(column).to_dict().items(), key=lambda x: x[1])[0]


def count_number_of_occurence(df: pd.DataFrame, column: str):
    """
    Count the number of occurence of each value in a column.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column to count.
    
    Returns:
        pd.DataFrame: The dataframe with the counts.
    """
    return df.value_counts(column)


def filter_rows_by_min_number_of_occurence(
    df: pd.DataFrame, column: str, at_least: int
):
    """
    Filter the rows of a dataframe by the minimum number of occurence of a value in a column.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column to count.
        at_least (int): The minimum number of occurence.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    df_count = count_number_of_occurence(df, column)
    df = df[df[column].isin(df_count.index[df_count >= at_least])]
    return df


def compute_sparsity(
    df: pd.DataFrame, username_column: str, item_column: str
) -> Tuple[float, int, int, int]:
    """
    Compute the sparsity of a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        username_column (str): The username column.
        item_column (str): The item column.

    Returns:
        Tuple[float, int, int, int]: The sparsity, number of users, number of items and number of interactions.
    """
    users = count_number_of_occurence(df, username_column)
    items = count_number_of_occurence(df, item_column)
    sparsity = 1.0 * df.shape[0] / (users.shape[0] * items.shape[0])
    return sparsity, users, items, df


def permute_users(
    users_count: pd.Series, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permute the users of a dataframe.
    :param df: The dataframe.
    :param username_column: The username column.
    :param item_column: The item column.
    :return: The permuted dataframe.
    """
    df_shuffled = users_count.to_frame()
    df_shuffled = df_shuffled.sample(frac=1, random_state=seed)
    df_shuffled["old_index"] = df_shuffled.index
    return df_shuffled


def split_users_train_validation_test(
    df_users_shuffled: pd.DataFrame,
    validation_percentage: float,
    test_percentage: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train, validation and test sets.

    Args:
        df_users_shuffled (pd.DataFrame): The dataframe.
        validation_percentage (float): The percentage of users to use for validation set.
        test_percentage (float): The percentage of users to use for test set.
        seed (int): The random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation and test sets.
    """
    dev_df, test_df = train_test_split(
        df_users_shuffled,
        test_size=test_percentage,
        random_state=seed,
        shuffle=False,
    )
    if validation_percentage > 0:
        train_df, validation_df = train_test_split(
            dev_df,
            test_size=validation_percentage,
            random_state=seed,
            shuffle=False,
        )
    else:
        train_df = dev_df
        validation_df = pd.DataFrame()

    return train_df, validation_df, test_df


def populate_train_test_dataframes(
    df: pd.DataFrame,
    selected_users: pd.DataFrame,
    test_percentage: 0.2,
    min_items_threshold: int = 5,
    verbose: bool = False,
    log_interval: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train and test sets.

    Args:
        df (pd.DataFrame): The dataframe.
        selected_users (pd.DataFrame): The selected users.
        test_percentage (float): The percentage of users to use for the test set.
        min_items_threshold (int): The minimum number of items per user.
        verbose (bool): Whether to print progress.
        log_interval (int): The interval to print progress.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test sets.
    """
    username_column = selected_users.index.name
    df = df.copy()
    df = df[df[username_column].isin(selected_users.index)]

    list_df_train: List[pd.DataFrame] = []
    list_df_test: List[pd.DataFrame] = []

    for data, user in df.groupby(username_column, observed=True):
        n_items = user.shape[0]

        if n_items >= min_items_threshold:
            idx = np.zeros(n_items, dtype=bool)
            idx[
                np.random.choice(
                    n_items, int(n_items * test_percentage), replace=False
                ).astype(int)
            ] = True
            list_df_train.append(user[np.logical_not(idx)])
            list_df_test.append(user[idx])
        else:
            list_df_train.append(user)

        if verbose and data % log_interval == 0:
            print(f"Processed {data} users out of {df[username_column].nunique()}")

    df_train = pd.concat(list_df_train)
    df_test = pd.concat(list_df_test)

    return df_train, df_test


def min_max_scale(
    tensor: torch.Tensor,
    bounds: Tuple[float, float],
    optimums: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Min-max scale a tensor.

    Args:
        tensor (torch.Tensor): The tensor.
        bounds (Tuple[float, float]): The bounds.
        optimums (Optional[Tuple[float, float]]): The optimums.

    Returns:
        torch.Tensor: The min-max scaled tensor.
    """
    if optimums is None:
        minimum, maximum = tensor.min().item(), tensor.max().item()
    else:
        minimum, maximum = optimums

    bound_minimum, bound_maximum = bounds

    return (
        (tensor - minimum) / (maximum - minimum) * (bound_maximum - bound_minimum)
        + bound_minimum,
        minimum,
        maximum,
    )
