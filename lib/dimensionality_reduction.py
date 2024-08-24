import numpy as np
import pandas as pd


def add_latent_features(
    df: pd.DataFrame, latent_features: np.ndarray, prefix: str = ""
) -> pd.DataFrame:
    """
    Add latent features to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        latent_features (np.ndarray): The latent features.
        prefix (str): The prefix to add to the column names.

    Returns:
        pd.DataFrame: The dataframe with latent features added.
    """
    df = df.copy()
    latent_size = latent_features.shape[1]
    latent_columns = []
    for i in range(latent_size):
        column_name = f"{prefix}latent_{i}"
        df[column_name] = latent_features[:, i]
        latent_columns.append(column_name)
    return df, latent_columns
