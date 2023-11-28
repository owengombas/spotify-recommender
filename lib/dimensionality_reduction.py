import numpy as np
import pandas as pd

def add_latent_features(
    df: pd.DataFrame, latent_features: np.ndarray, prefix: str = ""
) -> pd.DataFrame:
    """
    Add latent features to a dataframe.
    :param df: The dataframe.
    :param latent_features: The latent features.
    :param prefix: The prefix to add to the column names.
    :return: The dataframe with latent features added.
    """
    df = df.copy()
    latent_size = latent_features.shape[1]
    latent_columns = []
    for i in range(latent_size):
        column_name = f"{prefix}latent_{i}"
        df[column_name] = latent_features[:, i]
        latent_columns.append(column_name)
    return df, latent_columns
