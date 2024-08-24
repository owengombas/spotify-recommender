import torch
import pandas as pd
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import lib.preprocessing as preprocessing
from typing import Tuple, Optional


class MatrixDataset:
    """
    A class to represent a dataset as a matrix for recommendation systems.

    Args:
        _R (torch.Tensor): Internal representation of the matrix.
        _df (pd.DataFrame): DataFrame containing the user-item interactions.
        _user_col (str): Column name for users.
        _item_col (str): Column name for items.
        _value_col (str): Column name for values (e.g., ratings).
    """

    _R: torch.Tensor  # Changed to a PyTorch Tensor

    @property
    def R(self) -> torch.Tensor:
        """
        Returns the internal matrix as a dense tensor.

        Returns:
            torch.Tensor: The internal matrix representation.
        """
        return self._R

    @property
    def sorted_R(self) -> torch.Tensor:
        I = torch.argsort(
            self.R, axis=1, descending=True
        )  # Convert to dense for sorting
        return I

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the MatrixDataset object.

        Args:
            df (pd.DataFrame): The DataFrame containing user-item interactions.
            user_col (str): The column name representing users.
            item_col (str): The column name representing items.
            value_col (str): The column name representing interaction values (e.g., ratings).
            device (str, optional): The device to store the tensor on. Defaults to "cpu".
            dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.
        """
        # A matrix of shape (num_users, num_items)
        self._user_col = user_col
        self._item_col = item_col
        self._value_col = value_col

        self._df = df.copy()
        self._df[user_col] = self._df[user_col].astype("category")
        self._df[item_col] = self._df[item_col].astype("category")

        num_users = len(self._df[user_col].cat.categories)
        num_items = len(self._df[item_col].cat.categories)

        # Vectorized creation of rows and cols
        user_ids = self._df[user_col].cat.codes
        item_ids = self._df[item_col].cat.codes
        values = self._df[value_col]

        # Create sparse tensor
        indices = torch.LongTensor(np.array([user_ids, item_ids]))
        values = torch.FloatTensor(values.values)
        self._R = torch.sparse_coo_tensor(indices, values, (num_users, num_items))
        self._R = self._R.to_dense().to(device=device, dtype=dtype)

    def compute_alpha(self):
        """
        Computes the alpha value based on the sparsity of the matrix.

        Returns:
            float: The computed alpha value.
        """
        alpha = (torch.numel(self.R) - torch.count_nonzero(self.R)) / self.R.nansum()
        return alpha

    def set_alpha(self, alpha: float):
        """
        Multiplies the internal matrix by the given alpha value.

        Args:
            alpha (float): The alpha value to multiply with.
        """
        self._R *= alpha

    def min_max_scale(
        self,
        bounds: Tuple[float, float],
        optimums: Optional[Tuple[float, float]] = None,
    ):
        """
        Applies min-max scaling to the internal matrix.

        Args:
            bounds (Tuple[float, float]): The lower and upper bounds for scaling.
            optimums (Optional[Tuple[float, float]], optional): The optimum values for scaling. Defaults to None.

        Returns:
            Tuple[float, float]: The minimum and maximum values after scaling.
        """
        self._R, minimum, maximum = preprocessing.min_max_scale(
            self._R, bounds, optimums
        )
        return minimum, maximum

    def get_user(self, user_id: int) -> torch.Tensor:
        """
        Returns the vector representation of a user.

        Args:
            user_id (int): The user's ID.

        Returns:
            torch.Tensor: The user vector.
        """
        return self._R[user_id]

    def get_item(self, item_id: int) -> torch.Tensor:
        """
        Retrieves the item vector for a given item ID from the matrix.

        Args:
            item_id (int): The ID of the item.

        Returns:
        . torch.Tensor: The vector representing the specified item.
        """
        return self._R[:, item_id]

    def get_interaction(self, user_id: int, item_id: int) -> float:
        """
        Retrieves the interaction value between a specific user and item.

        Args:
            user_id (int): The ID of the user.
            item_id (int): The ID of the item.

        Returns:
            float: The interaction value between the given user and item.
        """
        return self._R[user_id, item_id]

    def get_interactions_ids(self, user_id: int) -> torch.Tensor:
        """
        Retrieves a list of item IDs with which a specific user has interacted.

        Args:
            user_id (int): The ID of the user.

        Returns:
            torch.Tensor: A tensor of item IDs that the user has interacted with.
        """
        return self._R[user_id].nonzero().squeeze()

    def usernames_to_ids(self, usernames: List[str]) -> List[int]:
        """
        Converts a list of usernames to their corresponding user IDs.

        Args:
            usernames (List[str]): The list of usernames.

        Returns:
            List[int]: A list of user IDs corresponding to the given usernames.
        """
        df_users = self._df[self._user_col].cat.categories
        return df_users.get_indexer(usernames)

    def itemnames_to_ids(self, itemnames: List[str]) -> List[int]:
        """
        Converts a list of item names to their corresponding item IDs.

        Args:
            itemnames (List[str]): The list of item names.

        Returns:
            List[int]: A list of item IDs corresponding to the given item names.
        """
        df_items = self._df[self._item_col].cat.categories
        return df_items.get_indexer(itemnames)

    def ids_to_usernames(self, ids: List[int]) -> List[str]:
        """
        Converts a list of user IDs to their corresponding usernames.

        Args:
            ids (List[int]): The list of user IDs.

        Returns:
            List[str]: A list of usernames corresponding to the given user IDs.
        """
        df_users = self._df[self._user_col].cat.categories
        return df_users[ids].tolist()

    def ids_to_itemnames(self, ids: List[int]) -> List[str]:
        """
        Converts a list of item IDs to their corresponding item names.

        Args:
            ids (List[int]): The list of item IDs.

        Returns:
            List[str]: A list of item names corresponding to the given item IDs.
        """
        df_items = self._df[self._item_col].cat.categories
        return df_items[ids].tolist()

    def get_usernames(self) -> List[str]:
        """
        Returns the list of usernames.
        :return: The list of usernames.
        """
        return self._df[self._user_col].cat.categories.tolist()

    def get_itemnames(self) -> List[str]:
        """
        Returns the list of itemnames.
        :return: The list of itemnames.
        """
        return self._df[self._item_col].cat.categories.tolist()

    def item_ids_to_df(self, item_ids: List[int]) -> pd.DataFrame:
        """
        Returns the dataframe of the items.

        Args:
            item_ids (List[int]): The list of item ids.

        Returns:
            pd.DataFrame: The dataframe of the items.
        """
        # Retrieve the rows of the items from the dataframe
        df_items = self._df[self._df[self._item_col].cat.codes.isin(item_ids)]
        return df_items

    def user_interacted_with(self, user_id: int) -> List[int]:
        """
        Returns the list of item ids that the user interacted with.

        Args:
            user_id (int): The user's ID.

        Returns:
            List[int]: The list of item ids.
        """
        interacted = self._R[user_id].nonzero().squeeze()
        return interacted

    def __str__(self):
        return str(self._R)
