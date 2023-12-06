import torch
import pandas as pd
from typing import List
import numpy as np


class MatrixDataset:
    _R: torch.Tensor  # Changed to a PyTorch Tensor

    @property
    def R(self) -> torch.Tensor:
        return self._R

    @property
    def sorted_R(self) -> torch.Tensor:
        I = torch.argsort(self.R, axis=1)  # Convert to dense for sorting
        return I

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
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
        alpha = (torch.numel(self.R) - torch.count_nonzero(self.R)) / self.R.sum()
        return alpha

    def get_user(self, user_id: int) -> torch.Tensor:
        """
        Returns the user vector.
        :param user_id: The user id.
        :return: The user vector.
        """
        return self._R[user_id]

    def get_item(self, item_id: int) -> torch.Tensor:
        """
        Returns the item vector.
        :param item_id: The item id.
        :return: The item vector.
        """
        return self._R[:, item_id]

    def get_interaction(self, user_id: int, item_id: int) -> float:
        """
        Returns the interaction between a user and an item.
        :param user_id: The user id.
        :param item_id: The item id.
        :return: The interaction.
        """
        return self._R[user_id, item_id]

    def usernames_to_ids(self, usernames: List[str]) -> List[int]:
        """
        Returns the ids of the given usernames.
        :param usernames: The usernames.
        :return: The ids.
        """
        return self._df[self._df[self._user_col].isin(usernames)][
            self._user_col
        ].cat.codes.tolist()

    def ids_to_usernames(self, ids: List[int]) -> List[str]:
        """
        Returns the usernames of the given ids.
        :param ids: The ids.
        :return: The usernames.
        """
        return self._df[self._df[self._user_col].cat.codes.isin(ids)][
            self._user_col
        ].tolist()

    def items_to_ids(self, items: List[str]) -> List[int]:
        """
        Returns the ids of the given items.
        :param items: The items.
        :return: The ids.
        """
        return self._df[self._df[self._item_col].isin(items)][
            self._item_col
        ].cat.codes.tolist()

    def ids_to_items(self, ids: List[int]) -> List[str]:
        """
        Returns the items of the given ids.
        :param ids: The ids.
        :return: The items.
        """
        return self._df[self._df[self._item_col].cat.codes.isin(ids)][
            self._item_col
        ].tolist()

    def items_ids_to_df(
        self, ids: List[int], score: List[float] = None
    ) -> pd.DataFrame:
        """
        Retrieve the items of the given ids and score from the dataframe.
        :param ids: The ids.
        :param score: The score.
        :return: The dataframe.
        """
        df = self._df[self._df[self._item_col].cat.codes.isin(ids)]
        if score is not None:
            df["score"] = score
        return df

    def __str__(self):
        return str(self._R)
