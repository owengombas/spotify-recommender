import numpy as np
import pandas as pd
from typing import List


class MatrixDataset:
    _R: np.ndarray

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def sorted_R(self) -> np.ndarray:
        I = np.argsort(self._R, axis=1)
        return I

    def __init__(self, df: pd.DataFrame, user_col: str, item_col: str, value_col: str):
        # A matrix of shape (num_users, num_items)
        self._user_col = user_col
        self._item_col = item_col
        self._value_col = value_col

        self._df = df.copy()
        self._df[user_col] = self._df[user_col].astype("category")
        self._df[item_col] = self._df[item_col].astype("category")

        num_users = len(self._df[user_col].cat.categories)
        num_items = len(self._df[item_col].cat.categories)

        self._R = np.zeros((num_users, num_items))

        for _, row in self._df.iterrows():
            # We use the codes to get the index of the category
            user_id = self.usernames_to_ids([row[user_col]])[0]
            item_id = self.items_to_ids([row[item_col]])[0]
            self._R[user_id, item_id] = row[value_col]

    def compute_alpha(self):
        alpha = len(np.where(self.R == 0)[0]) / self.R.sum()
        return alpha

    def get_user(self, user_id: int) -> np.ndarray:
        """
        Returns the user vector.
        :param user_id: The user id.
        :return: The user vector.
        """
        return self._R[user_id]

    def get_item(self, item_id: int) -> np.ndarray:
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
    
    def items_ids_to_df(self, ids: List[int], score: List[float] = None) -> pd.DataFrame:
        """
        Retrieve the items of the given ids and score from the dataframe.
        :param ids: The ids.
        :param score: The score.
        :return: The dataframe.
        """
        df = self._df[self._df[self._item_col].cat.codes.isin(ids)]
        if score is not None:
            df['score'] = score
        return df

    def __str__(self):
        return str(self._R)
