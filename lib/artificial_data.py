from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict
from IPython.display import display
from faker import Faker

fake = Faker()


def generate_from_normal_distribution(mean: float, std: float) -> np.ndarray:
    """
    Generate a function that returns samples from a normal distribution.

    Args:
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.

    Returns:
        Callable[[int], np.ndarray]: A function that takes an integer n_samples and returns an array of n_samples drawn from the normal distribution.
    """

    def inner(n_samples: int) -> np.ndarray:
        v = np.random.normal(mean, std, n_samples)
        return v

    return inner


class ArtificialTrackDataset:
    """
    A class representing an artificial dataset of tracks with various features.

    Attributes:
        data (pd.DataFrame): The underlying pandas DataFrame holding the dataset.
        numerical_features (List[str]): List of names of numerical features.
        n_samples (int): Number of samples in the dataset.
        default_value (float): Default value for numerical features.
        constant_features (Dict[str, str]): Dictionary of constant features and their values.
        ids (List[str]): List of unique identifiers for each sample in the dataset.
    """

    data: pd.DataFrame

    def __init__(
        self,
        numerical_features: List[str],
        n_samples: int,
        default_value: float = 0,
        constant_features: Dict[str, str] = None,
    ):
        self.numerical_features = numerical_features
        self.n_samples = n_samples
        self.default_value = default_value
        self.constant_features = constant_features
        self.ids = [fake.text(max_nb_chars=64)[:-1] for _ in range(n_samples)]
        self.data = pd.DataFrame()

        self.set_ids(self.ids)

        self._set_constant_features(constant_features)

        for feature in numerical_features:
            self.data[feature] = np.ones(n_samples) * default_value

    def merge(self, other: ArtificialTrackDataset) -> ArtificialTrackDataset:
        """
        Merge two datasets

        Args:
            other (ArtificialTrackDataset): The dataset to merge with.

        Returns:
            ArtificialTrackDataset: The merged dataset.
        """
        new_dataset = self.copy()
        new_dataset.data = pd.concat([self.data, other.data], ignore_index=True)
        return new_dataset

    def _set_constant_features(self, constant_features: Dict[str, str]):
        """
        Set the constant features of the dataset with the given values.

        Args:
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            None
        """
        for key, value in constant_features.items():
            self.data[key] = value

    def set_ids(self, ids: List[str]):
        """
        Set the ids of the dataset with the given values.

        Args:
            ids (List[str]): List of unique identifiers for each sample in the dataset.

        Returns:
            None
        """
        self.ids = ids
        self.data["id"] = self.ids

    def shuffle(self) -> "ArtificialTrackDataset":
        self.data = self.data.sample(frac=1)
        return self

    def insert_some_tracks(
        self,
        other: "ArtificialTrackDataset",
        min_percentage: float = 0.1,
        max_percentage: float = 0.5,
    ) -> Tuple["ArtificialTrackDataset", float]:
        """
        Replace a percentage of the ids with the given IDs.
        It uses a beta distribution to generate the percentage that has more chance to be low.

        Args:
            other (ArtificialTrackDataset): The dataset to replace with.
            min_percentage (float): Minimum percentage of tracks to replace.
            max_percentage (float): Maximum percentage of tracks to replace.

        Returns:
            Tuple[ArtificialTrackDataset, float]: The new dataset and the percentage of tracks replaced.
        """
        new_dataset = self.copy()
        other = other.copy()
        # Put more chance to have a low percentage, not using a uniform distribution
        percentage = np.random.beta(1, 4)
        percentage = min_percentage + (max_percentage - min_percentage) * percentage
        n_replace = int(new_dataset.n_samples * percentage)
        n_replace = min(n_replace, new_dataset.n_samples, other.n_samples)
        other.data = other.data.sample(frac=1)

        # replace the values of the first n_replace of the id column
        new_dataset.data["id"][:n_replace] = other.data["id"][:n_replace]
        new_dataset.data[self.numerical_features][:n_replace] = other.data[
            self.numerical_features
        ][:n_replace]

        new_dataset.data["username"] = self.data["username"].iloc[0]
        new_dataset.data = new_dataset.data.reset_index(drop=True)
        new_dataset.n_samples = new_dataset.data.shape[0]

        new_dataset._set_constant_features(self.constant_features)

        return new_dataset, n_replace / new_dataset.n_samples

    def randomize(
        self,
        numerical_features: List[str] = None,
        generator: Callable[[int], np.ndarray] = generate_from_normal_distribution(
            1, 0.1
        ),
    ) -> "ArtificialTrackDataset":
        """
        Randomize the numerical features of the dataset.

        Args:
            numerical_features (List[str]): List of names of numerical features.
            generator (Callable[[int], np.ndarray]): A function that takes an integer n_samples and returns an array of n_samples drawn from a distribution.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        if numerical_features is None:
            numerical_features = self.numerical_features
        for feature in numerical_features:
            self.data[feature] = generator(self.data.shape[0])
        return self

    def copy(
        self, ids: List[str] = None, constant_features: Dict[str, str] = None
    ) -> ArtificialTrackDataset:
        """
        Copy the dataset.

        Args:
            ids (List[str]): List of unique identifiers for each sample in the dataset.
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        if constant_features is None:
            constant_features = self.constant_features

        if ids is None:
            ids = self.ids

        new_dataset = ArtificialTrackDataset(
            numerical_features=self.numerical_features,
            n_samples=self.n_samples,
            default_value=self.default_value,
            constant_features=constant_features,
        )
        new_dataset.set_ids(ids)
        new_dataset.data[self.numerical_features] = self.data[self.numerical_features]
        return new_dataset

    def times(
        self,
        other: ArtificialTrackDataset,
        ids: List[str] = None,
        constant_features: List[str] = None,
    ) -> ArtificialTrackDataset:
        """
        Multiply component wise the numerical features of two datasets

        Args:
            other (ArtificialTrackDataset): The dataset to multiply with.
            ids (List[str]): List of unique identifiers for each sample in the dataset.
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        assert self.n_samples == other.n_samples
        assert self.numerical_features == other.numerical_features

        new_dataset = self.copy(ids, constant_features)
        new_dataset.data[self.numerical_features] = (
            self.data[self.numerical_features] * other.data[self.numerical_features]
        )
        return new_dataset

    def add(
        self,
        other: ArtificialTrackDataset,
        ids: List[str] = None,
        constant_features: List[str] = None,
    ) -> ArtificialTrackDataset:
        """
        Add component wise the numerical features of two datasets

        Args:
            other (ArtificialTrackDataset): The dataset to add with.
            ids (List[str]): List of unique identifiers for each sample in the dataset.
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        assert self.n_samples == other.n_samples
        assert self.numerical_features == other.numerical_features

        new_dataset = self.copy(ids, constant_features)
        new_dataset.data[self.numerical_features] = (
            self.data[self.numerical_features] + other.data[self.numerical_features]
        )
        return new_dataset

    def divide(
        self,
        other: ArtificialTrackDataset,
        ids: List[str] = None,
        constant_features: List[str] = None,
    ) -> ArtificialTrackDataset:
        """
        Divide component wise the numerical features of two datasets

        Args:
            other (ArtificialTrackDataset): The dataset to divide with.
            ids (List[str]): List of unique identifiers for each sample in the dataset.
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        assert self.n_samples == other.n_samples
        assert self.numerical_features == other.numerical_features

        new_dataset = self.copy(ids, constant_features)
        new_dataset.data[self.numerical_features] = (
            self.data[self.numerical_features] / other.data[self.numerical_features]
        )
        return new_dataset

    def subtract(
        self,
        other: ArtificialTrackDataset,
        ids: List[str] = None,
        constant_features: List[str] = None,
    ) -> ArtificialTrackDataset:
        """
        Subtract component wise the numerical features of two datasets

        Args:
            other (ArtificialTrackDataset): The dataset to substract with.
            ids (List[str]): List of unique identifiers for each sample in the dataset.
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        assert self.n_samples == other.n_samples
        assert self.numerical_features == other.numerical_features

        new_dataset = self.copy(ids, constant_features)
        new_dataset.data[self.numerical_features] = (
            self.data[self.numerical_features] - other.data[self.numerical_features]
        )
        return new_dataset

    def opposite(self, constant_features: List[str] = None) -> ArtificialTrackDataset:
        """
        Opposite component wise the numerical features of a dataset

        Args:
            constant_features (Dict[str, str]): Dictionary of constant features and their values.

        Returns:
            ArtificialTrackDataset: The new dataset.
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = -self.data[self.numerical_features]
        return new_dataset

    def to_tensor(
        self, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Convert the dataset to a torch tensor.

        Args:
            device (str): The device to use.
            dtype (torch.dtype): The dtype to use.

        Returns:
            torch.Tensor: The tensor representation of the dataset.
        """
        return torch.tensor(
            self.data[self.numerical_features].values, dtype=dtype, device=device
        )

    def to_numpy(self) -> np.ndarray:
        """
        Convert the dataset to a numpy array.

        Returns:
            np.ndarray: The numpy array representation of the dataset.
        """
        return self.data[self.numerical_features].values

    def __getitem__(self, index: int) -> ArtificialTrackDataset:
        new_dataset = self.copy()
        new_dataset.data = new_dataset.data.iloc[index]
        return new_dataset

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return display(self.data)

    def __str__(self) -> str:
        return self.data.__str__()


def generate_dataset_with_similarities(
    num_users: int,
    n_samples: List[int],
    numerical_features: List[str],
    affinity_feature: str,
    min_track_mixing_percentage: float = 0.1,
    max_track_mixing_percentage: float = 0.5,
    mix_with_someone_else_chance: float = 0.5,
) -> ArtificialTrackDataset:
    """
    Generate an artificial dataset with similarities between different users.

    Args:
        num_users (int): The number of users to simulate.
        n_samples (List[int]): A list of integers specifying the number of samples for each user.
        numerical_features (List[str]): List of names of numerical features.
        affinity_feature (str): The feature used to mix tracks between users.
        min_track_mixing_percentage (float): Minimum percentage of tracks to mix between users.
        max_track_mixing_percentage (float): Maximum percentage of tracks to mix between users.
        mix_with_someone_else_chance (float): Probability of a user mixing tracks with another user.

    Returns:
        ArtificialTrackDataset: The combined dataset after applying the mixing logic.
    """

    user_datasets: List[ArtificialTrackDataset] = []

    for user in range(num_users):
        user_dataset = ArtificialTrackDataset(
            numerical_features=numerical_features,
            n_samples=n_samples[user],
            default_value=0.0,
            constant_features={"username": fake.name()},
        ).randomize()
        user_datasets.append(user_dataset)

    for i in range(num_users):
        # pick a random chance to mix with another user
        if np.random.uniform() < mix_with_someone_else_chance:
            next_user: int = np.random.randint(0, num_users)
            # make sure we don't mix with ourselves
            while next_user == i:
                next_user: int = np.random.randint(0, num_users)

            user_datasets[i], replacement_percentage = user_datasets[
                i
            ].insert_some_tracks(
                user_datasets[next_user],
                min_percentage=min_track_mixing_percentage,
                max_percentage=max_track_mixing_percentage,
            )
            user_datasets[i].randomize([affinity_feature])

            print(
                f"Mixing user {user_datasets[i].constant_features['username']} with {user_datasets[next_user].constant_features['username']} with a replacement percentage of {replacement_percentage * 100:.2f}%"
            )

    df = user_datasets[0]
    for user_dataset in range(1, num_users):
        df = df.merge(user_datasets[user_dataset])

    df.n_samples = df.data.shape[0]
    df.randomize([affinity_feature])

    return df
