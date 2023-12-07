from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict
from IPython.display import display
from uuid import uuid4
from faker import Faker

fake = Faker()


def generate_from_normal_distribution(mean: float, std: float) -> np.ndarray:
    def inner(n_samples: int) -> np.ndarray:
        return np.random.normal(mean, std, n_samples)

    return inner


class ArtificialTrackDataset:
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
        self.ids = [fake.text(max_nb_chars=15)[:-1] for _ in range(n_samples)]
        self.data = pd.DataFrame()

        self.set_ids(self.ids)

        self._set_constant_features(constant_features)

        for feature in numerical_features:
            self.data[feature] = np.ones(n_samples) * default_value

    def merge(self, other: ArtificialTrackDataset) -> ArtificialTrackDataset:
        """
        Merge two datasets
        """
        new_dataset = self.copy()
        new_dataset.data = pd.concat([self.data, other.data], ignore_index=True)
        return new_dataset

    def _set_constant_features(self, constant_features: Dict[str, str]):
        for key, value in constant_features.items():
            self.data[key] = value

    def set_ids(self, ids: List[str]):
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
        Replace a percentage of the ids with the given ids
        """
        new_dataset = self.copy()
        n_replace = int(
            new_dataset.n_samples * np.random.uniform(min_percentage, max_percentage)
        )
        replace_ids_index = np.random.choice(
            new_dataset.n_samples, n_replace, replace=False
        )

        new_dataset.data.iloc[replace_ids_index] = other.data.iloc[replace_ids_index]
        new_dataset._set_constant_features(self.constant_features)

        return new_dataset, n_replace / new_dataset.n_samples

    def randomize(
        self,
        numerical_features: List[str] = None,
        generator: Callable[[int], np.ndarray] = generate_from_normal_distribution(
            1, 0.1
        ),
    ) -> "ArtificialTrackDataset":
        if numerical_features is None:
            numerical_features = self.numerical_features
        for feature in numerical_features:
            self.data[feature] = generator(self.n_samples)
        return self

    def copy(
        self, ids: List[str] = None, constant_features: Dict[str, str] = None
    ) -> ArtificialTrackDataset:
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
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = -self.data[self.numerical_features]
        return new_dataset

    def sqrt(self, constant_features: List[str] = None) -> ArtificialTrackDataset:
        """
        Square root component wise the numerical features of a dataset
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = np.sqrt(
            self.data[self.numerical_features]
        )
        return new_dataset

    def square(self, constant_features: List[str] = None) -> ArtificialTrackDataset:
        """
        Square component wise the numerical features of a dataset
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = np.square(
            self.data[self.numerical_features]
        )
        return new_dataset

    def pow(
        self, power: float, constant_features: List[str] = None
    ) -> ArtificialTrackDataset:
        """
        Raise to the power component wise the numerical features of a dataset
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = np.power(
            self.data[self.numerical_features], power
        )
        return new_dataset

    def exp(self, constant_features: List[str] = None) -> ArtificialTrackDataset:
        """
        Exponential component wise the numerical features of a dataset
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = np.exp(
            self.data[self.numerical_features]
        )
        return new_dataset

    def log(self, constant_features: List[str] = None) -> ArtificialTrackDataset:
        """
        Logarithm component wise the numerical features of a dataset
        """
        new_dataset = self.copy(constant_features)
        new_dataset.data[self.numerical_features] = np.log(
            self.data[self.numerical_features]
        )
        return new_dataset

    def to_tensor(
        self, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.tensor(
            self.data[self.numerical_features].values, dtype=dtype, device=device
        )

    def to_numpy(self) -> np.ndarray:
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
    n_samples: int,
    numerical_features: List[str],
    affinity_feature: str,
    min_track_mixing_percentage: float = 0.1,
    max_track_mixing_percentage: float = 0.5,
    mix_with_someone_else_chance: float = 0.5,
) -> ArtificialTrackDataset:
    user_datasets: List[ArtificialTrackDataset] = []

    for user in range(num_users):
        user_dataset = ArtificialTrackDataset(
            numerical_features=numerical_features,
            n_samples=n_samples,
            default_value=0.0,
            constant_features={"username": fake.name()},
        ).randomize()
        user_datasets.append(user_dataset)

    for i in range(num_users):
        # pick a random chance to mix with another user
        if np.random.uniform() < mix_with_someone_else_chance:
            index: int = np.random.randint(0, num_users)
            next_user: int = np.random.randint(0, num_users)
            # make sure we don't mix with ourselves
            while next_user == i:
                next_user = np.random.randint(0, num_users)

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

    return df
