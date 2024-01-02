from __future__ import annotations
from constants import users_list, data_path
from lib import (
    spoti,
    genre_normalizer,
    plotting,
    preprocessing,
    dimensionality_reduction,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors
from sklearn.metrics import pairwise_distances
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from IPython.display import display, HTML
from sklearn.base import BaseEstimator
from recommenders.metrics import similarities_score


class NearestNeighborsRecommender(SklearnNearestNeighbors, BaseEstimator):
    @property
    def n_neighbors(self) -> int:
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, n_neighbors: int):
        self._n_neighbors = n_neighbors

    @property
    def username_column(self) -> str:
        return self._username_column

    @username_column.setter
    def username_column(self, username_column: str):
        self._username_column = username_column

    @property
    def affinity_column(self) -> str:
        return self._affinity_column

    @affinity_column.setter
    def affinity_column(self, affinity_column: str):
        self._affinity_column = affinity_column

    @property
    def tracks_column(self) -> str:
        return self._tracks_column

    @tracks_column.setter
    def tracks_column(self, tracks_column: str):
        self._tracks_column = tracks_column

    @property
    def features_columns(self) -> List[str]:
        return self._features_columns

    @features_columns.setter
    def features_columns(self, features_columns: List[str]):
        self._features_columns = features_columns

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "cosine",
        p: int = 2,
        leaf_size: int = 30,
        radius: float = 1.0,
        n_jobs: int = -1,
        algorithm: str = "auto",
        metric_params: Optional[Dict[str, Any]] = None,
        features_columns: List[str] = None,
        username_column="username",
        affinity_column="affinity",
        tracks_column="id",
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            leaf_size=leaf_size,
            radius=radius,
            n_jobs=n_jobs,
            algorithm=algorithm,
            metric_params=metric_params,
        )
        self.n_neighbors: int = n_neighbors
        self.username_column: str = username_column
        self.affinity_column: str = affinity_column
        self.features_columns: List[str] = features_columns
        self.tracks_column: str = tracks_column

    def pick_user_tracks(
        self,
        df: pd.DataFrame,
        username: str,
        sort_by_affinity: bool = True,
        max_tracks: int = -1,
    ) -> pd.DataFrame:
        """
        Returns the tracks of the given user.
        """
        user_df = df[df[self._username_column] == username]

        if sort_by_affinity and self._affinity_column in user_df.columns:
            user_df = user_df.sort_values(by=self._affinity_column, ascending=False)

        if max_tracks > 0:
            user_df = user_df.head(max_tracks)

        return user_df

    def recommend_from(
        self,
        tracks: pd.DataFrame,
        based_on: pd.DataFrame = None,
        n_neighbors: int = None,
        sort_by_distance: bool = True,
        filter_out_tracks: bool = True,
        filter_out_if_seen: bool = True,
        max_tracks: int = -1,
    ) -> pd.DataFrame:
        """
        Returns the neighbours of the given tracks.
        """
        distances, indices = self.kneighbors(
            tracks[self._features_columns], n_neighbors
        )
        df_neighbours = based_on.iloc[indices.flatten()].copy()
        df_neighbours["distance"] = distances.flatten()

        username = tracks[self._username_column].iloc[0]

        # Add an attribute seen to df indicating if the track has been seen by the user
        # check if user <username> has seen the track
        df_neighbours["seen"] = df_neighbours[self._tracks_column].isin(
            based_on[based_on[self._username_column] == username][self._tracks_column]
        )

        if filter_out_if_seen:
            df_neighbours = df_neighbours[df_neighbours["seen"] == False]

        if filter_out_tracks:
            df_neighbours = df_neighbours[~df_neighbours.index.isin(tracks.index)]

        # Sort by distance
        if sort_by_distance:
            df_neighbours = df_neighbours.sort_values(by="distance", ascending=False)

        # Limit the number of tracks
        if max_tracks > 0:
            df_neighbours = df_neighbours.head(max_tracks)

        return df_neighbours

    def plot_recommendation(
        self,
        df_recommended_from: pd.DataFrame,
        df_recommended_tracks: pd.DataFrame,
        username: str = None,
    ) -> go.Figure:
        latent_columns = self._features_columns

        df_plot = df_recommended_from.copy()
        df_plot = pd.concat([df_plot, df_recommended_tracks])
        df_plot = df_plot.drop_duplicates(subset=self._tracks_column)

        if len(latent_columns) > 3:
            pca = PCA(n_components=3)
            latent_features = pca.fit_transform(df_plot[latent_columns])
            df_plot, latent_columns = dimensionality_reduction.add_latent_features(
                df_plot, latent_features
            )

        if username is not None:
            df_plot.loc[
                df_plot[self._username_column] != username, self._username_column
            ] = ""

        return plotting.plot_latent_space(
            df_plot,
            color=df_plot[self._username_column],
            text=df_plot[self._username_column],
            title="Recommendation space",
        )

    def retrievial_score_for_user(
        self,
        username: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_neighbours: int = None,
        verbose: bool = False,
    ):
        df_user = df_train[df_train["username"] == username]
        df_user = df_user.drop_duplicates(subset=["id"])
        df_user = df_user.sort_values(by="affinity", ascending=False)
        recommendations = self.recommend_from(
            df_user,
            based_on=df_train,
            n_neighbors=n_neighbours,
            filter_out_if_seen=False,
            filter_out_tracks=False,
        )
        recommendations = recommendations[recommendations["username"] == username]
        recommendations.drop_duplicates(subset=["id"], inplace=True)
        recommendations.sort_values(by="distance", ascending=False, inplace=True)
        heldout_sorted = df_test[df_test["username"] == username].sort_values(
            by="affinity", ascending=False
        )
        check_k = min(len(heldout_sorted), len(recommendations))
        if verbose:
            print(f"Check k: {check_k}")
            display(df_user[:check_k][spoti.PRETTY_PRINT_FEATURES])
            display(heldout_sorted[:check_k][spoti.PRETTY_PRINT_FEATURES])
            display(recommendations[:check_k][spoti.PRETTY_PRINT_FEATURES])
        number_of_items_appearing_in_both = len(
            set(heldout_sorted[:check_k]["id"]).intersection(
                set(recommendations[:check_k]["id"])
            )
        )
        value = 0
        if check_k > 0:
            value = number_of_items_appearing_in_both / check_k
        if verbose:
            print(
                f"Number of items appearing in both: {number_of_items_appearing_in_both}"
            )
            print(f"Precision@{check_k}: {value}")
        return value

    def retrievial_score(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_neighbors: int = None,
        verbose: bool = False,
    ):
        users = df_test["username"].unique()
        precision = 0
        for user in users:
            precision += self.retrievial_score_for_user(
                user, df_train, df_test, n_neighbours=n_neighbors, verbose=verbose
            )
        return precision / len(users)
    
    def similarities_score_for_user(
        self,
        username: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        n_neighbours: int = None,
        max_items: int = -1,
        verbose: bool = False,
    ):
        df_user = df_train[df_train["username"] == username]
        df_user = df_user.drop_duplicates(subset=["id"])
        df_user = df_user.sort_values(by="affinity", ascending=False)
        recommendations = self.recommend_from(
            df_user,
            based_on=df_train,
            n_neighbors=n_neighbours,
            filter_out_if_seen=False,
            filter_out_tracks=False,
        )
        recommendations.drop_duplicates(subset=["id"], inplace=True)
        recommendations.sort_values(by="distance", ascending=False, inplace=True)
        heldout_sorted = df_test[df_test["username"] == username].sort_values(
            by="affinity", ascending=False
        )
        check_k = min(len(heldout_sorted), len(recommendations), max_items)
        if verbose:
            print(f"Check k: {check_k}")
            display(df_user[:check_k][spoti.PRETTY_PRINT_FEATURES])
            display(heldout_sorted[:check_k][spoti.PRETTY_PRINT_FEATURES])
            display(recommendations[:check_k][spoti.PRETTY_PRINT_FEATURES])
        tensor_heldout = torch.tensor(heldout_sorted[:check_k][self._features_columns].values)
        tensor_recommendations = torch.tensor(recommendations[:check_k][self._features_columns].values)
        value = similarities_score(tensor_heldout, tensor_recommendations)
        if verbose:
            print(f"Similarities score: {value}")
        return value

    def similarities_score(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        max_items: int = -1,
        n_neighbors: int = None,
        verbose: bool = False,
    ):
        users = df_test["username"].unique()
        precision = 0
        for user in users:
            precision += self.similarities_score_for_user(
                user, df_train, df_test, n_neighbours=n_neighbors, verbose=verbose, max_items=max_items
            )
        return precision / len(users)

    def get_params(self, deep: bool = True) -> dict:
        base = super().get_params(deep)
        base["username_column"] = self.username_column
        base["affinity_column"] = self.affinity_column
        base["tracks_column"] = self.tracks_column
        base["features_columns"] = self.features_columns
        return base

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
