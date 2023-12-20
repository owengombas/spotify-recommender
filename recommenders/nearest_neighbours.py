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


class NearestNeighboursRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        n_neighbours: int = 10,
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
        self._df: pd.DataFrame = df.copy()
        self._n_neighbours: int = n_neighbours
        self._metric: str = metric
        self._username_column = username_column
        self._affinity_column = affinity_column
        self._features_columns = features_columns
        self._tracks_column = tracks_column

        self._model: SklearnNearestNeighbors = SklearnNearestNeighbors(
            n_neighbors=n_neighbours,
            metric=metric,
            p=p,
            leaf_size=leaf_size,
            radius=radius,
            n_jobs=n_jobs,
            algorithm=algorithm,
            metric_params=metric_params,
        )
        self._model.fit(df[features_columns])

    def pick_user_tracks(
        self, username: str, sort_by_affinity: bool = True, max_tracks: int = -1
    ) -> pd.DataFrame:
        """
        Returns the tracks of the given user.
        """
        user_df = self._df[self._df[self._username_column] == username]

        if sort_by_affinity and self._affinity_column in user_df.columns:
            user_df = user_df.sort_values(by=self._affinity_column, ascending=False)

        if max_tracks > 0:
            user_df = user_df.head(max_tracks)

        return user_df

    def recommend_from(
        self,
        tracks: pd.DataFrame,
        n_neighbours: int = None,
        sort_by_distance: bool = True,
        filter_out_tracks: bool = True,
        filter_out_if_seen: bool = True,
        max_tracks: int = -1,
    ) -> pd.DataFrame:
        """
        Returns the neighbours of the given tracks.
        """
        if n_neighbours is None:
            n_neighbours = self._n_neighbours

        distances, indices = self._model.kneighbors(tracks[self._features_columns], n_neighbours)
        df_neighbours = self._df.iloc[indices.flatten()].copy()
        df_neighbours["distance"] = distances.flatten()

        username = tracks[self._username_column].iloc[0]
        
        # Add an attribute seen to df indicating if the track has been seen by the user
        # check if user <username> has seen the track
        df_neighbours["seen"] = df_neighbours[self._tracks_column].isin(self._df[self._df[self._username_column] == username][self._tracks_column])

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
        username: str = None
    ) -> go.Figure:
        latent_columns = self._features_columns

        df_plot = df_recommended_from.copy()
        df_plot = pd.concat([df_plot, df_recommended_tracks])
        df_plot = df_plot.drop_duplicates(subset=self._tracks_column)

        if len(latent_columns) > 3:
            pca = PCA(n_components=3)
            latent_features = pca.fit_transform(df_plot[latent_columns])
            df_plot, latent_columns = dimensionality_reduction.add_latent_features(df_plot, latent_features)
        
        if username is not None:
            df_plot.loc[df_plot[self._username_column] != username, self._username_column] = ""

        return plotting.plot_latent_space(df_plot, color=df_plot[self._username_column], text=df_plot[self._username_column], title="Recommendation space")