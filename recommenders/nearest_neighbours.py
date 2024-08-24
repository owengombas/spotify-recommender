from __future__ import annotations

from IPython.display import display
from lib import (
    spoti,
    plotting,
    dimensionality_reduction,
)
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from recommenders.metrics import similarities_score


class NearestNeighborsRecommender(SklearnNearestNeighbors, BaseEstimator):
    """
    Recommender system that uses nearest neighbors algorithm.

    Inherits from SklearnNearestNeighbors and BaseEstimator.

    Properties:
        n_neighbors (int): Number of neighbors to use for kneighbors queries.
        username_column (str): Column name in the DataFrame for username.
        affinity_column (str): Column name in the DataFrame for affinity.
        tracks_column (str): Column name in the DataFrame for tracks.
        features_columns (List[str]): List of column names to be used as features.
        dataframe (pd.DataFrame): DataFrame on which the model is fitted.
    """

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

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

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
        """
        Initializes the NearestNeighborsRecommender object.

        Args:
            n_neighbors (int, optional): Number of neighbors to use for kneighbors queries.
            metric (str, optional): The distance metric to use.
            p (int, optional): Power parameter for the Minkowski metric.
            leaf_size (int, optional): Leaf size passed to BallTree or KDTree.
            radius (float, optional): Range of parameter space to use by default for radius_neighbors queries.
            n_jobs (int, optional): The number of parallel jobs to run for neighbors search.
            algorithm (str, optional): Algorithm used to compute the nearest neighbors.
            metric_params (Optional[Dict[str, Any]], optional): Additional keyword arguments for the metric function.
            features_columns (List[str], optional): List of column names to be used as features.
            username_column (str, optional): Column name for usernames.
            affinity_column (str, optional): Column name for affinity.
            tracks_column (str, optional): Column name for tracks.
        """
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

    def fit(self, df: pd.DataFrame) -> NearestNeighborsRecommender:
        """
        Fit the model using df as training data.

        Args:
            df (pd.DataFrame): DataFrame to fit the model.

        Returns:
            NearestNeighborsRecommender: The fitted recommender model.
        """
        self.dataframe = df
        super().fit(df[self._features_columns])
        return self

    def pick_user_tracks(
        self,
        df: pd.DataFrame,
        username: str,
        sort_by_affinity: bool = True,
        max_tracks: int = -1,
    ) -> pd.DataFrame:
        """
        Picks tracks associated with a given user.

        Args:
            df (pd.DataFrame): DataFrame containing user-track information.
            username (str): Username for which to pick tracks.
            sort_by_affinity (bool, optional): Sort tracks by affinity if True. Defaults to True.
            max_tracks (int, optional): Maximum number of tracks to return. Defaults to -1 (all tracks).

        Returns:
            pd.DataFrame: DataFrame containing the user's tracks.
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
        Recommends tracks based on a given set of tracks.

        Args:
            tracks (pd.DataFrame): DataFrame containing tracks to base recommendations on.
            based_on (pd.DataFrame, optional): DataFrame to use for finding recommendations. Defaults to None (uses self.dataframe).
            n_neighbors (int, optional): Number of neighbors to use for recommendations. Defaults to None.
            sort_by_distance (bool, optional): Sort recommendations by distance if True. Defaults to True.
            filter_out_tracks (bool, optional): Exclude tracks in the 'tracks' DataFrame from the recommendations if True. Defaults to True.
            filter_out_if_seen (bool, optional): Exclude tracks seen by the user if True. Defaults to True.
            max_tracks (int, optional): Maximum number of recommended tracks to return. Defaults to -1 (all recommended tracks).

        Returns:
            pd.DataFrame: DataFrame containing recommended tracks.
        """
        if based_on is None:
            based_on = self.dataframe

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
        """
        Plots the recommendation space using the provided DataFrames.

        Args:
            df_recommended_from (pd.DataFrame): DataFrame from which recommendations are made.
            df_recommended_tracks (pd.DataFrame): DataFrame containing recommended tracks.
            username (str, optional): Username to highlight in the plot. Defaults to None.

        Returns:
            go.Figure: Plotly figure representing the recommendation space.
        """
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
        max_items: int = np.inf,
        n_neighbours: int = None,
        verbose: bool = False,
    ):
        df_user = df_train[df_train["username"] == username]
        df_user = df_user.drop_duplicates(subset=["id"])
        df_user = df_user.sort_values(by="affinity", ascending=False)
        recommendations = self.recommend_from(
            df_user,
            based_on=None,
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
        check_k = min(len(heldout_sorted), len(recommendations), max_items)
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
        max_items: int = np.inf,
        n_neighbors: int = None,
        verbose: bool = False,
    ):
        """
        Computes the retrievial score for the model for every user in df_test.

        Args:
            df_train (pd.DataFrame): DataFrame containing training data.
            df_test (pd.DataFrame): DataFrame containing test data.
            max_items (int, optional): Maximum number of items to consider. Defaults to np.inf.
            n_neighbors (int, optional): Number of neighbors to use for recommendations. Defaults to None.
            verbose (bool, optional): Print verbose output if True. Defaults to False.

        Returns:
            float: Retrieval score.
        """
        users = df_test["username"].unique()
        precision = 0
        for user in users:
            precision += self.retrievial_score_for_user(
                user,
                df_train,
                df_test,
                n_neighbours=n_neighbors,
                verbose=verbose,
                max_items=max_items,
            )
        return precision / len(users)

    def similarities_score_for_user(
        self,
        username: str,
        df_ground_truth: pd.DataFrame,
        n_neighbours: int = None,
        max_items: int = np.inf,
        verbose: bool = False,
    ):
        """
        Computes the similarities score for a given user.

        Args:
            username (str): Username for which to compute the similarities score.
            df_train (pd.DataFrame): DataFrame containing training data.
            df_test (pd.DataFrame): DataFrame containing test data.
            n_neighbours (int, optional): Number of neighbors to use for recommendations. Defaults to None.
            max_items (int, optional): Maximum number of items to consider. Defaults to np.inf.
            verbose (bool, optional): Print verbose output if True. Defaults to False.

        Returns:
            float: Similarities score.
        """
        # Get the user's tracks which is the ground truth
        df_user = df_ground_truth[df_ground_truth["username"] == username]
        df_user = df_user.sort_values(by="affinity", ascending=False)
        df_user = df_user.drop_duplicates(subset=["id"])

        # Get the recommendations for the user from the model
        recommendations = self.recommend_from(
            df_user,
            based_on=None,
            n_neighbors=n_neighbours,
            filter_out_if_seen=False,
            filter_out_tracks=False,
        )

        # Sort the recommendations by distance and remove duplicates
        recommendations.sort_values(by="distance", ascending=False, inplace=True)
        recommendations.drop_duplicates(subset=["id"], inplace=True)

        # Remove the user's tracks from the recommendations: df_user[:check_k]["id"]
        recommendations = recommendations[~recommendations["id"].isin(df_user["id"].head(max_items))]

        check_k = max_items
        if verbose:
            print(f"Check k: {check_k}")
            display(df_user[:check_k][spoti.PRETTY_PRINT_FEATURES])
            display(recommendations[:check_k][spoti.PRETTY_PRINT_FEATURES])

        # Convert the DataFrames to tensors
        tensor_heldout = torch.tensor(
            df_user[:check_k][self._features_columns].values
        )
        tensor_recommendations = torch.tensor(
            recommendations[:check_k][self._features_columns].values
        )

        assert len(tensor_heldout) == len(tensor_recommendations) == check_k, f"Lengths of tensors do not match: {len(tensor_heldout)}, {len(tensor_recommendations)}, {check_k}"

        # Compute the similarities score
        value = similarities_score(tensor_heldout, tensor_recommendations)

        if verbose:
            print(f"Similarities score: {value}")

        return value

    def similarities_score(
        self,
        df_ground_truth: pd.DataFrame,
        max_items: int = np.inf,
        n_neighbors: int = None,
        verbose: bool = False,
        users: List[str] = None,
    ):
        """
        Computes the similarities score for the model for every user in df_test.

        Args:
            df_train (pd.DataFrame): DataFrame containing training data.
            df_test (pd.DataFrame): DataFrame containing test data.
            max_items (int, optional): Maximum number of items to consider. Defaults to np.inf.
            n_neighbors (int, optional): Number of neighbors to use for recommendations. Defaults to None.
            verbose (bool, optional): Print verbose output if True. Defaults to False.

        Returns:
            float: Similarities score.
        """
        precision = 0
        for user in users:
            precision += self.similarities_score_for_user(
                user,
                df_ground_truth,
                n_neighbours=n_neighbors,
                verbose=verbose,
                max_items=max_items,
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
