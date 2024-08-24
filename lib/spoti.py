from __future__ import annotations
import dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Union
import time
import os
from lib import genre_normalizer

DEFAULT_SCOPE = (
    "user-library-read "
    "user-read-currently-playing "
    "user-read-playback-state "
    "playlist-read-collaborative "
    "user-top-read "
    "user-read-recently-played "
    "user-library-read "
    "user-read-email "
)

NUMERICAL_FEATURES = [
    "danceability",
    "energy",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
    "release_year",
    "time_signature",
    "popularity",
]

TRACKS_TYPES = ["top_track", "liked_track", "playlist"]

PRETTY_PRINT_FEATURES = [
    "username",
    "artists_names",
    "name",
    "release_year",
    "popularity",
] + NUMERICAL_FEATURES


class SpotiUser:
    """
    Represents a Spotify user and provides methods to interact with the Spotify API.
    """

    @property
    def username(self) -> str:
        return self._username

    @property
    def sp(self) -> spotipy.Spotify:
        return self._sp

    def __init__(self, username: str) -> None:
        """
        Initialize a SpotiUser object.

        Args:
            username (str): Username of the user, it can be an alias it doesn't have to be the real username.

        Returns:
            None
        """
        self._username: str = username
        self._scope = DEFAULT_SCOPE
        self._sp: spotipy.Spotify = None

    def infos(self) -> pd.DataFrame:
        """
        Get the user's infos.

        Returns:
            pd.DataFrame: DataFrame containing the user's infos.
        """
        result = self.sp.current_user()
        return pd.DataFrame([result])

    def get_email(self) -> str:
        """
        Get the user's email.

        Returns:
            str: User's email.
        """
        return self.infos()["email"].iloc[0]

    def load_top_tracks(
        self,
        top_tracks_affinity_start: Dict[str, float] = {
            "short_term": 1.0,
            "medium_term": 1.0,
            "long_term": 1.0,
        },
        top_tracks_affinity_end: Dict[str, float] = {
            "short_term": 0,
            "medium_term": 0,
            "long_term": 0,
        },
        top_tracks_include_end=False,
        base_path: str = "data",
    ) -> pd.DataFrame:
        """
        Load the top tracks from the user (already downloaded from Spotify)

        Args:
            top_tracks_affinity_start (Dict[str, float], optional): Affinity start for top tracks. Defaults to {"short_term": 1.0, "medium_term": 1.0, "long_term": 1.0}.
            top_tracks_affinity_end (Dict[str, float], optional): Affinity end for top tracks. Defaults to {"short_term": 0, "medium_term": 0, "long_term": 0}.
            top_tracks_include_end (bool, optional): If True, includes the end value in the affinity range. Defaults to False.
            base_path (str, optional): Base path to load data from. Defaults to "data".

        Returns:
            pd.DataFrame: DataFrame containing the top tracks.
        """
        df: pd.DataFrame = pd.read_json(
            os.path.join(base_path, f"{self._username}_top_tracks.json")
        )

        for time_range, df_sub_time_range in df.groupby("time_range"):
            # Add affinity column, 1 to 0.1
            decreasing_affinity = np.linspace(
                top_tracks_affinity_start[time_range],
                top_tracks_affinity_end[time_range],
                num=len(df_sub_time_range),
                endpoint=top_tracks_include_end,
            )
            df.loc[df_sub_time_range.index, "affinity"] = decreasing_affinity

        df["type"] = "top_track"
        df["username"] = self.username
        df["release_year"] = df["album_release_date"].apply(
            lambda x: int(x.split("-")[0])
        )
        df["normalized_genres"] = genre_normalizer.normalize_genres(df["genres"])
        return df

    def load_liked_tracks(
        self,
        base_path: str = "data",
        start_affinity=1.0,
        end_affinity=0.1,
        include_endpoint=False,
    ) -> pd.DataFrame:
        """
        Load the liked tracks from the user (already downloaded from Spotify)

        Args:
            base_path (str, optional): Base path to load data from. Defaults to "data".
            start_affinity (float, optional): Affinity start for liked tracks. Defaults to 1.0.
            end_affinity (float, optional): Affinity end for liked tracks. Defaults to 0.1.
            include_endpoint (bool, optional): If True, includes the end value in the affinity range. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the liked tracks.
        """
        df = pd.read_json(
            os.path.join(base_path, f"{self._username}_liked_tracks.json")
        )
        df["added_at"] = pd.to_datetime(df["added_at"])
        df.sort_values(by="added_at", inplace=True)
        df["type"] = "liked_track"
        df["username"] = self.username
        df["release_year"] = df["album_release_date"].apply(
            lambda x: int(x.split("-")[0])
        )
        df["normalized_genres"] = genre_normalizer.normalize_genres(df["genres"])

        for time_range, df_sub_time_range in df.groupby("username"):
            # Add affinity column, 1 to 0.1
            decreasing_affinity = np.linspace(
                start_affinity,
                end_affinity,
                num=len(df_sub_time_range),
                endpoint=include_endpoint,
            )
            df.loc[df_sub_time_range.index, "affinity"] = decreasing_affinity

        return df

    def load_playlists(
        self, base_path: str = "data", affinity: float = 0.2
    ) -> pd.DataFrame:
        """
        Load the playlists from the user (already downloaded from Spotify)

        Args:
            base_path (str, optional): Base path to load data from. Defaults to "data".
            affinity (float, optional): Affinity for playlists. Defaults to 0.2.

        Returns:
            pd.DataFrame: DataFrame containing the playlists.
        """
        df = pd.read_json(
            os.path.join(base_path, f"{self._username}_playlists_tracks.json")
        )
        # df["added_at"] = pd.to_datetime(df["added_at"])
        df["type"] = "playlist"
        df["username"] = self.username
        df["release_year"] = df["album_release_date"].apply(
            lambda x: int(x.split("-")[0])
        )
        df["normalized_genres"] = genre_normalizer.normalize_genres(df["genres"])
        df["affinity"] = affinity
        return df

    def load_all(
        self,
        top_tracks_affinity_start: Dict[str, float] = {
            "short_term": 1.0,
            "medium_term": 1.0,
            "long_term": 1.0,
        },
        top_tracks_affinity_end: Dict[str, float] = {
            "short_term": 0,
            "medium_term": 0,
            "long_term": 0,
        },
        top_tracks_include_end=False,
        liked_tracks_start_affinity=1.0,
        liked_tracks_end_affinity=0.1,
        liked_tracks_include_endpoint=False,
        playlists_affinity=0.2,
        base_path: str = "data",
    ) -> pd.DataFrame:
        """
        Load all tracks from the user.

        Args:
            top_tracks_affinity_start (Dict[str, float], optional): Affinity start for top tracks. Defaults to {"short_term": 1.0, "medium_term": 1.0, "long_term": 1.0}.
            top_tracks_affinity_end (Dict[str, float], optional): Affinity end for top tracks. Defaults to {"short_term": 0, "medium_term": 0, "long_term": 0}.
            top_tracks_include_end (bool, optional): If True, includes the end value in the affinity range. Defaults to False.
            liked_tracks_start_affinity (float, optional): Affinity start for liked tracks. Defaults to 1.0.
            liked_tracks_end_affinity (float, optional): Affinity end for liked tracks. Defaults to 0.1.
            liked_tracks_include_endpoint (bool, optional): If True, includes the end value in the affinity range. Defaults to False.
            playlists_affinity (float, optional): Affinity for playlists. Defaults to 0.2.
            base_path (str, optional): Base path to load data from. Defaults to "data".

        Returns:
            pd.DataFrame: DataFrame containing all loaded tracks.
        """
        df_top_tracks = self.load_top_tracks(
            top_tracks_affinity_start=top_tracks_affinity_start,
            top_tracks_affinity_end=top_tracks_affinity_end,
            top_tracks_include_end=top_tracks_include_end,
            base_path=base_path,
        )
        df_liked_tracks = self.load_liked_tracks(
            base_path=base_path,
            start_affinity=liked_tracks_start_affinity,
            end_affinity=liked_tracks_end_affinity,
            include_endpoint=liked_tracks_include_endpoint,
        )
        df_playlists = self.load_playlists(
            base_path=base_path, affinity=playlists_affinity
        )

        df = pd.concat(
            [df_top_tracks, df_liked_tracks, df_playlists], ignore_index=True
        )
        df["username"] = self.username

        return df

    def get_auth_url(self) -> str:
        """
        Get the URL to authenticate the user, to can be send to the user.

        Returns:
            str: URL to authenticate the user
        """
        dotenv.load_dotenv()
        auth_manager = SpotifyOAuth(
            scope=self._scope,
            username=self._username,
            cache_path=f".cache-{self._username}.json",
        )
        return auth_manager.get_authorize_url()

    def log_user(self) -> spotipy.Spotify:
        """
        Log the user in and return a spotipy.Spotify object.

        Returns:
            spotipy.Spotify: Spotify object
        """
        dotenv.load_dotenv()
        auth_manager = SpotifyOAuth(
            scope=self._scope,
            username=self._username,
            cache_path=f".cache-{self._username}.json",
        )
        self._sp = spotipy.Spotify(auth_manager=auth_manager)

    def _preprocess_artists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the artists column of the DataFrame by extracting the artist name and ID.

        Args:
            df (pd.DataFrame): DataFrame containing the tracks

        Returns:
            pd.DataFrame: DataFrame containing the tracks with the artists column preprocessed
        """
        if "artists" not in df.columns:
            return df

        df["artists_names"] = df["artists"].apply(lambda x: x[0]["name"])
        df["artists_ids"] = df["artists"].apply(lambda x: x[0]["id"])
        return df

    def _preprocess_album(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the album column of the DataFrame by extracting the album name, release date and total tracks.

        Args:
            df (pd.DataFrame): DataFrame containing the tracks

        Returns:
            pd.DataFrame: DataFrame containing the tracks with the album column preprocessed
        """
        if "album" not in df.columns:
            return df

        df["album_name"] = df["album"].apply(lambda x: x["name"])
        df["album_release_date"] = df["album"].apply(lambda x: x["release_date"])
        df["album_total_tracks"] = df["album"].apply(lambda x: x["total_tracks"])
        return df

    def get_playlist_tracks(
        self, playlist_id: str, limit: int = 100, offset: int = 0
    ) -> pd.DataFrame:
        """
        Get the tracks from a playlist from Spotify.

        Args:
            playlist_id (str): Playlist ID
            limit (int, optional): Number of tracks to get per page. Defaults to 100.
            offset (int, optional): Offset to start from. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame containing the tracks.
        """
        first_result = self.sp.playlist_tracks(playlist_id, limit=limit, offset=offset)

        def add_tracks(x: Dict) -> Dict:
            if "track" not in x:
                return None
            if x["track"] is None:
                return None
            return {
                **x["track"],
                "added_at": x["added_at"],
                "username": self._username,
                "added_by": x["added_by"]["id"],
                "playlist_id": playlist_id,
            }

        tracks_df = pd.DataFrame([add_tracks(x) for x in first_result["items"]])
        if first_result["next"]:
            while first_result["next"]:
                result_next = self.sp.next(first_result)
                if result_next is None:
                    break
                result_data = pd.DataFrame(
                    list(
                        filter(
                            lambda x: x is not None,
                            [add_tracks(x) for x in result_next["items"]],
                        )
                    )
                )
                tracks_df = pd.concat([tracks_df, result_data])
                first_result["next"] = result_next["next"]
        tracks_df["username"] = self._username
        tracks_df.reset_index(inplace=True, drop=True)
        tracks_df.dropna(inplace=True)

        # remove duplicates
        tracks_df.drop_duplicates(subset=["id"], inplace=True)

        tracks_df = self._preprocess_artists(tracks_df)
        tracks_df = self._preprocess_album(tracks_df)
        tracks_df = self.populate_artist_infos(tracks_df)
        tracks_df = self.populate_track_features(tracks_df)

        return tracks_df

    def get_playlists(self, limit: int = 50, offset: int = 0) -> pd.DataFrame:
        """
        Get the user's playlists list from Spotify.

        Args:
            limit (int, optional): Number of playlists to get per page. Defaults to 50.
            offset (int, optional): Offset to start from. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame containing the playlists.
        """
        first_result = self.sp.current_user_playlists(limit=limit, offset=offset)
        playlists_df = pd.DataFrame(first_result["items"])
        if first_result["next"]:
            while first_result["next"]:
                result_next = self.sp.next(first_result)
                result_data = pd.DataFrame(result_next["items"])
                playlists_df = pd.concat([playlists_df, result_data])
                first_result["next"] = result_next["next"]
        playlists_df["username"] = self._username
        playlists_df.reset_index(inplace=True, drop=True)
        return playlists_df

    def get_artists(self, artist_ids: List[str]) -> pd.DataFrame:
        """
        Get artists infos from Spotify.

        Args:
            artist_ids (List[str]): List of artists IDs

        Returns:
            pd.DataFrame: DataFrame containing the artists infos
        """
        artists = self.sp.artists(artist_ids)
        artists_df = pd.DataFrame(artists["artists"])
        return artists_df

    def top_tracks(
        self,
        limit: int = 20,
        offset: int = 0,
        time_range: str = "medium_term",
        wait_seconds: float = 2.0,
    ) -> pd.DataFrame:
        """
        Get the user's top tracks from Spotify.

        Args:
            limit (int, optional): Number of tracks to get per page. Defaults to 20.
            offset (int, optional): Offset to start from. Defaults to 0.
            time_range (str, optional): Time range to get the tracks from. Defaults to "medium_term", can be "short_term", "medium_term" or "long_term".
            wait_seconds (float, optional): Number of seconds to wait between requests. Defaults to 2.0.

        Returns:
            pd.DataFrame: DataFrame containing the top tracks.
        """
        results = self.sp.current_user_top_tracks(
            limit=limit, offset=offset, time_range=time_range
        )
        result_df = pd.DataFrame(results["items"])
        result_df = self._preprocess_artists(result_df)
        result_df = self._preprocess_album(result_df)
        result_df = self.populate_artist_infos(result_df, wait_seconds=wait_seconds)
        result_df = self.populate_track_features(result_df, wait_seconds=wait_seconds)
        return result_df

    def populate_track_features(
        self, tracks: pd.DataFrame, wait_seconds: float = 2.0
    ) -> pd.DataFrame:
        """
        Populates the DataFrame with track features from Spotify.

        Args:
            tracks (pd.DataFrame): DataFrame containing the tracks
            wait_seconds (float, optional): Number of seconds to wait between requests. Defaults to 2.0.

        Returns:
            pd.DataFrame: DataFrame containing the tracks with track features
        """
        if "id" not in tracks.columns:
            return tracks

        tracks_ids = tracks["id"].unique().tolist()
        chunks = [tracks_ids[x : x + 100] for x in range(0, len(tracks_ids), 100)]

        tracks_features_df = pd.DataFrame()
        for chunk in chunks:
            tracks_features_df = pd.concat(
                [tracks_features_df, pd.DataFrame(self.sp.audio_features(chunk))]
            )
            time.sleep(wait_seconds)

        df = tracks.merge(
            tracks_features_df, left_on="id", right_on="id", suffixes=("", "_features")
        )
        return df

    def playlists(self, limit: int = 50, offset: int = 0) -> pd.DataFrame:
        """
        Get the user's playlists list from Spotify.

        Args:
            limit (int, optional): Number of playlists to get per page. Defaults to 50.
            offset (int, optional): Offset to start from. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame containing the playlists.
        """
        result = self.sp.current_user_playlists(limit=limit, offset=offset)
        result_df = pd.DataFrame(result["items"])
        return result_df

    def populate_artist_infos(
        self, df: pd.DataFrame, wait_seconds: float = 2.0
    ) -> pd.DataFrame:
        """
        Populates the DataFrame with artist infos from Spotify.

        Args:
            df (pd.DataFrame): DataFrame containing the tracks
            wait_seconds (float, optional): Number of seconds to wait between requests. Defaults to 2.0.

        Returns:
            pd.DataFrame: DataFrame containing the tracks with artist infos
        """
        if "artists_ids" not in df.columns:
            return df

        unique_artists = df["artists_ids"].unique().tolist()
        chunks = [unique_artists[x : x + 50] for x in range(0, len(unique_artists), 50)]

        artists_df = pd.DataFrame()
        for chunk in chunks:
            artists_df = pd.concat([artists_df, self.get_artists(chunk)])
            time.sleep(wait_seconds)

        df = df.merge(
            artists_df, left_on="artists_ids", right_on="id", suffixes=("", "_artists")
        )
        return df

    def liked_tracks(
        self,
        limit: int = 50,
        offset: int = 0,
        pages_max: int = 1,
        wait_seconds: float = 2.0,
    ) -> pd.DataFrame:
        """
        Get the user's liked tracks from Spotify.

        Args:
            limit (int, optional): Number of tracks to get per page. Defaults to 50.
            offset (int, optional): Offset to start from. Defaults to 0.
            pages_max (int, optional): Maximum number of pages to get. Defaults to 1.
            wait_seconds (float, optional): Number of seconds to wait between requests. Defaults to 2.0.

        Returns:
            pd.DataFrame: DataFrame containing the liked tracks.
        """
        if pages_max is None:
            pages_max = np.inf

        result = self.sp.current_user_saved_tracks(limit=limit, offset=offset)

        items = list(
            map(lambda x: {**x["track"], "added_at": x["added_at"]}, result["items"])
        )
        if pages_max is not None:
            while result["next"] and pages_max > 0:
                result_next = self.sp.next(result)
                result_data = map(
                    lambda x: {**x["track"], "added_at": x["added_at"]},
                    result_next["items"],
                )
                items.extend(result_data)
                result["next"] = result_next["next"]
                pages_max -= 1
                time.sleep(wait_seconds)

        result_df = pd.DataFrame(items)
        result_df = self._preprocess_artists(result_df)
        result_df = self._preprocess_album(result_df)
        result_df = self.populate_artist_infos(result_df, wait_seconds=wait_seconds)
        result_df = self.populate_track_features(result_df, wait_seconds=wait_seconds)

        return result_df


def load_all_tracks(
    base_path: str = "data",
    users: List[SpotiUser] = None,
    load_spotify_tracks: bool = True,
    top_tracks_affinity_start: Dict[str, float] = {
        "short_term": 1.0,
        "medium_term": 1.0,
        "long_term": 1.0,
    },
    top_tracks_affinity_end: Dict[str, float] = {
        "short_term": 0,
        "medium_term": 0,
        "long_term": 0,
    },
    top_tracks_include_end=False,
    liked_tracks_start_affinity=1.0,
    liked_tracks_end_affinity=0.1,
    liked_tracks_include_endpoint=False,
    playlists_affinity=0.2,
) -> pd.DataFrame:
    """
    Loads all tracks from specified users and, optionally, from Spotify's preloaded tracks.

    Args:
        base_path (str, optional): Base path to load data from. Defaults to "data".
        users (List[SpotiUser], optional): List of SpotiUser objects to load data for. Defaults to None.
        load_spotify_tracks (bool, optional): If True, loads Spotify's preloaded tracks. Defaults to True.
        top_tracks_affinity_start (Dict[str, float], optional): Affinity start for top tracks. Defaults to {"short_term": 1.0, "medium_term": 1.0, "long_term": 1.0}.
        top_tracks_affinity_end (Dict[str, float], optional): Affinity end for top tracks. Defaults to {"short_term": 0, "medium_term": 0, "long_term": 0}.
        top_tracks_include_end (bool, optional): If True, includes the end value in the affinity range. Defaults to False.
        liked_tracks_start_affinity (float, optional): Affinity start for liked tracks. Defaults to 1.0.
        liked_tracks_end_affinity (float, optional): Affinity end for liked tracks. Defaults to 0.1.
        liked_tracks_include_endpoint (bool, optional): If True, includes the end value in the affinity range. Defaults to False.
        playlists_affinity (float, optional): Affinity for playlists. Defaults to 0.2.

    Returns:
        pd.DataFrame: DataFrame containing all loaded tracks.
    """

    df = pd.DataFrame()

    for user in users:
        df_user = user.load_all(
            top_tracks_affinity_start=top_tracks_affinity_start,
            top_tracks_affinity_end=top_tracks_affinity_end,
            top_tracks_include_end=top_tracks_include_end,
            liked_tracks_start_affinity=liked_tracks_start_affinity,
            liked_tracks_end_affinity=liked_tracks_end_affinity,
            liked_tracks_include_endpoint=liked_tracks_include_endpoint,
            playlists_affinity=playlists_affinity,
            base_path=base_path,
        )
        df = pd.concat([df, df_user], ignore_index=True)

    if load_spotify_tracks:
        df_spotify = pd.read_json(
            os.path.join(base_path, "tracks.json"), orient="records"
        ).reset_index(drop=True)
        df_spotify["username"] = "Spotify"
        df_spotify["affinity"] = 0.0
        df = pd.concat([df, df_spotify], ignore_index=True)

    df = df.reset_index(drop=True)

    return df
