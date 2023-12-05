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
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms",
    "release_year",
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
    @property
    def username(self) -> str:
        return self._username

    @property
    def sp(self) -> spotipy.Spotify:
        return self._sp

    def __init__(self, username: str) -> None:
        self._username: str = username
        self._scope = DEFAULT_SCOPE
        self._sp: spotipy.Spotify = None
    
    def infos(self) -> pd.DataFrame:
        result = self.sp.current_user()
        return pd.DataFrame([result])
    
    def get_email(self) -> str:
        return self.infos()["email"].iloc[0]

    def load_top_tracks(
        self,
        penality_factors: Dict[str, float] = {
            "short_term": 1.0,
            "medium_term": 1.0,
            "long_term": 1.0,
        },
        base_path: str = "data",
    ) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_json(
            os.path.join(base_path, f"{self._username}_top_tracks.json")
        )
        for time_range, df_sub_time_range in df.groupby("time_range"):
            # Add affinity column, to goes from 0 to 1, it's just a incremental index
            size = len(df_sub_time_range)
            penality_factor = penality_factors[time_range]
            df_sub_time_range["affinity"] = range(1, size + 1)[::-1]
            df.loc[df_sub_time_range.index, "affinity"] = (
                df_sub_time_range["affinity"] * penality_factor
            ) / size
        df["type"] = "top_track"
        df["username"] = self.username
        df["release_year"] = df["album_release_date"].apply(
            lambda x: int(x.split("-")[0])
        )
        df["normalized_genres"] = genre_normalizer.normalize_genres(df["genres"])
        return df

    def load_liked_tracks(self, base_path: str = "data") -> pd.DataFrame:
        df = pd.read_json(
            os.path.join(base_path, f"{self._username}_liked_tracks.json")
        )
        df["added_at"] = pd.to_datetime(df["added_at"])
        df["type"] = "liked_track"
        df["username"] = self.username
        df["release_year"] = df["album_release_date"].apply(
            lambda x: int(x.split("-")[0])
        )
        df["normalized_genres"] = genre_normalizer.normalize_genres(df["genres"])
        return df

    def load_playlists(self, base_path: str = "data") -> pd.DataFrame:
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
        return df

    def load_all(
        self,
        penality_factors: Dict[str, float] = {
            "short_term": 1.0,
            "medium_term": 1.0,
            "long_term": 1.0,
        },
        base_path: str = "data",
    ) -> pd.DataFrame:
        df_top_tracks = self.load_top_tracks(
            penality_factors=penality_factors, base_path=base_path
        )
        df_liked_tracks = self.load_liked_tracks(base_path=base_path)
        df_playlists = self.load_playlists(base_path=base_path)

        df = pd.concat(
            [df_top_tracks, df_liked_tracks, df_playlists], ignore_index=True
        )
        df["username"] = self.username

        return df

    def get_auth_url(self) -> str:
        dotenv.load_dotenv()
        auth_manager = SpotifyOAuth(
            scope=self._scope,
            username=self._username,
            cache_path=f".cache-{self._username}.json",
        )
        return auth_manager.get_authorize_url()

    def log_user(self) -> spotipy.Spotify:
        dotenv.load_dotenv()
        auth_manager = SpotifyOAuth(
            scope=self._scope,
            username=self._username,
            cache_path=f".cache-{self._username}.json",
        )
        self._sp = spotipy.Spotify(auth_manager=auth_manager)

    def _preprocess_artists(self, df: pd.DataFrame) -> pd.DataFrame:
        if "artists" not in df.columns:
            return df

        df["artists_names"] = df["artists"].apply(lambda x: x[0]["name"])
        df["artists_ids"] = df["artists"].apply(lambda x: x[0]["id"])
        return df

    def _preprocess_album(self, df: pd.DataFrame) -> pd.DataFrame:
        if "album" not in df.columns:
            return df

        df["album_name"] = df["album"].apply(lambda x: x["name"])
        df["album_release_date"] = df["album"].apply(lambda x: x["release_date"])
        df["album_total_tracks"] = df["album"].apply(lambda x: x["total_tracks"])
        return df

    def get_playlist_tracks(
        self, playlist_id: str, limit: int = 100, offset: int = 0
    ) -> pd.DataFrame:
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
        result = self.sp.current_user_playlists(limit=limit, offset=offset)
        result_df = pd.DataFrame(result["items"])
        return result_df

    def populate_artist_infos(
        self, df: pd.DataFrame, wait_seconds: float = 2.0
    ) -> pd.DataFrame:
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
    penality_factors: Dict[str, float] = {
        "short_term": 1.0,
        "medium_term": 1.0,
        "long_term": 1.0,
    },
) -> pd.DataFrame:
    df = pd.DataFrame()

    if load_spotify_tracks:
        df = pd.read_json(
            os.path.join(base_path, "tracks.json"), orient="records"
        ).reset_index(drop=True)
        df["username"] = "Spotify"

    for user in users:
        df_user = user.load_all(penality_factors=penality_factors, base_path=base_path)
        df = pd.concat([df, df_user], ignore_index=True)

    df = df.reset_index(drop=True)

    return df
