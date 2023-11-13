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
        df["artists_names"] = df["artists"].apply(lambda x: x[0]["name"])
        df["artists_ids"] = df["artists"].apply(lambda x: x[0]["id"])
        return df

    def _preprocess_album(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df)
        df["album_name"] = df["album"].apply(lambda x: x["name"])
        df["album_release_date"] = df["album"].apply(lambda x: x["release_date"])
        df["album_total_tracks"] = df["album"].apply(lambda x: x["total_tracks"])
        return df
    
    def get_artists(self, artist_ids: List[str]) -> pd.DataFrame:
        artists = self.sp.artists(artist_ids)
        artists_df = pd.DataFrame(artists["artists"])
        return artists_df
    
    def top_tracks(self, limit: int = 20, offset: int = 0, time_range: str = "medium_term", wait_seconds: float = 2.) -> pd.DataFrame:
        results = self.sp.current_user_top_tracks(limit=limit, offset=offset, time_range=time_range)
        result_df = pd.DataFrame(results["items"])
        result_df = self._preprocess_artists(result_df)
        result_df = self._preprocess_album(result_df)
        result_df = self.populate_artist_infos(result_df, wait_seconds=wait_seconds)
        result_df = self.populate_track_features(result_df, wait_seconds=wait_seconds)
        return result_df
    
    def populate_track_features(self, tracks: pd.DataFrame, wait_seconds: float = 2.) -> pd.DataFrame:
        tracks_ids = tracks["id"].unique().tolist()
        chunks = [tracks_ids[x:x+100] for x in range(0, len(tracks_ids), 100)]

        tracks_features_df = pd.DataFrame()
        for chunk in chunks:
            tracks_features_df = pd.concat([tracks_features_df, pd.DataFrame(self.sp.audio_features(chunk))])
            time.sleep(wait_seconds)
        
        df = tracks.merge(tracks_features_df, left_on="id", right_on="id", suffixes=("", "_features"))
        return df
    
    def playlists(self, limit: int = 50, offset: int = 0) -> pd.DataFrame:
        result = self.sp.current_user_playlists(limit=limit, offset=offset)
        result_df = pd.DataFrame(result["items"])
        return result_df
    
    def populate_artist_infos(self, df: pd.DataFrame, wait_seconds: float = 2.) -> pd.DataFrame:
        unique_artists = df["artists_ids"].unique().tolist()
        chunks = [unique_artists[x:x+50] for x in range(0, len(unique_artists), 50)]

        artists_df = pd.DataFrame()
        for chunk in chunks:
            artists_df = pd.concat([artists_df, self.get_artists(chunk)])
            time.sleep(wait_seconds)

        df = df.merge(artists_df, left_on="artists_ids", right_on="id", suffixes=("", "_artists"))
        return df
    
    def liked_tracks(self, limit: int = 50, offset: int = 0, pages_max: int = 1, wait_seconds: float = 2.) -> pd.DataFrame:
        if pages_max is None: pages_max = np.inf

        result = self.sp.current_user_saved_tracks(limit=limit, offset=offset)
        
        items = list(map(lambda x: {**x["track"], "added_at": x["added_at"]}, result["items"]))
        if pages_max is not None:
            while(result["next"] and pages_max > 0):
                result_next = self.sp.next(result)
                result_data = map(lambda x: {**x["track"], "added_at": x["added_at"]}, result_next["items"])
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
