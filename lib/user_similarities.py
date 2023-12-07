import pandas as pd


def calculate_users_track_common_percentage(data: pd.DataFrame, username_col: str, trackname_col: str) -> float:
    """
    Calculate the percentage of tracks that are common between users as a matrix.
    :param data: Dataframe containing the data.
    :param username_col: Name of the column containing the username.
    :param trackname_col: Name of the column containing the track name.
    :return: Percentage of tracks that are common between users as a matrix.
    """
    # Pre-compute the total number of tracks for each user
    user_total_tracks = data.groupby(username_col)[trackname_col].count()

    # Create a DataFrame where each row is a track and columns are users with boolean values indicating ownership
    track_user_matrix = data.pivot_table(index=trackname_col, columns=username_col, aggfunc='size', fill_value=0)

    # Compute the dot product of the matrix with its transpose to get shared track counts
    shared_tracks = track_user_matrix.T.dot(track_user_matrix)

    # Normalize by user total tracks to get the percentage
    user_track_share = shared_tracks.div(user_total_tracks, axis=0)

    return user_track_share


