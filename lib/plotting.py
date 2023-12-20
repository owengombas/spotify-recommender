import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lib import spoti
import matplotlib.colors as mcolors


def plot_feature_distribution(
    df: pd.DataFrame, numeric_features: List[str], cols: int = 5, rows: int = 2
) -> plt.Figure:
    """
    Plot the distribution of numeric features.
    :param df: The dataframe to plot.
    :param numeric_features: The numeric features to plot.
    :param cols: The number of columns in the plot.
    :param rows: The number of rows in the plot.
    :return: The figure.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    for i, feature in enumerate(numeric_features):
        sns.histplot(data=df, x=feature, ax=axes[i // cols, i % cols], kde=True)
    plt.tight_layout()
    return fig


def plot_latent_space(
    df: pd.DataFrame,
    color: pd.Series = None,
    text: pd.Series = None,
    title: str = None,
    latent_columns: List[str] = None,
) -> go.Figure:
    """
    Plot the latent space of a dataframe.
    :param df: The dataframe.
    :param color: The color of each point.
    :param text: The text of each point.
    :param title: The title of the plot.
    :return: The figure.
    """
    if latent_columns is None:
        latent_columns = list(filter(lambda x: "latent_" in x, df.columns))

    if len(latent_columns) == 2:
        fig = px.scatter(
            df,
            x=latent_columns[0],
            y=latent_columns[1],
            color=color,
            text=text,
            title=title,
            opacity=0.5,
        )
    elif len(latent_columns) == 3:
        fig = px.scatter_3d(
            df,
            x=latent_columns[0],
            y=latent_columns[1],
            z=latent_columns[2],
            color=color,
            text=text,
            title=title,
            opacity=0.5,
        )
    else:
        fig = px.scatter_matrix(
            df, dimensions=latent_columns, color=color, title=title, opacity=0.5
        )
    fig.update_traces(textposition="top center")
    return fig


def plot_genres(genres: pd.Series) -> go.Figure:
    """
    Plot the distribution of genres.
    :param genres: A series of lists of genres List[str].
    :return: The figure.
    """
    genre_counts = genres.explode().value_counts()
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Genre Distribution",
        labels={"x": "Genre", "y": "Count"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_column_distributions(
    sub_dfs: pd.DataFrame, numeric_features: List[str] = spoti.NUMERICAL_FEATURES
) -> plt.Figure:
    fig, axs = plt.subplots(
        len(sub_dfs),
        len(numeric_features),
        figsize=(len(numeric_features) * 5, len(sub_dfs) * 5),
    )

    i = 0
    for key, df in sub_dfs:
        for j, feature in enumerate(numeric_features):
            sns.histplot(data=df, x=feature, ax=axs[i, j], kde=True)
            axs[i, j].set_title(f"{key} - {feature}")
        i += 1
    plt.tight_layout()
    plt.xticks(rotation=90)

    return fig


def plot_multivariate_gaussian_image_with_labels(
    means, variances, labels, color_map="viridis", num_points=1000, figsize=(10, 10),
    title="Latent Space Representation", x_bounds = (-3, 3), y_bounds = (-3, 3), dpi=None
):
    # Create a grid of points
    x = np.linspace(
        np.min(means[:, 0]) + x_bounds[0] * np.sqrt(np.max(variances[:, 0])),
        np.max(means[:, 0]) + x_bounds[1] * np.sqrt(np.max(variances[:, 0])),
        num_points,
    )
    y = np.linspace(
        np.min(means[:, 1]) + y_bounds[0] * np.sqrt(np.max(variances[:, 1])),
        np.max(means[:, 1]) + y_bounds[1] * np.sqrt(np.max(variances[:, 1])),
        num_points,
    )
    X, Y = np.meshgrid(x, y)

    # Initialize a blank image
    image = np.zeros(X.shape)

    # Plot each gaussian
    for mean, variance in zip(means, variances):
        # Calculate the Z values (probabilities) for each X, Y in the grid
        Z = (
            1.0
            / (2.0 * np.pi * np.sqrt(variance[0] * variance[1]))
            * np.exp(
                -(
                    (X - mean[0]) ** 2 / (2 * variance[0])
                    + (Y - mean[1]) ** 2 / (2 * variance[1])
                )
            )
        )
        # Add the Z values to the image
        image += Z

    # Normalize the image to get values between 0 and 1
    image = image / np.max(image)

    # Plot the image
    if dpi is None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=figsize, dpi=dpi)

    plt.imshow(
        image,
        extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        origin="lower",
        cmap=color_map,
        norm=mcolors.PowerNorm(gamma=1.0 / 2.0),
    )

    # Add labels near each mean
    if labels != None:
        for mean, label in zip(means, labels):
            plt.text(
                mean[0] - 0.5,
                mean[1],
                label,
                fontsize=8,
                ha="right",
                va="bottom",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
            )

    plt.colorbar()
    plt.xlabel("Latent Variable 1")
    plt.ylabel("Latent Variable 2")
    plt.title(title)
    plt.show()

def plot_user_similarities_matrix(matrix: pd.DataFrame, figsize=(50, 50), lower_triangle=True):
    plt.figure(figsize=figsize)
    if lower_triangle:
        mask = np.triu(np.ones_like(matrix, dtype=bool))
    else:
        mask = np.identity(n=matrix.shape[0])
    sns.heatmap(matrix, annot=False, cmap='viridis', mask=mask)
    plt.title("User Track Common Percentage Heatmap")
    plt.ylabel("User")
    plt.xlabel("Other User")
    plt.show()
