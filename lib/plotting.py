import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lib import spoti
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator


def plot_feature_distribution(
    df: pd.DataFrame, numeric_features: List[str], cols: int = 5, rows: int = 2
) -> plt.Figure:
    """
    Plots the distribution of numeric features in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        numeric_features (List[str]): List of numeric feature names to plot.
        cols (int): Number of columns in the plot grid.
        rows (int): Number of rows in the plot grid.

    Returns:
        plt.Figure: Matplotlib figure object with the distribution plots.
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
    Plots the latent space of a DataFrame using Plotly.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        color (pd.Series, optional): Series to color the points.
        text (pd.Series, optional): Text to display at each point.
        title (str, optional): Title of the plot.
        latent_columns (List[str], optional): List of columns representing latent dimensions.

    Returns:
        go.Figure: Plotly figure object for the latent space plot.
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
    Plots a bar chart of genre distributions.

    Args:
        genres (pd.Series): Series of genres.

    Returns:
        go.Figure: Plotly figure object for the genre distribution plot.
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
    """
    Plots distributions of numeric columns for different subsets of a DataFrame.

    Args:
        sub_dfs (pd.DataFrame): DataFrame containing subsets of data.
        numeric_features (List[str]): List of numeric feature names.

    Returns:
        plt.Figure: Matplotlib figure object with the distribution plots.
    """
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
    means,
    variances,
    labels,
    color_map="viridis",
    num_points=1000,
    figsize=(10, 10),
    title="Latent Space Representation",
    x_bounds=None,
    y_bounds=None,
    dpi=None,
):
    """
    Plots a multivariate Gaussian image with labels.

    Args:
        means: Mean coordinates for Gaussian distributions.
        variances: Variance for each Gaussian distribution.
        labels: Labels for each Gaussian distribution.
        color_map (str): Color map for the plot.
        num_points (int): Number of points in the grid.
        figsize (tuple): Size of the figure.
        title (str): Title of the plot.
        x_bounds (tuple, optional): Bounds for the x-axis.
        y_bounds (tuple, optional): Bounds for the y-axis.
        dpi (optional): Dots per inch for the figure.

    Returns:
        None: This function does not return a value but shows a plot.
    """
    # Create a grid of points
    if x_bounds is None:
        x_bounds = (
            np.min(means[:, 0]) - 1 * np.sqrt(np.max(variances[:, 0])),
            np.max(means[:, 0]) + 1 * np.sqrt(np.max(variances[:, 0])),
        )
    if y_bounds is None:
        y_bounds = (
            np.min(means[:, 1]) - 1 * np.sqrt(np.max(variances[:, 1])),
            np.max(means[:, 1]) + 1 * np.sqrt(np.max(variances[:, 1])),
        )
    x = np.linspace(
        x_bounds[0],
        x_bounds[1],
        num_points,
    )
    y = np.linspace(
        y_bounds[0],
        y_bounds[1],
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


def plot_user_similarities_matrix(
    matrix: pd.DataFrame,
    figsize=(50, 50),
    lower_triangle=True,
    title="User Similarities Matrix",
    log_scale=False,
):
    """
    Plots a heatmap of the user similarities matrix.

    Args:
        matrix (pd.DataFrame): DataFrame representing the user similarities matrix.
        figsize (tuple): Size of the figure.
        lower_triangle (bool): Whether to plot only the lower triangle of the matrix.
        title (str): Title of the plot.
        log_scale (bool): Whether to use logarithmic scale.

    Returns:
        None: This function does not return a value but shows a plot.
    """
    matrix = matrix.to_numpy()
    matrix[matrix == 0] = 1e-4
    plt.figure(figsize=figsize)
    # Drop rows that contains NaN values
    if lower_triangle:
        mask = np.triu(np.ones_like(matrix, dtype=bool))
    else:
        mask = np.identity(n=matrix.shape[0])
    sns.heatmap(
        matrix,
        annot=False,
        cmap="viridis",
        mask=mask,
        norm=LogNorm(vmin=matrix.min().min(), vmax=matrix.max().max(), clip=True)
        if log_scale
        else None
    )
    plt.title(title)
    plt.ylabel("User")
    plt.xlabel("Other User")
    plt.show()
