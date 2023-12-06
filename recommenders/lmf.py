import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class LogisticMatrixFactorization(nn.Module):
    """
    Logistic Matrix Factorization model.
    """

    @property
    def name(self) -> str:
        s = f"LMF_"
        s += f"{self.num_users}_"
        s += f"{self.num_items}_"
        s += f"{self.num_factors}_"
        s += f"{self.alpha}_"
        s += f"{self.lambd}_"
        s += f"{self.device}_"
        s += f"{self.name_suffix}"

        if s[-1] == "_":
            s = s[:-1]

        return s

    def __init__(
        self,
        R: torch.tensor,
        num_factors: int,
        alpha: float,
        lambd: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        name_suffix: str = "",
    ):
        """
        Constructor.
        :param num_users: The number of users.
        :param num_items: The number of items.
        :param num_factors: The number of latent factors.
        """
        super(LogisticMatrixFactorization, self).__init__()

        self.num_users, self.num_items = R.shape
        self.num_factors = num_factors
        self.alpha = alpha
        self.lambd = lambd
        self.device = device
        self.name_suffix = name_suffix
        self.dtype = dtype

        # Initialize the latent vectors
        self.R = R
        self.X = torch.randn(
            self.num_users, self.num_factors, dtype=dtype, device=device
        )
        self.Y = torch.randn(
            self.num_items, self.num_factors, dtype=dtype, device=device
        )

        # Initialize the biases
        self.beta_u = torch.randn(self.num_users, dtype=dtype, device=device)
        self.beta_i = torch.randn(self.num_items, dtype=dtype, device=device)

        self.mprs = torch.zeros(0, dtype=dtype, device=device)
        self.losses = torch.zeros(0, dtype=dtype, device=device)

    def loss(self) -> torch.Tensor:
        """
        Computes the loss function.
        :param R: The matrix of interactions.
        :param alpha: The alpha parameter.
        :param lambd: The lambda parameter.
        :return: The loss value.
        """
        # Compute the product x_u * y_i^T + beta_u + beta_i for all user-item pairs
        user_item_interactions = (
            torch.matmul(self.X, self.Y.t())
            + self.beta_u[:, None]
            + self.beta_i[None, :]
        )

        # Compute the log posterior term by term
        log_posterior = torch.sum(
            self.alpha * self.R * user_item_interactions
            - (1 + self.alpha * self.R) * torch.log1p(torch.exp(user_item_interactions))
        )

        # Subtract regularization terms
        log_posterior -= (
            self.lambd / 2 * (torch.sum(self.X**2) + torch.sum(self.Y**2))
        )

        return -log_posterior.item()

    def mpr(self, I: torch.Tensor) -> float:
        num_users, num_items = self.R.shape
        total_interactions = torch.sum(self.R)

        # Create a tensor for percentile ranks (shape: num_items)
        percentile_ranks = (
            torch.arange(num_items, dtype=self.dtype, device=self.device) / num_items
        )

        # Use I to index into R and rearrange it, then multiply by the percentile ranks
        # Reshape and expand percentile ranks to match the shape of R
        # (shape of expanded percentile ranks: 1 x num_items)
        weighted_ranks = self.R.gather(1, I) * percentile_ranks.expand(num_users, -1)

        # Sum over all items and users, and normalize
        mpr = torch.sum(weighted_ranks) / total_interactions

        return mpr.item()

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predicts the interaction for a user and an item.
        :param user_id: The user id.
        :param item_id: The item id.
        :return: The interaction.
        """
        return (
            torch.matmul(self.X[user_id], self.Y[item_id].t())
            + self.beta_u[user_id]
            + self.beta_i[item_id]
        )

    def predict_all(self) -> torch.Tensor:
        """
        Predicts the interactions for all users and items.
        :return: The interaction matrix.
        """
        return (
            torch.matmul(self.X, self.Y.t())
            + self.beta_u[:, None]
            + self.beta_i[None, :]
        )

    def get_user_latent_factors(self, user_id: int) -> torch.Tensor:
        """
        Returns the latent factors for a user.
        :param user_id: The user id.
        :return: The latent factors.
        """
        return self.X[user_id]

    def get_item_latent_factors(self, item_id: int) -> torch.Tensor:
        """
        Returns the latent factors for an item.
        :param item_id: The item id.
        :return: The latent factors.
        """
        return self.Y[item_id]

    def get_user_bias(self, user_id: int) -> float:
        """
        Returns the bias for a user.
        :param user_id: The user id.
        :return: The bias.
        """
        return self.beta_u[user_id]

    def get_item_bias(self, item_id: int) -> float:
        """
        Returns the bias for an item.
        :param item_id: The item id.
        :return: The bias.
        """
        return self.beta_i[item_id]

    def recommend(
        self, user_id: int, top_k: int = 10, filter_user_items: bool = True
    ) -> Tuple[List[int], List[float]]:
        """
        Recommends the top k items for a user.
        :param user_id: The user id.
        :param top_k: The number of items to recommend.
        :param filter_user_items: Whether to filter the items that the user has already interacted with.
        :return: The list of item ids and the list of scores.
        """
        # Get the predictions for the user
        predictions = self.predict_all()[user_id]

        # Filter the items that the user has already interacted with
        if filter_user_items:
            predictions[self.R[user_id] > 0] = float("-inf")

        # Get the top k items
        top_k_items = torch.topk(predictions, top_k)

        return top_k_items.indices.tolist(), top_k_items.values.tolist()

    def train_auto_gradients(
        self, num_epochs: int, learning_rate: float, verbose: bool = True
    ):
        """
        Trains the model.
        :param num_epochs: The number of epochs.
        :param learning_rate: The learning rate.
        :param verbose: Whether to print the loss and MPR.
        """
        self.X.requires_grad_(True)
        self.Y.requires_grad_(True)
        self.beta_u.requires_grad_(True)
        self.beta_i.requires_grad_(True)

        # Initialize the optimizer which is adagrad
        optimizer = optim.Adagrad(
            [self.X, self.Y, self.beta_u, self.beta_i], lr=learning_rate
        )

        # Iterate over the epochs
        for epoch in tqdm(range(num_epochs)):
            # Fix X and B and take a step toward Y and B
            self.X.requires_grad = False
            self.Y.requires_grad = True
            self.beta_u.requires_grad = False
            self.beta_i.requires_grad = True
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()

            # Fix Y and B and take a step toward X and B
            self.X.requires_grad = True
            self.Y.requires_grad = False
            self.beta_u.requires_grad = True
            self.beta_i.requires_grad = False
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()

            # Compute the predictions
            predictions = self.predict_all()

            # Compute the MPR
            I = torch.argsort(predictions, descending=True)
            mpr = self.mpr(I)

            # Save the loss and MPR
            self.losses.append(loss)
            self.mprs.append(mpr)

            if verbose:
                # Print the loss and MPR
                print(f"Epoch {epoch + 1}: loss = {loss}, MPR = {mpr}")

        self.X.requires_grad_(False)
        self.Y.requires_grad_(False)
        self.beta_u.requires_grad_(False)
        self.beta_i.requires_grad_(False)

    def train_with_gradients(
        self,
        num_epochs: int,
        learning_rate: float,
        verbose: bool = True,
        log_interval: int = 10,
    ):
        """
        Trains the model.
        :param num_epochs: The number of epochs.
        :param learning_rate: The learning rate.
        :param verbose: Whether to print the loss and MPR.
        """
        # Initialize user and item latent factor matrices and bias vectors
        self.X.requires_grad_(False)
        self.Y.requires_grad_(False)
        self.beta_u.requires_grad_(False)
        self.beta_i.requires_grad_(False)

        grad_accumulator_X = torch.zeros_like(
            self.X, dtype=self.dtype, device=self.device
        )
        grad_accumulator_Y = torch.zeros_like(
            self.Y, dtype=self.dtype, device=self.device
        )

        self.mprs = torch.zeros(
            num_epochs // log_interval, dtype=self.dtype, device=self.device
        )
        self.losses = torch.zeros(
            num_epochs // log_interval, dtype=self.dtype, device=self.device
        )

        for epoch in range(num_epochs):
            # Iterative approach to computing gradients
            #
            # Fix X and B and take a step toward Y and B
            # for i in range(self.num_items):
            #     term1 = self.alpha * self.R[:, i][:, None]

            #     exp_term = torch.exp(
            #         torch.matmul(self.Y[i], self.X.t()) + self.beta_u + self.beta_i[i]
            #     )
            #     exp_term = exp_term[:, None]

            #     gradients_Y = (
            #         torch.sum(
            #             term1 * self.X
            #             - self.X * (1 + term1) * exp_term / (1 + exp_term),
            #             dim=0,
            #         )
            #         - self.lambd * self.Y[i]
            #     )
            #     gradients_beta_i = torch.sum(
            #         term1 - (1 + term1) * exp_term / (1 + exp_term), dim=0
            #     ).squeeze()

            #     grad_accumulator_Y[i] += gradients_Y**2
            #     self.Y[i] += (
            #         learning_rate * gradients_Y / torch.sqrt(grad_accumulator_Y[i])
            #     )
            #     self.beta_i[i] += learning_rate * gradients_beta_i

            # # Fix Y and B and take a step toward X and B
            # for u in range(self.num_users):
            #     term1 = self.alpha * self.R[u, :][:, None]

            #     exp_term = torch.exp(
            #         torch.matmul(self.X[u], self.Y.t()) + self.beta_u[u] + self.beta_i
            #     )
            #     exp_term = exp_term[:, None]

            #     gradients_X = (
            #         torch.sum(
            #             term1 * self.Y
            #             - self.Y * (1 + term1) * exp_term / (1 + exp_term),
            #             dim=0,
            #         )
            #         - self.lambd * self.X[u]
            #     )
            #     gradients_beta_u = torch.sum(
            #         term1 - (1 + term1) * exp_term / (1 + exp_term), dim=0
            #     ).squeeze()

            #     grad_accumulator_X[u] += gradients_X**2
            #     self.X[u] += (
            #         learning_rate * gradients_X / torch.sqrt(grad_accumulator_X[u])
            #     )
            #     self.beta_u[u] += learning_rate * gradients_beta_u

            # Vectorized computation for Y and beta_i updates
            term1_Y = self.alpha * self.R.t()[:, :, None]
            exp_term_Y = torch.exp(
                torch.matmul(self.Y, self.X.t()) + self.beta_u + self.beta_i[:, None]
            )
            exp_term_Y = exp_term_Y[:, :, None]

            gradients_Y = (
                torch.sum(
                    term1_Y * self.X[None, :, :]
                    - self.X[None, :, :]
                    * (1 + term1_Y)
                    * exp_term_Y
                    / (1 + exp_term_Y),
                    dim=1,
                )
                - self.lambd * self.Y
            )
            gradients_beta_i = torch.sum(
                term1_Y - (1 + term1_Y) * exp_term_Y / (1 + exp_term_Y), dim=1
            ).squeeze()

            grad_accumulator_Y += gradients_Y**2
            self.Y += learning_rate * gradients_Y / torch.sqrt(grad_accumulator_Y)
            self.beta_i += learning_rate * gradients_beta_i

            # Vectorized computation for X and beta_u updates
            term1_X = self.alpha * self.R[:, :, None]
            exp_term_X = torch.exp(
                torch.matmul(self.X, self.Y.t()) + self.beta_u[:, None] + self.beta_i
            )
            exp_term_X = exp_term_X[:, :, None]

            gradients_X = (
                torch.sum(
                    term1_X * self.Y[None, :, :]
                    - self.Y[None, :, :]
                    * (1 + term1_X)
                    * exp_term_X
                    / (1 + exp_term_X),
                    dim=1,
                )
                - self.lambd * self.X
            )
            gradients_beta_u = torch.sum(
                term1_X - (1 + term1_X) * exp_term_X / (1 + exp_term_X), dim=1
            ).squeeze()

            grad_accumulator_X += gradients_X**2
            self.X += learning_rate * gradients_X / torch.sqrt(grad_accumulator_X)
            self.beta_u += learning_rate * gradients_beta_u

            if epoch % log_interval == 0:
                loss = self.loss()
                self.losses[epoch // log_interval] = loss

                predictions = self.predict_all()
                I = torch.argsort(predictions, descending=True, dim=1)
                mpr = self.mpr(I)
                self.mprs[epoch // log_interval] = mpr

                if verbose:
                    print(f"Epoch {epoch + 1}: loss = {loss}, MPR = {mpr}")

    def save(self, path: str, name: str = None):
        """
        Saves the model.
        :param path: The path to the file.
        :param name: The name of the file.
        """
        if name == None:
            name = self.name

        torch.save(
            {
                "R": self.R,
                "num_factors": self.num_factors,
                "alpha": self.alpha,
                "lambd": self.lambd,
                "X": self.X,
                "Y": self.Y,
                "beta_u": self.beta_u,
                "beta_i": self.beta_i,
                "mprs": self.mprs,
                "losses": self.losses,
                "device": self.device,
                "dtype": self.dtype,
                "name_suffix": self.name_suffix,
            },
            os.path.join(path, f"{name}.pt"),
        )

    @staticmethod
    def load(path: str):
        """
        Loads the model.
        :param path: The path to the file.
        :return: The model.
        """
        checkpoint = torch.load(path)
        model = LogisticMatrixFactorization(
            checkpoint["R"],
            checkpoint["num_factors"],
            checkpoint["alpha"],
            checkpoint["lambd"],
            checkpoint["device"],
            checkpoint["dtype"],
            checkpoint["name_suffix"],
        )
        model.X = checkpoint["X"]
        model.Y = checkpoint["Y"]
        model.beta_u = checkpoint["beta_u"]
        model.beta_i = checkpoint["beta_i"]
        model.mprs = checkpoint["mprs"]
        model.losses = checkpoint["losses"]

        return model
