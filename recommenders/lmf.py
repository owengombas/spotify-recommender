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
    Matrix Factorization model with logistic loss.
    """

    def __init__(
        self,
        counts: torch.Tensor,
        num_factors: int,
        reg_param: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        name_suffix: str = "",
    ):
        """
        Constructor.
        :param R: The matrix of interactions.
        :param num_factors: The number of latent factors.
        :param alpha: The alpha parameter.
        :param lambd: The lambda parameter.
        :param device: The device for computations (e.g., "cpu" or "cuda").
        :param dtype: The data type for tensors (e.g., torch.float32 or torch.float64).
        :param name_suffix: A suffix to add to the model's name.
        """
        super(LogisticMatrixFactorization, self).__init__()

        self.counts = counts
        self.num_users, self.num_items = counts.shape
        self.num_factors = num_factors
        self.reg_param = reg_param
        self.device = device
        self.name_suffix = name_suffix
        self.dtype = dtype

        self.num_factors = num_factors

        self.ones = torch.ones((self.num_users, self.num_items), dtype=self.dtype, device=self.device, requires_grad=False)
        self.user_vectors = nn.Parameter(torch.randn(self.num_users, num_factors, dtype=self.dtype, device=self.device), requires_grad=False)
        self.item_vectors = nn.Parameter(torch.randn(self.num_items, num_factors, dtype=self.dtype, device=self.device), requires_grad=False)
        self.user_biases = nn.Parameter(torch.randn(self.num_users, 1, dtype=self.dtype, device=self.device), requires_grad=False)
        self.item_biases = nn.Parameter(torch.randn(self.num_items, 1, dtype=self.dtype, device=self.device), requires_grad=False)

        self.losses = torch.zeros(0, dtype=self.dtype, device=self.device, requires_grad=False)
        self.mprs = torch.zeros(0, dtype=self.dtype, device=self.device, requires_grad=False)

        self.to(self.device)

    def forward(self, user: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute derivatives for either users or items.
        :param user: If True, compute derivatives for users; otherwise, compute derivatives for items.
        :return: Tuple of derivatives for latent vectors and biases.
        """
        if user:
            vec_deriv = torch.matmul(self.counts, self.item_vectors)
            bias_deriv = torch.unsqueeze(torch.sum(self.counts, axis=1), dim=1)
        else:
            vec_deriv = torch.matmul(self.counts.t(), self.user_vectors)
            bias_deriv = torch.unsqueeze(torch.sum(self.counts, axis=0), dim=1)

        A = torch.mm(self.user_vectors, self.item_vectors.t())
        A += self.user_biases
        A += self.item_biases.t()
        A = torch.exp(A)
        A /= (A + self.ones)
        A = (self.counts + self.ones) * A

        if user:
            vec_deriv -= torch.mm(A, self.item_vectors)
            bias_deriv -= torch.unsqueeze(torch.sum(A, axis=1), dim=1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.user_vectors
        else:
            vec_deriv -= torch.mm(A.t(), self.user_vectors)
            bias_deriv -= torch.unsqueeze(torch.sum(A, axis=0), dim=1)
            # L2 regularization
            vec_deriv -= self.reg_param * self.item_vectors

        return vec_deriv, bias_deriv

    def log_likelihood(self):  
        loglik = 0
        A = torch.mm(self.user_vectors, self.item_vectors.t())
        A += self.user_biases
        A += self.item_biases.t()
        B = A * self.counts
        loglik += torch.sum(B)

        A = torch.exp(A)
        A += self.ones

        A = torch.log(A)
        A = (self.counts + self.ones) * A
        loglik -= torch.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * torch.sum(torch.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * torch.sum(torch.square(self.item_vectors))
        return loglik

    def mpr(self, I: torch.Tensor) -> float:
        num_users, num_items = self.counts.shape
        total_interactions = torch.sum(self.counts)

        percentile_ranks = (
            torch.arange(num_items, dtype=self.dtype, device=self.device) / num_items
        )

        weighted_ranks = self.counts.gather(1, I) * percentile_ranks.expand(num_users, -1)

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
            torch.matmul(self.user_vectors[user_id], self.item_vectors[item_id].T)
            + self.user_biases[user_id]
            + self.item_biases[item_id]
        )
    
    def predict_all(self, user: bool = True) -> torch.Tensor:
        """
        Predicts the interactions for all users or all items.
        :param user: If True, return predictions for all users; otherwise, return predictions for all items.
        :return: The matrix of interactions.
        """
        if user:
            return torch.matmul(self.user_vectors, self.item_vectors.T) + self.user_biases
        else:
            return torch.matmul(self.item_vectors, self.user_vectors.T) + self.item_biases

    def train_model(
        self,
        num_epochs: int,
        learning_rate: float,
        verbose: bool = True,
        log_interval: int = 10,
        tqdm: bool = True,
    ):
        """
        Train the matrix factorization model.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for optimization.
        :param verbose: If True, print progress.
        :param log_interval: Interval for logging progress.
        """
        self.losses = torch.zeros(num_epochs // log_interval, dtype=self.dtype, device=self.device, requires_grad=False)
        self.mprs = torch.zeros(num_epochs // log_interval, dtype=self.dtype, device=self.device, requires_grad=False)

        optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)

        user_vec_deriv_sum = torch.zeros((self.num_users, self.num_factors), device=self.device, dtype=self.dtype, requires_grad=False)
        item_vec_deriv_sum = torch.zeros((self.num_items, self.num_factors), device=self.device, dtype=self.dtype, requires_grad=False)
        user_bias_deriv_sum = torch.zeros((self.num_users, 1), device=self.device, dtype=self.dtype, requires_grad=False)
        item_bias_deriv_sum = torch.zeros((self.num_items, 1), device=self.device, dtype=self.dtype, requires_grad=False)
        for epoch in tqdm(range(num_epochs)) if tqdm else range(num_epochs):
            optimizer.zero_grad()

            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            user_vec_deriv, user_bias_deriv = self.forward(True)
            user_vec_deriv_sum += torch.square(user_vec_deriv)
            user_bias_deriv_sum += torch.square(user_bias_deriv)
            vec_step_size = learning_rate / torch.sqrt(user_vec_deriv_sum)
            bias_step_size = learning_rate / torch.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.forward(False)
            item_vec_deriv_sum += torch.square(item_vec_deriv)
            item_bias_deriv_sum += torch.square(item_bias_deriv)
            vec_step_size = learning_rate / torch.sqrt(item_vec_deriv_sum)
            bias_step_size = learning_rate / torch.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv

            if verbose and (epoch + 1) % log_interval == 0:
                # We calculate the log-likelihood
                loglik = -self.log_likelihood()

                # We calculate the MPR
                I = torch.argsort(self.predict_all(True), dim=1, descending=True)
                mpr = self.mpr(I)

                self.losses[epoch // log_interval] = loglik
                self.mprs[epoch // log_interval] = mpr

                print(
                    f"Epoch: {epoch+1}, log-likelihood: {loglik:.4f}, MPR: {mpr:.4f}"
                )
    
    def recommend(self, user_id: int, top_k: int = 10, filter_user_items: bool = True) -> Tuple[List[int], List[float]]:
        """
        Recommends items for a user.
        :param user_id: The user id.
        :param top_k: The number of items to recommend.
        :param filter_user_items: If True, items already interacted by the user are not recommended.
        :return: Tuple of recommended items and their scores.
        """
        scores = self.predict_all(True)[user_id]
        if filter_user_items:
            scores[self.counts[user_id].bool()] = -np.inf
        top_k_scores, top_k_items = torch.topk(scores, top_k)
        return top_k_items.cpu().numpy(), top_k_scores.cpu().numpy()
    
    def get_item_factors(self, item_ids: List[int]) -> np.ndarray:
        """
        Returns the factors for a list of items.
        :param item_ids: The list of item ids.
        :return: The item factors.
        """
        return self.item_vectors[item_ids].detach().cpu().numpy()
    
    def get_user_factors(self, user_ids: List[int]) -> np.ndarray:
        """
        Returns the factors for a list of users.
        :param user_ids: The list of user ids.
        :return: The user factors.
        """
        return self.user_vectors[user_ids].detach().cpu().numpy()

    def save(self, path: str, name: str = "model") -> None:
        """
        Saves the model.
        :param path: The path to the directory where the model will be saved.
        :param name: The name of the model.
        """
        torch.save(self.state_dict(), os.path.join(path, f"{name}{self.name_suffix}.pt"))
    
    def load(self, path: str, name: str = "model") -> None:
        """
        Loads the model.
        :param path: The path to the directory where the model will be loaded from.
        :param name: The name of the model.
        """
        self.load_state_dict(torch.load(os.path.join(path, f"{name}{self.name_suffix}.pt")))
