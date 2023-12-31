import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from recommenders.metrics import (
    NDCG_binary_at_k_batch,
    mpr,
    NDCG_binary_at_k_batch_torch,
    Recall_at_k_batch_torch,
)
from torchmetrics.retrieval import RetrievalNormalizedDCG
import os


class LogisticMatrixFactorization(nn.Module):
    """
    Matrix Factorization model with logistic loss.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
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

        self.num_users, self.num_items = num_users, num_items
        self.num_factors = num_factors
        self.reg_param = reg_param
        self.device = device
        self.name_suffix = name_suffix
        self.dtype = dtype

        self.num_factors = num_factors

        self.user_vectors = nn.Parameter(
            torch.randn(
                self.num_users, num_factors, dtype=self.dtype, device=self.device
            ),
            requires_grad=False,
        )
        self.item_vectors = nn.Parameter(
            torch.randn(
                self.num_items, num_factors, dtype=self.dtype, device=self.device
            ),
            requires_grad=False,
        )
        self.user_biases = nn.Parameter(
            torch.randn(self.num_users, 1, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        self.item_biases = nn.Parameter(
            torch.randn(self.num_items, 1, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )

        self.losses = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.validation_losses = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.mpr_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.ndcg_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.recall20_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.recall50_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )

        self.to(self.device, self.dtype)

    def forward(
        self, R: torch.Tensor, user: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute derivatives for either users or items.
        :param user: If True, compute derivatives for users; otherwise, compute derivatives for items.
        :return: Tuple of derivatives for latent vectors and biases.
        """
        ones = torch.ones((R.shape[0], R.shape[1]), dtype=self.dtype, device=self.device)

        if user:
            vec_deriv = torch.matmul(R, self.item_vectors)
            bias_deriv = torch.unsqueeze(torch.sum(R, axis=1), dim=1)
        else:
            vec_deriv = torch.matmul(R.t(), self.user_vectors)
            bias_deriv = torch.unsqueeze(torch.sum(R, axis=0), dim=1)

        A = torch.mm(self.user_vectors, self.item_vectors.t())
        A += self.user_biases
        A += self.item_biases.t()
        A = torch.exp(A)
        A /= A + ones
        A = (R + ones) * A

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

    def log_likelihood(self, R: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((R.shape[0], R.shape[1]), dtype=self.dtype, device=self.device)

        loglik = 0
        A = torch.mm(self.user_vectors, self.item_vectors.t())
        A += self.user_biases
        A += self.item_biases.t()
        B = A * R
        loglik += torch.sum(B)

        A = torch.exp(A)
        A += ones

        A = torch.log(A)
        A = (R + ones) * A
        loglik -= torch.sum(A)

        # L2 regularization
        loglik -= 0.5 * self.reg_param * torch.sum(torch.square(self.user_vectors))
        loglik -= 0.5 * self.reg_param * torch.sum(torch.square(self.item_vectors))
        return loglik

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
            return (
                torch.matmul(self.user_vectors, self.item_vectors.T) + self.user_biases
            )
        else:
            return (
                torch.matmul(self.item_vectors, self.user_vectors.T) + self.item_biases
            )

    def train_model(
        self,
        R: torch.Tensor,
        num_epochs: int,
        learning_rate: float,
        validation_train: torch.Tensor,
        validation_test: torch.Tensor,
        verbose: bool = True,
        log_interval: int = 10,
        tqdm: bool = True,
    ) -> "LogisticMatrixFactorization":
        """
        Train the matrix factorization model.
        :param num_epochs: Number of training epochs.
        :param learning_rate: Learning rate for optimization.
        :param verbose: If True, print progress.
        :param log_interval: Interval for logging progress.
        """
        self.losses = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.mpr_history = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.ndcg_history = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.recall20_history = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.recall50_history = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.validation_losses = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        user_vec_deriv_sum = torch.zeros(
            (self.num_users, self.num_factors),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        item_vec_deriv_sum = torch.zeros(
            (self.num_items, self.num_factors),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        user_bias_deriv_sum = torch.zeros(
            (self.num_users, 1),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        item_bias_deriv_sum = torch.zeros(
            (self.num_items, 1),
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )

        best_ndcg = 0.0
        best_model: LogisticMatrixFactorization = None
        optimizer = optim.Adagrad(self.parameters(), lr=learning_rate)

        for epoch in tqdm(range(num_epochs)) if tqdm else range(num_epochs):
            optimizer.zero_grad()

            # Fix items and solve for users
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            user_vec_deriv, user_bias_deriv = self.forward(R, True)
            user_vec_deriv_sum += torch.square(user_vec_deriv)
            user_bias_deriv_sum += torch.square(user_bias_deriv)
            vec_step_size = learning_rate / torch.sqrt(user_vec_deriv_sum)
            bias_step_size = learning_rate / torch.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            # Fix users and solve for items
            # take step towards gradient of deriv of log likelihood
            # we take a step in positive direction because we are maximizing LL
            item_vec_deriv, item_bias_deriv = self.forward(R, False)
            item_vec_deriv_sum += torch.square(item_vec_deriv)
            item_bias_deriv_sum += torch.square(item_bias_deriv)
            vec_step_size = learning_rate / torch.sqrt(item_vec_deriv_sum)
            bias_step_size = learning_rate / torch.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv

            # We calculate the log-likelihood
            loglik = -self.log_likelihood(R)


            (
                validation_loss,
                ndcg_value,
                recall20_value,
                recall50_value,
                mpr_value,
            ) = self.evaluate_model(R, validation_train, validation_test)
            # We calculate the MPR
            self.losses[epoch] = loglik
            self.mpr_history[epoch] = mpr_value
            self.validation_losses[epoch] = validation_loss
            self.ndcg_history[epoch] = ndcg_value
            self.recall20_history[epoch] = recall20_value
            self.recall50_history[epoch] = recall50_value

            if ndcg_value > best_ndcg:
                best_ndcg = ndcg_value
                best_model = self.copy()

            if verbose and (epoch + 1) % log_interval == 0:
                print(
                    f"Epoch: {epoch+1} \t Loss: {loglik:.4f} \t MPR: {mpr_value:.4f} \t NDCG@100: {ndcg_value:.4f} \t Recall@20: {recall20_value:.4f} \t Recall@50: {recall50_value:.4f}"
                )
        return best_model

    def evaluate_model(
        self,
        R: torch.Tensor,
        validation_train: torch.Tensor,
        validation_test: torch.Tensor,
    ):
        """
        Evaluates the model on the validation set.
        :param validation_train: The training set of the validation split.
        :param validation_test: The test set of the validation split.
        :param top_k: The number of items to recommend.
        """
        self.eval()

        with torch.no_grad():
            recon_batch = self.predict_all(True)

            loss = -self.log_likelihood(validation_train)

            recon_batch[validation_train.nonzero(as_tuple=True)] = -np.inf

            NDCG = RetrievalNormalizedDCG(top_k=100, empty_target_action="skip")
            indexes = torch.arange(validation_test.size(0), device=self.device, dtype=torch.long).unsqueeze(1).expand(-1, validation_test.size(1))
            n100 = NDCG(recon_batch, validation_test, indexes)
            r20 = Recall_at_k_batch_torch(
                recon_batch, validation_test, 20, device=self.device, dtype=self.dtype
            )
            r50 = Recall_at_k_batch_torch(
                recon_batch, validation_test, 50, device=self.device, dtype=self.dtype
            )
            I = torch.argsort(self.predict_all(True), dim=1, descending=True)
            mpr_value = mpr(R, I, device=self.device, dtype=self.dtype)

        return loss, torch.nanmean(n100), torch.nanmean(r20), torch.nanmean(r50), mpr_value

    def recommend(
        self,
        R: torch.Tensor,
        user_id: int,
        top_k: int = 10,
        filter_user_items: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Recommends items for a user.
        :param user_id: The user id.
        :param top_k: The number of items to recommend.
        :param filter_user_items: If True, items already interacted by the user are not recommended.
        :return: Tuple of recommended items and their scores.
        """
        scores = self.predict_all(True)[user_id]
        if filter_user_items:
            scores[R[user_id].bool()] = -np.inf
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
        torch.save(
            self.state_dict(), os.path.join(path, f"{name}{self.name_suffix}.pt")
        )

    def load(self, path: str, name: str = "model") -> None:
        """
        Loads the model.
        :param path: The path to the directory where the model will be loaded from.
        :param name: The name of the model.
        """
        self.load_state_dict(
            torch.load(os.path.join(path, f"{name}{self.name_suffix}.pt"))
        )

    def copy(self):
        """
        Returns a copy of the model.
        :return: A copy of the model.
        """
        model = LogisticMatrixFactorization(
            self.num_users,
            self.num_items,
            self.num_factors,
            self.reg_param,
            self.device,
            self.dtype,
            self.name_suffix,
        )
        model.user_vectors = nn.Parameter(
            self.user_vectors.clone(), requires_grad=False
        )
        model.item_vectors = nn.Parameter(
            self.item_vectors.clone(), requires_grad=False
        )
        model.user_biases = nn.Parameter(self.user_biases.clone(), requires_grad=False)
        model.item_biases = nn.Parameter(self.item_biases.clone(), requires_grad=False)
        return model
