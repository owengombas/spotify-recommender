from IPython.display import display
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from recommenders.metrics import (
    mpr,
    Recall_at_k_batch_torch,
    similarities_score,
)
from torchmetrics.retrieval import RetrievalNormalizedDCG
import os


class LogisticMatrixFactorization(nn.Module):
    """
    Implements a Matrix Factorization model with a logistic loss for recommendation systems.

    Attributes:
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.
        num_factors (int): Number of latent factors for matrix factorization.
        reg_param (float): Regularization parameter.
        device (str): Device on which computations are performed (e.g., 'cpu' or 'cuda').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float32 or torch.float64).
        name_suffix (str): Suffix added to the model's name for identification.
        user_vectors (torch.Tensor): User latent factor matrix.
        item_vectors (torch.Tensor): Item latent factor matrix.
        user_biases (torch.Tensor): User bias vector.
        item_biases (torch.Tensor): Item bias vector.
        losses (torch.Tensor): Tensor storing loss values during training.
        validation_losses (torch.Tensor): Tensor storing validation loss values during training.
        mpr_history (torch.Tensor): Tensor storing Mean Percentile Rank values during training.
        ndcg_history (torch.Tensor): Tensor storing Normalized Discounted Cumulative Gain values during training.
        recall20_history (torch.Tensor): Tensor storing Recall@20 values during training.
        recall50_history (torch.Tensor): Tensor storing Recall@50 values during training.
        similarity_history (torch.Tensor): Tensor storing similarity scores during training.
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
        Initializes the LogisticMatrixFactorization model.

        Args:
            num_users (int): Number of users in the dataset.
            num_items (int): Number of items in the dataset.
            num_factors (int): Number of latent factors for matrix factorization.
            reg_param (float): Regularization parameter.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float64.
            name_suffix (str, optional): Suffix for the model's name. Defaults to an empty string.
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
        self.similarity_train_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )
        self.similarity_val_history = torch.zeros(
            0, dtype=self.dtype, device=self.device, requires_grad=False
        )

        self.to(self.device, self.dtype)

    def forward(
        self, R: torch.Tensor, user: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model to compute derivatives for either users or items.

        Args:
            R (torch.Tensor): Interaction matrix.
            user (bool, optional): If True, computes derivatives for users; else for items. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Derivatives for latent vectors and biases.
        """
        ones = torch.ones(
            (R.shape[0], R.shape[1]), dtype=self.dtype, device=self.device
        )

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
        """
        Computes the log-likelihood of the model given the interaction matrix.

        Args:
            R (torch.Tensor): Interaction matrix.

        Returns:
            torch.Tensor: Computed log-likelihood.
        """

        ones = torch.ones(
            (R.shape[0], R.shape[1]), dtype=self.dtype, device=self.device
        )

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

        Args:
            user (bool, optional): If True, return predictions for all users; otherwise, return predictions for all items. Defaults to True.

        Returns:    
            torch.Tensor: The matrix of interactions.
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
        validation_df_features: pd.DataFrame = None,
        validation_features_columns: List[str] = None,
        validation_users_train: List[str] = None,
        validation_users_val: List[str] = None,
        similarity_max_item: int = 50,
    ) -> "LogisticMatrixFactorization":
        """
        Trains the matrix factorization model.

        Args:
            R (torch.Tensor): Interaction matrix for training.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.
            validation_train (torch.Tensor): Training set of the validation split.
            validation_test (torch.Tensor): Test set of the validation split.
            verbose (bool, optional): If True, prints progress. Defaults to True.
            log_interval (int, optional): Interval for logging progress. Defaults to 10.
            tqdm (bool, optional): If True, uses tqdm for progress bar. Defaults to True.
            validation_df_features (pd.DataFrame, optional): DataFrame of features for validation. Defaults to None.
            validation_features_columns (List[str], optional): List of feature column names for validation. Defaults to None.
            similarity_max_item (int, optional): Maximum number of items for similarity computation. Defaults to 50.

        Returns:
            LogisticMatrixFactorization: Trained model.
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
        self.similarity_train_history = torch.zeros(
            num_epochs,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        self.similarity_val_history = torch.zeros(
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

        best_val = -np.inf
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
                similarity_train,
                similarity_val,
            ) = self.evaluate_model(
                R,
                validation_train,
                validation_test,
                validation_df_features,
                validation_features_columns,
                similarity_max_item,
                users_train=validation_users_train,
                users_val=validation_users_val,
            )
            # We calculate the MPR
            self.losses[epoch] = loglik
            self.mpr_history[epoch] = mpr_value
            self.validation_losses[epoch] = validation_loss
            self.ndcg_history[epoch] = ndcg_value
            self.recall20_history[epoch] = recall20_value
            self.recall50_history[epoch] = recall50_value
            self.similarity_train_history[epoch] = similarity_train
            self.similarity_val_history[epoch] = similarity_val

            if similarity_val > best_val:
                best_val = similarity_val
                best_model = self.copy()
                if verbose:
                    print(
                        f"Best model updated at epoch {epoch+1}, with score {best_val}"
                    )

            if verbose and (epoch + 1) % log_interval == 0:
                print(
                    f"Epoch: {epoch+1} \t Loss: {loglik:.4f} \t MPR: {mpr_value:.4f} \t NDCG@100: {ndcg_value:.4f} \t Recall@20: {recall20_value:.4f} \t Recall@50: {recall50_value:.4f} \t Similarity (train): {similarity_train:.4f} \t Similarity (val): {similarity_val:.4f}"
                )

        return best_model

    def evaluate_model(
        self,
        R: torch.Tensor,
        validation_train: torch.Tensor,
        validation_test: torch.Tensor,
        df_features: pd.DataFrame,
        features_columns: List[str],
        similarity_max_item: int,
        users_train: List[str] = None,
        users_val: List[str] = None,
    ):
        """
        Evaluates the model on the validation set.

        Args:
            R (torch.Tensor): Interaction matrix for training.
            validation_train (torch.Tensor): Training set of the validation split.
            validation_test (torch.Tensor): Test set of the validation split.
            df_features (pd.DataFrame): DataFrame of features for validation.
            features_columns (List[str]): List of feature column names for validation.
            similarity_max_item (int): Maximum number of items for similarity computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of loss, NDCG@100, Recall@20, Recall@50, MPR, similarity train and validation scores.
        """
        self.eval()

        with torch.no_grad():
            recon_batch = self.predict_all(True)

            loss = -self.log_likelihood(validation_train)

            recon_batch[validation_train.nonzero(as_tuple=True)] = -np.inf

            NDCG = RetrievalNormalizedDCG(top_k=100, empty_target_action="skip")
            indexes = (
                torch.arange(
                    validation_test.size(0), device=self.device, dtype=torch.long
                )
                .unsqueeze(1)
                .expand(-1, validation_test.size(1))
            )
            n100 = NDCG(recon_batch, validation_test, indexes)
            r20 = Recall_at_k_batch_torch(
                recon_batch, validation_test, 20, device=self.device, dtype=self.dtype
            )
            r50 = Recall_at_k_batch_torch(
                recon_batch, validation_test, 50, device=self.device, dtype=self.dtype
            )
            I = torch.argsort(recon_batch, dim=1, descending=True)
            mpr_value = mpr(R, I, device=self.device, dtype=self.dtype)

            users_train_acc = []
            users_val_acc = []
            users = users_train + users_val
            for user_id in users:
                # Get the ground truth tracks for the user
                user_tracks_df = df_features[
                    df_features["username"].cat.codes == user_id
                ]
                user_tracks_df = user_tracks_df.sort_values(
                    by="affinity", ascending=False
                )
                user_tracks_features = torch.tensor(
                    user_tracks_df[features_columns].to_numpy(),
                    dtype=self.dtype,
                    device=self.device,
                )

                # Perform recommendations
                predicted_tracks_scores = recon_batch[user_id]
                predicted_tracks_idx = (
                    torch.argsort(predicted_tracks_scores, descending=True)
                    .cpu()
                    .detach()
                    .numpy()
                )
                predicted_tracks_scores = predicted_tracks_scores.cpu().detach().numpy()

                # Add the scores to the predicted tracks
                predicted_tracks_features_df = df_features.copy()
                predicted_tracks_features_df["item_id"] = df_features[
                    "id"
                ].cat.codes.astype("int")
                predicted_tracks_features_df.set_index("item_id", inplace=True)
                predicted_tracks_features_df = predicted_tracks_features_df.iloc[
                    predicted_tracks_idx
                ]
                predicted_tracks_features_df["score"] = pd.Series(
                    predicted_tracks_scores,
                    index=predicted_tracks_features_df.index,
                )

                # Remove duplicates and sort by score
                predicted_tracks_features_df = predicted_tracks_features_df.sort_values(
                    by="score", ascending=False
                )
                predicted_tracks_features_df.drop_duplicates(
                    subset=["id"], inplace=True
                )
                # Remove the tracks that the user already listened to
                predicted_tracks_features_df = predicted_tracks_features_df[
                    ~predicted_tracks_features_df["id"].isin(
                        user_tracks_df["id"].head(similarity_max_item)
                    )
                ]

                # Get the predicted tracks features
                predicted_tracks_features = torch.tensor(
                    predicted_tracks_features_df[features_columns].to_numpy(),
                    dtype=self.dtype,
                    device=self.device,
                )

                # Make sure that the number of tracks is the same
                predicted_tracks_features = predicted_tracks_features[
                    :similarity_max_item
                ]
                user_tracks_features = user_tracks_features[:similarity_max_item]

                assert (
                    len(predicted_tracks_features)
                    == len(user_tracks_features)
                    == similarity_max_item
                ), f"Number of tracks is not {similarity_max_item} ({len(predicted_tracks_features)} and {len(user_tracks_features)})"

                # Calculate the similarity score between the ground truth and the predicted tracks
                similarity_score = similarities_score(
                    user_tracks_features,
                    predicted_tracks_features,
                )

                if user_id in users_train:
                    users_train_acc.append(similarity_score.cpu().detach().item())

                if user_id in users_val:
                    users_val_acc.append(similarity_score.cpu().detach().item())

            similarity_train = np.nanmean(users_train_acc)
            similarity_val = np.nanmean(users_val_acc)

        return (
            loss,
            torch.nanmean(n100),
            torch.nanmean(r20),
            torch.nanmean(r50),
            mpr_value,
            similarity_train,
            similarity_val,
        )

    def recommend(
        self,
        R: torch.Tensor,
        user_id: int,
        top_k: int = 10,
        filter_user_items: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Recommends items for a user.

        Args:
            R (torch.Tensor): Interaction matrix.
            user_id (int): The user id.
            top_k (int, optional): The number of items to recommend. Defaults to 10.
            filter_user_items (bool, optional): If True, items already interacted by the user are not recommended. Defaults to True.

        Returns:
            Tuple[List[int], List[float]]: Tuple of recommended items and their scores.
        """
        scores = self.predict_all(True)[user_id]
        if filter_user_items:
            scores[R[user_id].bool()] = -np.inf
        top_k_scores, top_k_items = torch.topk(scores, top_k)
        return top_k_items.cpu().numpy(), top_k_scores.cpu().numpy()

    def get_item_factors(self, item_ids: List[int]) -> np.ndarray:
        """
        Returns the factors for a list of items.

        Args:
            item_ids (List[int]): The list of item ids.

        Returns:
            np.ndarray: The item factors.
        """
        return self.item_vectors[item_ids].detach().cpu().numpy()

    def get_user_factors(self, user_ids: List[int]) -> np.ndarray:
        """
        Returns the factors for a list of users.

        Args:
            user_ids (List[int]): The list of user ids.

        Returns:
            np.ndarray: The user factors.
        """
        return self.user_vectors[user_ids].detach().cpu().numpy()

    def save(self, path: str, name: str = "model") -> None:
        """
        Saves the model.

        Args:
            path (str): The path to the directory where the model will be saved.
            name (str, optional): The name of the model. Defaults to "model".
        """
        torch.save(
            self.state_dict(), os.path.join(path, f"{name}{self.name_suffix}.pt")
        )

    def load(self, path: str, name: str = "model") -> None:
        """
        Loads the model.

        Args:
            path (str): The path to the directory where the model will be loaded from.
            name (str, optional): The name of the model. Defaults to "model".
        """
        self.load_state_dict(
            torch.load(os.path.join(path, f"{name}{self.name_suffix}.pt"))
        )

    def copy(self):
        """
        Returns a copy of the model.

        Returns:
            LogisticMatrixFactorization: Copy of the model.
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
