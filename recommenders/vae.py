import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from typing import Tuple
from recommenders.metrics import (
    mpr,
    Recall_at_k_batch_torch,
    mpr,
    similarities_score,
)
from torchmetrics.retrieval import RetrievalNormalizedDCG
import pandas as pd
import copy


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    anneal: torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Computes the loss function for the VAE.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Input.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.
        anneal (torch.Tensor, optional): Annealing factor for KL divergence. Defaults to 1.0.

    Returns:
        torch.Tensor: Loss value.
    """
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD


class MultiVAE(nn.Module):
    """
    Variational Autoencoder with Multinomial Likelihood (Multi-VAE).

    This VAE is designed for collaborative filtering tasks. The architecture
    comprises of encoder (q-network) and decoder (p-network) layers. The encoder
    produces a latent representation, and the decoder reconstructs the input data.

    Args:
        encoder_dims (List[int]): Dimensions for the encoder layers.
        decoder_dims (List[int]): Dimensions for the decoder layers.
        dropout_rate (float): Dropout rate for regularization.

    References:
        [1] Liang, Dawen, et al. "Variational autoencoders for collaborative filtering."
            Proceedings of the 2018 World Wide Web Conference. 2018.
        [2] Variational autoencoders for collaborative filtering
            by @dawenl.
            https://github.com/dawenl/vae_cf
        [3] Variational Autoencoders for Collaborative Filtering - Implementation in PyTorch
            by @younggyoseo.
            https://github.com/younggyoseo/vae-cf-pytorch
    """

    def __init__(
        self, p_dims: List[int], q_dims: Optional[List[int]] = None, dropout=0.5
    ):
        super(MultiVAE, self).__init__()
        self.p_dims: List[int] = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers: nn.ModuleList = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def train_model(
        self,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        num_epochs: int = 10,
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200000,
        log_interval=10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        validation_dataloader: DataLoader = None,
        validation_features_df: pd.DataFrame = None,
        validation_user_item_matrix: torch.Tensor = None,
        validation_features_columns: List[str] = None,
        validation_similarity_max_item: int = 100,
        validation_users_train: List[str] = None,
        validation_users_val: List[str] = None,
    ) -> Tuple[
        "MultiVAE", torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Trains the model on the given dataset.

        Args:
            dataset (Dataset): Dataset object.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            epochs (int, optional): Number of epochs for training. Defaults to 10.
            lr (float, optional): Learning rate for training. Defaults to 0.001.
            anneal (float, optional): Annealing factor for KL divergence. Defaults to 0.0.
            weight_decay (float, optional): Weight decay for regularization. Defaults to 0.0.
            device (str, optional): Device to use for training. Defaults to "cpu".
        """
        self.to(device, dtype)

        loss_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        validation_loss_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        ndcg_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        r20_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        r50_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        mpr_history = torch.zeros(num_epochs, dtype=dtype, device=device)
        similarities_train_score_history = torch.zeros(
            num_epochs, dtype=dtype, device=device
        )
        similarities_validation_score_history = torch.zeros(
            num_epochs, dtype=dtype, device=device
        )

        l = len(train_dataloader)

        best_model_state: collections.OrderedDict = None
        best_val_loss = -np.inf

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.train()
            for batch in train_dataloader:
                batch = batch[0].to(device=device, dtype=dtype)
                anneal = anneal_cap
                if total_anneal_steps > 0:
                    anneal_cap = min(anneal_cap, 1.0 * epoch / total_anneal_steps)

                optimizer.zero_grad()
                recon_batch, mu, logvar = self(batch)
                loss = loss_function(recon_batch, batch, mu, logvar, anneal)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            loss_history[epoch] = epoch_loss / l

            # Evaluate model
            if validation_dataloader:
                self.eval()
                (
                    total_loss,
                    ndcg,
                    r20,
                    r50,
                    mpr_value,
                    similarities_train_score,
                    similarities_validation_score,
                ) = self.evaluate_model(
                    validation_dataloader=validation_dataloader,
                    anneal_cap=anneal_cap,
                    total_anneal_steps=total_anneal_steps,
                    df_features=validation_features_df,
                    features_columns=validation_features_columns,
                    user_item_matrix=validation_user_item_matrix,
                    similarity_max_item=validation_similarity_max_item,
                    users_train=validation_users_train,
                    users_validation=validation_users_val,
                    device=device,
                    dtype=dtype,
                )
                validation_loss_history[epoch] = total_loss
                ndcg_history[epoch] = ndcg
                r20_history[epoch] = r20
                r50_history[epoch] = r50
                mpr_history[epoch] = mpr_value
                similarities_train_score_history[epoch] = similarities_train_score
                similarities_validation_score_history[
                    epoch
                ] = similarities_validation_score

                if similarities_validation_score > best_val_loss:
                    best_val_loss = similarities_validation_score
                    # copy the model state
                    best_model_state = copy.deepcopy(self.state_dict())
                    print(
                        f"Best model found at epoch {epoch} with score: {similarities_validation_score.item():.6f}"
                    )

                if epoch % log_interval == 0:
                    print(
                        f"Epoch: {epoch} \t Loss: {loss_history[epoch].item():.6f} \t Val Loss: {validation_loss_history[epoch].item():.6f} \t NDCG_100: {ndcg.item():.6f} \t Recall_20: {r20.item():.6f} \t Recall_50: {r50.item():.6f} \t MPR: {mpr_value.item():.6f}, \t Similarity Score (train): {similarities_validation_score.item():.6f} \t Similarity Score (validation): {similarities_validation_score.item():.6f}"
                    )

        best_model = MultiVAE(
            p_dims=self.p_dims, q_dims=self.q_dims, dropout=self.drop.p
        )
        best_model.load_state_dict(best_model_state)
        best_model.to(device, dtype)
        best_model.eval()

        return (
            best_model,
            loss_history,
            validation_loss_history,
            ndcg_history,
            r20_history,
            r50_history,
            mpr_history,
            similarities_train_score_history,
            similarities_validation_score_history,
        )

    def evaluate_model(
        self,
        validation_dataloader: DataLoader,
        user_item_matrix: torch.Tensor,
        df_features: pd.DataFrame,
        features_columns: List[str],
        users_train: List[str],
        users_validation: List[str],
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200000,
        similarity_max_item: int = 100,
        users: List[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Evaluate the model on the given dataset

        Args:
            validation_dataloader (DataLoader): The validation dataloader.
            user_item_matrix (torch.Tensor): The user-item interaction matrix.
            df_features (pd.DataFrame): The dataframe containing the features.
            features_columns (List[str]): The list of features columns.
            anneal_cap (float, optional): Annealing factor for KL divergence. Defaults to 0.2.
            total_anneal_steps (int, optional): Total number of annealing steps. Defaults to 200000.
            similarity_max_item (int, optional): Number of items to consider for similarity score. Defaults to 100.
            validation_users (List[str], optional): The list of users to consider for the similarity score. Defaults to None.
            device (str, optional): The device to use. Defaults to "cpu".
            dtype (torch.dtype, optional): The dtype to use. Defaults to torch.float32.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The loss, NDCG, Recall@20, Recall@50, MPR and similarity score.
        """
        self.to(device, dtype)
        self.eval()
        total_loss = 0.0
        n100_list = []
        r20_list = []
        r50_list = []
        mpr_list = []
        similarity_train_score_list = []
        similarity_validation_score_list = []
        update_count = 0
        N = len(validation_dataloader.dataset)
        batch_size = validation_dataloader.batch_size

        with torch.no_grad():
            for batch in validation_dataloader:
                validation_train, validation_test = batch
                validation_train: torch.Tensor = validation_train.to(
                    device=device, dtype=dtype
                )
                validation_test: torch.Tensor = validation_test.to(
                    device=device, dtype=dtype
                )

                anneal = anneal_cap
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)

                recon_batch, mu, logvar = self(validation_train)

                loss = loss_function(recon_batch, validation_train, mu, logvar, anneal)
                total_loss += loss.item()

                recon_batch[validation_train.nonzero(as_tuple=True)] = -np.inf

                NDCG = RetrievalNormalizedDCG(top_k=100, empty_target_action="skip")
                indexes = (
                    torch.arange(
                        validation_test.size(0), device=device, dtype=torch.long
                    )
                    .unsqueeze(1)
                    .expand(-1, validation_test.size(1))
                )
                n100 = NDCG(recon_batch, validation_test, indexes)
                r20 = Recall_at_k_batch_torch(
                    recon_batch, validation_test, 20, device=device, dtype=dtype
                )
                r50 = Recall_at_k_batch_torch(
                    recon_batch, validation_test, 50, device=device, dtype=dtype
                )

                predictions_idx, predictions = self.recommend(validation_train)
                mpr_score = mpr(
                    validation_test,
                    predictions_idx,
                    device=device,
                    dtype=dtype,
                )

                mpr_list.append(mpr_score)
                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)

                update_count += 1

        users = users_train + users_validation
        for index, user_id in enumerate(users):
            user_row = user_item_matrix[user_id]
            # Retrieve the ground truth
            user_tracks_df = df_features[
                df_features["username"].cat.codes == user_id.item()
            ]
            user_tracks_df = user_tracks_df.sort_values(by="affinity", ascending=False)
            user_tracks_features = torch.tensor(
                user_tracks_df[features_columns].to_numpy(), dtype=dtype, device=device
            )

            # Get the recommendations for the user
            mu, logvar = self.encode(user_row.unsqueeze(0))
            z = self.reparameterize(mu, logvar)
            predicted_scores = self.decode(z)
            predicted_scores = predicted_scores.cpu().flatten()
            # sort the items by score
            sorted_predicted_idx = torch.argsort(predicted_scores, descending=True)
            sorted_predicted_scores = (
                torch.gather(predicted_scores, 0, sorted_predicted_idx)
                .cpu()
                .detach()
                .numpy()
            )
            sorted_predicted_idx = sorted_predicted_idx.cpu().detach().numpy()

            # Add the scores to the dataframe
            predicted_tracks_features_df = df_features.copy()
            predicted_tracks_features_df["item_id"] = df_features[
                "id"
            ].cat.codes.astype("int")
            predicted_tracks_features_df.set_index("item_id", inplace=True)
            predicted_tracks_features_df = predicted_tracks_features_df.iloc[
                sorted_predicted_idx
            ]
            predicted_tracks_features_df["score"] = pd.Series(
                sorted_predicted_scores,
                index=predicted_tracks_features_df.index,
            )

            # Remove duplicates
            predicted_tracks_features_df = predicted_tracks_features_df.sort_values(
                by="score", ascending=False
            )
            predicted_tracks_features_df.drop_duplicates(subset=["id"], inplace=True)

            # Remove the tracks that the user already listened to
            predicted_tracks_features_df = predicted_tracks_features_df[
                ~predicted_tracks_features_df["id"].isin(
                    user_tracks_df["id"].head(similarity_max_item)
                )
            ]

            # Get the features of the recommended tracks
            predicted_tracks_features = torch.tensor(
                predicted_tracks_features_df[features_columns].to_numpy(),
                dtype=dtype,
                device=device,
            )

            # Make sure that the number of tracks is the same
            user_tracks_features = user_tracks_features[:similarity_max_item]
            predicted_tracks_features = predicted_tracks_features[:similarity_max_item]

            assert (
                len(predicted_tracks_features)
                == len(user_tracks_features)
                == similarity_max_item
            ), f"Number of tracks is not {similarity_max_item} ({len(predicted_tracks_features)} and {len(user_tracks_features)})"

            # Calculate the similarity score
            similarity_score = similarities_score(
                user_tracks_features,
                predicted_tracks_features,
            )

            if user_id in users_train:
                similarity_train_score_list.append(
                    similarity_score.cpu().detach().item()
                )

            if user_id in users_validation:
                similarity_validation_score_list.append(
                    similarity_score.cpu().detach().item()
                )

        total_loss /= len(range(0, N, batch_size))
        n100_list = torch.tensor(n100_list, dtype=dtype, device=device)
        r20_list = torch.cat(r20_list)
        r50_list = torch.cat(r50_list)
        mpr_list = torch.tensor(mpr_list, dtype=dtype, device=device)
        similarity_train_score_list = torch.tensor(
            similarity_train_score_list, dtype=dtype, device=device
        )
        similarity_validation_score_list = torch.tensor(
            similarity_validation_score_list, dtype=dtype, device=device
        )

        return (
            total_loss,
            torch.nanmean(n100_list),
            torch.nanmean(r20_list),
            torch.nanmean(r50_list),
            torch.nanmean(mpr_list),
            torch.nanmean(similarity_train_score_list),
            torch.nanmean(similarity_validation_score_list),
        )

    def get_latent_representations(
        self, matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(matrix)
            z = self.reparameterize(mu, logvar)
            std = torch.exp(0.5 * logvar)
        return z, mu, std, logvar

    def sort_similar_users(
        self, user_id: int, latent_representations: torch.Tensor
    ) -> torch.Tensor:
        user_z = latent_representations[user_id]
        # Compute cosine similarity between user_z and all other users
        distances = torch.nn.CosineSimilarity(dim=-1)(latent_representations, user_z)
        # Sort by similarity
        sorted_distances = torch.argsort(distances, descending=True)
        return sorted_distances

    def recommend(
        self,
        matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate recommendations for a given user.

        Args:
            user_id (int): The ID of the user for whom to generate recommendations.
            matrix (torch.Tensor): The user-item interaction matrix.
            latent_representations (torch.Tensor): Latent representations of users.
            filter_items (torch.Tensor, optional): Items to be filtered out. Defaults to None.
            top_similar_users (int, optional): Number of top similar users to consider. Defaults to -1 (not used).
            num_items (int, optional): Number of items to recommend. Defaults to -1 (all).

        Returns:
            torch.Tensor: Indices of recommended items.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(matrix)
            z = self.reparameterize(mu, logvar)
            predicted_scores = self.decode(z)

        recommended_items = torch.argsort(predicted_scores, descending=True)

        sorted_predicted_scores = torch.gather(predicted_scores, 1, recommended_items)

        return recommended_items, sorted_predicted_scores
