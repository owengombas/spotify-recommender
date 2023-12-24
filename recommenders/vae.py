import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from recommenders.metrics import NDCG_binary_at_k_batch, mpr, NDCG_binary_at_k_batch_torch, Recall_at_k_batch_torch


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    anneal: torch.Tensor = 1.0,
) -> torch.Tensor:
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

    Attributes:
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

    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
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
        summary_writer: SummaryWriter = None,
        validation_dataloader: DataLoader = None,
        ndcg_k: int = 75,
    ) -> Tuple[torch.Tensor, torch.Tensor, "MultiVAE"]:
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
        val_loss_history = torch.zeros(num_epochs, dtype=dtype, device=device)
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
            ndcg_users: List[np.ndarray] = []
            if validation_dataloader:
                self.eval()
                with torch.no_grad():
                    for batch in validation_dataloader:
                        validation_train, validation_test = batch
                        validation_train = validation_train.to(device=device, dtype=dtype)
                        validation_test = validation_test.to(device=device, dtype=dtype)
                        prediction, _, _ = self(validation_train)
                        prediction[validation_train.nonzero(as_tuple=True)] = -np.inf # remove watched items from recommendations
                        ndcg = NDCG_binary_at_k_batch_torch(prediction, validation_test, ndcg_k, device=device, dtype=dtype)
                        ndcg_users.append(ndcg)
                    ndcg_users = torch.concatenate(ndcg_users)
                    val_loss_history[epoch] = torch.mean(ndcg_users)

                    if val_loss_history[epoch] > best_val_loss:
                        best_val_loss = val_loss_history[epoch]
                        best_model_state = self.state_dict()
                        print(f"Best model found at epoch {epoch} with NDCG: {best_val_loss.item():.6f}")

                    if epoch % log_interval == 0:
                        print(
                            f"Epoch: {epoch} \t Loss: {loss_history[epoch].item():.6f} \t NDCG: {val_loss_history[epoch].item():.6f}"
                        )
        
        best_model = MultiVAE(self.p_dims, self.q_dims)
        best_model.load_state_dict(best_model_state)
        best_model.to(device, dtype)
        best_model.eval()

        return loss_history, val_loss_history, best_model

    def evaluate_model(
        self,
        validation_dataloader: DataLoader,
        batch_size: int,
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        summary_writer: SummaryWriter = None,
    ):
        self.to(device, dtype)
        self.eval()
        total_loss = 0.0
        n100_list = []
        r20_list = []
        r50_list = []
        update_count = 0
        N = len(validation_dataloader.dataset)

        with torch.no_grad():
            for batch in validation_dataloader:
                validation_train, validation_test = batch
                validation_train: torch.Tensor = validation_train.to(device=device, dtype=dtype)
                validation_test: torch.Tensor = validation_test.to(device=device, dtype=dtype)

                anneal = anneal_cap
                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)

                recon_batch, mu, logvar = self(validation_train)

                loss = loss_function(recon_batch, validation_train, mu, logvar, anneal)
                total_loss += loss.item()

                recon_batch[validation_train.nonzero(as_tuple=True)] = -np.inf

                n100 = NDCG_binary_at_k_batch_torch(recon_batch, validation_test, 100, device=device, dtype=dtype)
                r20 = Recall_at_k_batch_torch(recon_batch, validation_test, 20, device=device, dtype=dtype)
                r50 = Recall_at_k_batch_torch(recon_batch, validation_test, 50, device=device, dtype=dtype)

                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)

        total_loss /= len(range(0, N, batch_size))
        n100_list = torch.cat(n100_list)
        r20_list = torch.cat(r20_list)
        r50_list = torch.cat(r50_list)

        return total_loss, torch.mean(n100_list), torch.mean(r20_list), torch.mean(r50_list)

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
