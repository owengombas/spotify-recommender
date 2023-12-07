import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import optim


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

    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super(MultiVAE, self).__init__()
        self.device = device
        self.dtype = dtype

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims if decoder_dims else encoder_dims[::-1]

        assert (
            self.decoder_dims[0] == encoder_dims[-1]
        ), "Output dimension of encoder must match input dimension of decoder."
        assert (
            self.decoder_dims[-1] == encoder_dims[0]
        ), "Latent dimension mismatch between encoder and decoder."

        # Modify the last dimension of encoder for mean and variance
        modified_encoder_dims = self.encoder_dims[:-1] + [self.encoder_dims[-1] * 2]
        self.encoder_layers = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in zip(
                    modified_encoder_dims[:-1], modified_encoder_dims[1:]
                )
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in zip(
                    self.decoder_dims[:-1], self.decoder_dims[1:]
                )
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.initialize_weights()
        
        self.mprs = torch.zeros(0, dtype=dtype, device=device)
        self.losses = torch.zeros(0, dtype=dtype, device=device)

        self.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        h = F.normalize(input)
        h = self.dropout(h)

        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i != len(self.encoder_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, : self.encoder_dims[-1]]
                logvar = h[:, self.encoder_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i != len(self.decoder_layers) - 1:
                h = torch.tanh(h)
        return h

    def initialize_weights(self) -> None:
        for layer in self.encoder_layers + self.decoder_layers:
            size = layer.weight.size()
            fan_out, fan_in = size[0], size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        anneal: float = 1.0,
    ) -> torch.Tensor:
        """
        Loss function for MultiVAE.

        Combines Binary Cross Entropy (BCE) and Kullback–Leibler Divergence (KLD)
        to form the Variational Autoencoder loss.

        Parameters:
            recon_x (torch.Tensor): Reconstructed input.
            x (torch.Tensor): Original input.
            mu (torch.Tensor): Mean from the latent space.
            logvar (torch.Tensor): Log variance from the latent space.
            anneal (float): Annealing factor for KLD.

        Returns:
            torch.Tensor: Calculated loss.
        """
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + anneal * KLD

    def mpr(self, I: torch.Tensor, R: torch.Tensor) -> float:
        num_users, num_items = R.shape
        total_interactions = torch.sum(R)

        # Create a tensor for percentile ranks (shape: num_items)
        percentile_ranks = (
            torch.arange(num_items, dtype=self.dtype, device=self.device) / num_items
        )

        # Use I to index into R and rearrange it, then multiply by the percentile ranks
        # Reshape and expand percentile ranks to match the shape of R
        # (shape of expanded percentile ranks: 1 x num_items)
        weighted_ranks = R.gather(1, I) * percentile_ranks.expand(num_users, -1)

        # Sum over all items and users, and normalize
        mpr = torch.sum(weighted_ranks) / total_interactions

        return mpr.item()

    def recommend_items(
        self,
        user_data: torch.Tensor,
        interacted_indices: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> List[int]:
        """
        Generate item recommendations for a user using the MultiVAE model.

        Args:
        model (MultiVAE): The trained MultiVAE model.
        user_data (torch.Tensor): The user-item interaction vector for a single user.
        interacted_indices (Optional[torch.Tensor]): A binary vector indicating items the user has already interacted with. If None, no filtering is applied.
        top_k (int): Number of top recommendations to return.

        Returns:
        List[int]: List of item indices recommended for the user.
        """
        self.eval()  # Ensure the model is in evaluation mode

        # Encode user data to latent space
        mu, logvar = self.encode(user_data)
        z = self.reparameterize(mu, logvar)

        # Decode the latent representation
        reconstructed_user_data = self.decode(z)

        # Convert to probabilities
        proba = torch.softmax(reconstructed_user_data, dim=1)

        # If interacted_indices is provided, mask out already interacted items
        if interacted_indices is not None:
            interaction_mask = torch.zeros(
                proba.size(), dtype=torch.bool, device=self.device
            )
            interaction_mask[
                0, interacted_indices
            ] = True  # Assumes user_data is a single user batch
            proba.masked_fill_(interaction_mask, 0)

        # Get top k items that have the highest probability
        recommended_items = torch.topk(proba, top_k, dim=1)[1].squeeze().tolist()

        return recommended_items

    def train_model(
        self,
        train_loader: DataLoader,
        optimizer: optim,
        num_epochs: int,
        max_anneal=1.0,
        anneal_steps=20000,
        log_interval=1000,
    ):
        self.train()
        
        global_step = 0
        self.mprs = torch.zeros(num_epochs // log_interval, dtype=self.dtype, device=self.device)
        self.losses = torch.zeros(num_epochs // log_interval, dtype=self.dtype, device=self.device)

        for epoch in range(num_epochs):
            for batch in train_loader:
                user_data = batch[0]  # Assuming batch contains user data

                optimizer.zero_grad()

                # Annealing schedule
                anneal = torch.min(
                    torch.tensor([max_anneal, (global_step + 1) / anneal_steps])
                )

                # Forward pass
                reconstructed, mu, logvar = self(user_data)

                # Loss calculation with annealing factor
                loss = self.loss(reconstructed, user_data, mu, logvar, anneal)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                global_step += 1

                # Optional: Print loss and anneal factor
                if global_step % log_interval == 0:
                    self.losses[global_step // log_interval - 1] = loss.item()
                    print(
                        f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}, Anneal: {anneal}"
                    )
