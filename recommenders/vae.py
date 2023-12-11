import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch import optim
import bottleneck as bn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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


def NDCG_binary_at_k_batch(X_pred: np.ndarray, heldout_batch: np.ndarray, k: int = 100):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (
        heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp
    ).sum(axis=1)
    IDCG = np.array([(tp[: min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred: np.ndarray, heldout_batch: np.ndarray, k: int = 100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = heldout_batch > 0
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814

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

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
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

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.dims[:-1], self.dims[1:])
            ]
        )
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


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
    ):
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
        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
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

            if epoch % log_interval == 0:
                print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_dataloader)}")
                summary_writer.add_scalars(
                    "data/loss", {"train": epoch_loss / log_interval}, epoch
                )

    def evaluate_model(
        self,
        training_set: torch.Tensor,
        test_set: torch.Tensor,
        batch_size: int,
        anneal_cap: float = 0.2,
        total_anneal_steps: int = 200000,
        log_interval=10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        summary_writer: SummaryWriter = None,
    ):
        self.eval()
        total_loss = 0.0
        e_idxlist = list(range(training_set.shape[0]))
        n100_list = []
        r20_list = []
        r50_list = []
        update_count = 0
        N = training_set.shape[0]

        with torch.no_grad():
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                data = training_set[e_idxlist[start_idx:end_idx]]
                heldout_data = test_set[e_idxlist[start_idx:end_idx]].cpu().numpy()

                anneal = anneal_cap
                if total_anneal_steps > 0:
                    anneal = min(
                        anneal_cap, 1.0 * update_count / total_anneal_steps
                    )

                recon_batch, mu, logvar = self(data)

                loss = loss_function(recon_batch, data, mu, logvar, anneal)
                total_loss += loss.item()

                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.cpu().numpy().nonzero()] = -np.inf

                n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
                r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)
        
        total_loss /= len(range(0, N, batch_size))
        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

        return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)
