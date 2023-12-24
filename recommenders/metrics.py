import torch
import numpy as np
import bottleneck as bn


def mpr(
    R: torch.Tensor,
    I: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    num_users, num_items = I.shape
    total_interactions = torch.sum(R)

    percentile_ranks = torch.arange(num_items, dtype=dtype, device=device) / num_items

    weighted_ranks = R.gather(1, I) * percentile_ranks.expand(num_users, -1)

    mpr = torch.sum(weighted_ranks) / total_interactions

    return mpr


def NDCG_binary_at_k_batch(X_pred: np.ndarray, heldout_batch: np.ndarray, k: int = 75):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    :param X_pred: The predicted scores.
    :param heldout_batch: The ground truth, it's the matrix of the heldout data (heldout data is the data that we use to test the model).
    :param k: The number of recommendations to consider.
    :return: The Normalized Discounted Cumulative Gain@k for binary relevance.
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(
        axis=1
    )
    IDCG = np.array(
        [(tp[: min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)]
    )
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def NDCG_binary_at_k_batch_torch(
    X_pred: torch.Tensor,
    heldout_batch: torch.Tensor,
    k: int = 100,
    device="cpu",
    dtype=torch.float32,
):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance in PyTorch without using for loops
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance

    :param X_pred: The predicted scores.
    :param heldout_batch: The ground truth, it's the matrix of the heldout data (heldout data is the data that we use to test the model).
    :param k: The number of recommendations to consider.
    :return: The Normalized Discounted Cumulative Gain@k for binary relevance.
    """
    batch_users = X_pred.size(0)

    # Use topk to find the top-k indices
    _, idx_topk = X_pred.topk(k, dim=1)
    tp = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=dtype))

    # Gathering the top-k items
    topk_part = torch.gather(X_pred, 1, idx_topk)

    # Calculating DCG
    DCG = (torch.gather(heldout_batch, 1, idx_topk) * tp).sum(dim=1)

    # Calculating IDCG without a for loop
    count_nonzero = heldout_batch.count_nonzero(dim=1)
    tp_expanded = tp.unsqueeze(0).expand(batch_users, -1)
    IDCG = torch.where(
        count_nonzero.unsqueeze(1) > torch.arange(k, device=device, dtype=dtype),
        tp_expanded,
        torch.zeros_like(tp_expanded, device=device, dtype=dtype),
    ).sum(dim=1)

    return DCG / IDCG


def Recall_at_k_batch_torch(
    X_pred: torch.Tensor, heldout_batch: torch.Tensor, k: int = 100, dtype=torch.float32, device="cpu"
):
    """
    Recall@k in PyTorch.

    :param X_pred: The predicted scores (as a PyTorch Tensor).
    :param heldout_batch: The ground truth, it's the matrix of the heldout data (as a PyTorch Tensor).
    :param k: The number of recommendations to consider.
    :return: Recall@k.
    """
    batch_users = X_pred.size(0)

    # Using topk for getting the indices of the top k predictions
    _, idx = X_pred.topk(k, dim=1)

    # Creating a binary matrix for predictions
    X_pred_binary = torch.zeros_like(X_pred, dtype=torch.bool, device=device)
    X_pred_binary.scatter_(1, idx, True)

    # Creating a binary matrix for ground truth
    X_true_binary = (heldout_batch > 0)

    # Calculating the logical AND and then summing over the rows
    tmp = (X_true_binary & X_pred_binary).sum(dim=1).float()

    # Calculating recall
    recall = tmp / torch.minimum(torch.tensor(k, device=device, dtype=dtype), X_true_binary.sum(dim=1))

    return recall
