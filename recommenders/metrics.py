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


def NDCG_binary_at_k_batch(X_pred: np.ndarray, heldout_batch: np.ndarray, k: int = 100):
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
    nn = heldout_batch.nonzero()
    IDCG = np.array(
        [(tp[: min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)]
    )
    return DCG / (IDCG + 1e-8)


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

    # Return the indices that would sort an array in descending order (max k)
    idx_topk_part: torch.Tensor = torch.topk(X_pred, k, dim=1, sorted=False)[1]

    # Get the top k predictions for each user
    topk_part: torch.Tensor = X_pred.gather(1, idx_topk_part)

    # Return the indices that would sort an array in descending order (max k)
    idx_part: torch.Tensor = torch.argsort(-topk_part, dim=1)

    # Get the top k predictions for each user sorted in descending order
    idx_topk: torch.Tensor = idx_topk_part.gather(1, idx_part)

    # Discounted gain
    tp = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=dtype))

    # Discounted cumulative gain
    DCG: torch.Tensor = (heldout_batch.gather(1, idx_topk) * tp).sum(dim=1, dtype=dtype)

    # Count the number of non-zero relevance scores in each row
    rel_count: torch.Tensor = heldout_batch.sum(dim=1, dtype=dtype)

    # Use broadcasting to create a mask where each row contains the sequence [1, ..., min(n, k)]
    mask: torch.Tensor = torch.arange(1, k + 1, dtype=dtype).unsqueeze(
        0
    ) <= rel_count.unsqueeze(1)

    # Ideal discounted cumulative gain
    IDCG: torch.Tensor = (mask.float() * tp).sum(dim=1, dtype=dtype)

    return DCG / IDCG  # Normalized discounted cumulative gain


def Recall_at_k_batch_torch(
    X_pred: torch.Tensor, heldout_batch: torch.Tensor, k: int = 100
):
    """
    Recall@k in PyTorch. Measures how many of the true items are in the top k recommendations.
    """
    batch_users = X_pred.size(0)

    # Return the indices that would partition the array in descending order (top k)
    idx = torch.topk(X_pred, k, dim=1, sorted=False)[1]

    # Create a binary tensor of the same shape as X_pred
    X_pred_binary = torch.zeros_like(X_pred, dtype=torch.bool)

    # Mark the top k predictions as True
    X_pred_binary.scatter_(1, idx, True)

    # Convert heldout_batch to a binary tensor
    X_true_binary = heldout_batch > 0

    # Calculate the number of true positives
    true_positives = torch.logical_and(X_true_binary, X_pred_binary).sum(dim=1).float()

    # Calculate recall
    recall = true_positives / torch.clamp(X_true_binary.sum(dim=1), max=k).float()

    return recall
