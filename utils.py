"""
Utility functions for model merging: pruning, sign operations, and helpers.

Code adapted from:
  - HuggingFace PEFT merge utilities:
    https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py
  - DELLA magnitude-ranked pruning:
    https://github.com/flagflag0/DELLA
  - Model Breadcrumbs outlier pruning:
    https://github.com/MahdiDavari/Model-Breadcrumbs
"""

import warnings
from typing import Literal

import torch


def reshape_weight_task_tensors(task_tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reshape weights to broadcast over stacked task tensors.

    Args:
        task_tensors: Stacked task tensors with shape [N, ...].
        weights: Per-adapter weights with shape [N].

    Returns:
        Weights reshaped to [N, 1, 1, ...] for broadcasting.
    """
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    return weights.view(new_shape)


def magnitude_based_pruning(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Retain the top-k values by magnitude, zeroing the rest.

    Args:
        tensor: The tensor to prune.
        density: Fraction of values to preserve, in (0, 1].

    Returns:
        Pruned tensor with the same shape, where only top-k values remain.
    """
    mask = torch.zeros_like(tensor).reshape(-1)
    k = int(density * tensor.numel())
    top_k = torch.topk(tensor.abs().reshape(-1), k=k, largest=True)
    mask[top_k[1]] = 1
    return tensor * mask.reshape(tensor.shape)


def random_pruning(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """Randomly drop values via Bernoulli masking and optionally rescale.

    Args:
        tensor: The tensor to prune.
        density: Probability of keeping each value, in (0, 1].
        rescale: Whether to divide by density to preserve expected value.

    Returns:
        Pruned tensor with the same shape.
    """
    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    pruned_tensor = tensor * mask
    if rescale:
        torch.div(input=pruned_tensor, other=density)
    return pruned_tensor


def magnitude_outliers_pruning(tensor: torch.Tensor, density: float, gamma: float = 0.01) -> torch.Tensor:
    """Model Breadcrumbs pruning: remove both largest and smallest weights.

    Removes the top gamma fraction of largest-magnitude weights (outliers)
    and the smallest weights until only the density fraction remains.

    Args:
        tensor: The tensor to prune.
        density: Fraction of values to preserve, in (0, 1].
        gamma: Fraction of largest-magnitude weights to remove as outliers.

    Returns:
        Pruned tensor with the same shape.
    """
    if density >= 1:
        return tensor

    num_elems = tensor.numel()
    target_n = int(density * num_elems)
    n_top = int(gamma * num_elems)
    n_bot = num_elems - target_n - n_top

    if n_bot < 0:
        n_top += n_bot
        n_bot = 0

    flat = tensor.flatten()
    abs_flat = flat.abs()
    sorted_idx = abs_flat.argsort()

    mask = torch.ones_like(flat)
    if n_bot > 0:
        mask[sorted_idx[:n_bot]] = 0
    if n_top > 0:
        mask[sorted_idx[-n_top:]] = 0

    return (flat * mask).reshape(tensor.shape)


def della_magprune_pruning(tensor: torch.Tensor, density: float, epsilon: float = 0.15) -> torch.Tensor:
    """DELLA: magnitude-ranked probabilistic pruning with L1 norm rescaling.

    Each element's survival probability is proportional to its magnitude rank.
    After masking, the result is rescaled to match the original tensor's L1 norm.

    Args:
        tensor: The tensor to prune.
        density: Fraction of values to preserve, in (0, 1).
        epsilon: Controls rank influence on survival probability.

    Returns:
        Pruned and L1-rescaled tensor with the same shape.
    """
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)

    orig_shape = tensor.shape
    t = tensor
    if len(t.shape) < 2:
        t = t.unsqueeze(0)

    magnitudes = t.abs()
    sorted_indices = torch.argsort(magnitudes, dim=1, descending=False)
    ranks = sorted_indices.argsort(dim=1).float() + 1

    min_ranks = ranks.min(dim=1, keepdim=True).values
    max_ranks = ranks.max(dim=1, keepdim=True).values
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks)).clamp(0, 1)
    probs = ((density - epsilon) + rank_norm * 2 * epsilon).clamp(0, 1)
    mask = torch.bernoulli(probs)

    masked = t.float() * mask
    before_l1 = t.float().abs().sum()
    after_l1 = masked.abs().sum()
    if before_l1 > 1e-7 and after_l1 > 1e-7:
        masked = masked * (before_l1 / after_l1)

    return masked.to(tensor.dtype).reshape(orig_shape)


def calculate_majority_sign_mask(
    tensor: torch.Tensor, method: Literal["total", "frequency"] = "total"
) -> torch.Tensor:
    """Compute the majority sign mask across stacked task tensors.

    For each parameter position, determines which adapters agree with the
    majority sign direction.

    Args:
        tensor: Stacked task tensors with shape [N, ...].
        method: "total" uses magnitude-weighted sign sum, "frequency" uses
            unweighted sign count.

    Returns:
        Boolean mask of shape [N, ...] where True indicates agreement with
        the majority sign at that position.
    """
    sign = tensor.sign()
    if method == "total":
        sign_magnitude = tensor.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = sign.sum(dim=0)
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return sign == majority_sign


def disjoint_merge(task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    """Average only values that match the elected majority sign.

    Args:
        task_tensors: Stacked weighted task tensors with shape [N, ...].
        majority_sign_mask: Boolean mask of shape [N, ...] from sign election.

    Returns:
        Merged tensor with shape [...], averaged over agreeing adapters.
    """
    mixed_task_tensors = (task_tensors * majority_sign_mask).sum(dim=0)
    num_params_preserved = majority_sign_mask.sum(dim=0)
    return mixed_task_tensors / torch.clamp(num_params_preserved, min=1.0)


def detect_sign_flips(
    task_tensors: list[torch.Tensor],
    correlation_threshold: float = -0.7,
) -> list[torch.Tensor]:
    """Detect and correct global sign flips between adapters.

    If an adapter's task vector is strongly negatively correlated with the
    reference (first adapter), it is flipped to prevent catastrophic
    cancellation during merging.

    Args:
        task_tensors: List of N adapter delta tensors to align.
        correlation_threshold: If correlation < threshold, flip the adapter.

    Returns:
        List of sign-aligned task tensors (same shapes as input).
    """
    if len(task_tensors) <= 1:
        return task_tensors

    reference = task_tensors[0].flatten()
    aligned_tensors = [task_tensors[0]]

    for i, tensor in enumerate(task_tensors[1:], 1):
        flat = tensor.flatten()
        ref_centered = reference - reference.mean()
        flat_centered = flat - flat.mean()
        correlation = (ref_centered * flat_centered).sum() / (
            torch.norm(ref_centered) * torch.norm(flat_centered) + 1e-8
        )
        if correlation < correlation_threshold:
            warnings.warn(f"Detected sign flip in adapter {i}: correlation = {correlation:.3f}. Flipping signs.")
            aligned_tensors.append(-tensor)
        else:
            aligned_tensors.append(tensor)

    return aligned_tensors


def sparsify_task_tensors(
    task_tensors: list[torch.Tensor],
    density: float,
    default_method: str,
    default_rescale: bool = False,
) -> list[torch.Tensor]:
    """Apply pruning independently to each task tensor.

    Args:
        task_tensors: List of N adapter delta tensors.
        density: Fraction of values to preserve, in (0, 1].
        default_method: Pruning method ("magnitude" or "random").
        default_rescale: Whether to rescale after random pruning.

    Returns:
        List of pruned task tensors (same shapes as input).
    """
    if density >= 1.0:
        return task_tensors
    result = []
    for t in task_tensors:
        if default_method == "magnitude":
            result.append(magnitude_based_pruning(t, density))
        elif default_method == "random":
            result.append(random_pruning(t, density, rescale=default_rescale))
        else:
            result.append(magnitude_based_pruning(t, density))
    return result


def weighted_quantile_threshold(
    values: torch.Tensor,
    counts: torch.Tensor,
    mask: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """Compute the q-th quantile of values weighted by counts.

    Args:
        values: Per-cluster values (e.g., importance scores).
        counts: Per-cluster element counts used as weights.
        mask: Boolean mask indicating which clusters are non-empty.
        q: Quantile in [0, 1] (e.g., 0.95 for the 95th percentile).

    Returns:
        Scalar threshold tensor at the q-th weighted quantile.
    """
    active_vals = values[mask]
    active_counts = counts[mask]

    if active_vals.numel() == 0:
        return values.new_tensor(0.0)

    sorted_idx = active_vals.argsort()
    sorted_vals = active_vals[sorted_idx]
    sorted_counts = active_counts[sorted_idx]

    cum_weights = sorted_counts.cumsum(dim=0)
    total = cum_weights[-1]
    target = q * total

    above = (cum_weights >= target).nonzero(as_tuple=False)
    if above.numel() == 0:
        return sorted_vals[-1]
    return sorted_vals[above[0, 0]]


def compute_conflict_score(task_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Compute per-position conflict score across adapters.

    Conflict is high when adapters have similar magnitudes but opposite signs.
    Conflict is low when adapters agree or when one clearly dominates.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.

    Returns:
        Conflict score tensor in [0, 1] with the same shape as each input tensor.
    """
    stacked = torch.stack(task_tensors, dim=0)
    n = stacked.shape[0]
    signs = torch.sign(stacked)
    sign_agreement = signs.sum(dim=0).abs().float() / n
    magnitudes = stacked.abs()
    max_mag = magnitudes.max(dim=0).values
    sum_mag = magnitudes.sum(dim=0)
    balance = 1.0 - (max_mag / (sum_mag + 1e-8))
    conflict = balance * (1.0 - sign_agreement)
    return conflict
