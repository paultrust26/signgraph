"""
SignGraph: Sign-Pattern Graph Diffusion for Model Merging

Core merging algorithms compared in the paper. Each function takes a list of
task vectors (delta weights), per-adapter weights, and a density parameter,
and returns the merged task vector.

Methods implemented:
  - sign_graph: SignGraph (ours)
  - ties: TIES-Merging (Yadav et al. 2023)
  - dare_ties: DARE-TIES (Yu et al. 2024)
  - della: DELLA (Shen et al. 2024)
  - consensus_ties: Consensus-TIES
  - fisher_merging: Fisher Merging (Matena & Raffel 2022)
  - task_arithmetic: Linear / Task Arithmetic (Ilharco et al. 2023)
  - simple_average: Simple Average (Wortsman et al. 2022)
  - breadcrumbs: Model Breadcrumbs (Davari & Belilovsky 2023)

References & code sources:
  - TIES, DARE-TIES, Task Arithmetic utilities adapted from HuggingFace PEFT:
    https://github.com/huggingface/peft/blob/main/src/peft/utils/merge_utils.py
  - DELLA: https://github.com/flagflag0/DELLA
    Paper: https://arxiv.org/abs/2406.11617
  - TIES-Merging: https://github.com/prateeky2806/ties-merging
    Paper: https://arxiv.org/abs/2306.01708
  - DARE: https://github.com/yule-BUAA/MergeLM
    Paper: https://arxiv.org/abs/2311.03099
  - Fisher Merging: https://github.com/mmatena/model_merging
    Paper: https://arxiv.org/abs/2111.09832
  - Task Arithmetic: https://github.com/mlfoundations/task_vectors
    Paper: https://arxiv.org/abs/2212.04089
  - Model Breadcrumbs: https://github.com/MahdiDavari/Model-Breadcrumbs
    Paper: https://arxiv.org/abs/2312.06795
  - Simple Average / Model Soups: https://github.com/mlfoundations/model-soups
    Paper: https://arxiv.org/abs/2203.05482
"""

from typing import Literal

import torch

from utils import (
    reshape_weight_task_tensors,
    magnitude_outliers_pruning,
    della_magprune_pruning,
    calculate_majority_sign_mask,
    disjoint_merge,
    detect_sign_flips,
    sparsify_task_tensors,
    weighted_quantile_threshold,
    compute_conflict_score,
)


# ---------------------------------------------------------------------------
# Method 1: SignGraph (Ours)
# ---------------------------------------------------------------------------


def sign_graph(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    num_diffusion_steps: int = 3,
    restart_prob: float = 0.3,
    max_hamming: int = 2,
    consensus_weight: float = 0.3,
    correlation_threshold: float = -0.7,
    steepness: float = 10.0,
) -> torch.Tensor:
    """Sign-Pattern Graph Diffusion for Model Merging.

    Clusters parameters by their N-bit sign fingerprint across task vectors,
    builds a Hamming-distance graph over clusters, and runs personalized
    PageRank diffusion to propagate consensus. Density controls soft
    suppression of weak clusters via sigmoid gating.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Suppression threshold in (0, 1]. Higher keeps more parameters.
        num_diffusion_steps: Number of PPR iterations on the cluster graph.
        restart_prob: PPR restart probability (alpha).
        max_hamming: Maximum Hamming distance for graph edges.
        consensus_weight: Strength of the consensus shift (beta).
        correlation_threshold: Threshold for sign-flip detection (gamma).
        steepness: Sigmoid steepness for soft suppression.

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    aligned = detect_sign_flips(task_tensors, correlation_threshold=correlation_threshold)

    N = len(aligned)
    device = aligned[0].device
    orig_shape = aligned[0].shape
    orig_dtype = task_tensors[0].dtype

    w_list = weights.float().tolist()
    n_params = aligned[0].numel()

    if n_params < 4:
        flat_initial = torch.zeros(n_params, device=device)
        for i in range(N):
            flat_initial.add_(aligned[i].flatten().float(), alpha=w_list[i])
        return flat_initial.reshape(orig_shape).to(orig_dtype)

    n_clusters = 2 ** N

    # Stage 1: Weighted average + sign fingerprint clustering (single pass)
    flat_initial = torch.zeros(n_params, device=device)
    cluster_ids = torch.zeros(n_params, device=device, dtype=torch.long)

    for i in range(N):
        flat_i = aligned[i].flatten().float()
        flat_initial.add_(flat_i, alpha=w_list[i])
        digit = (flat_i > 0).long()
        cluster_ids.add_(digit * (2 ** i))
        del flat_i, digit

    # Cluster statistics
    cluster_means = torch.zeros(n_clusters, device=device)
    cluster_counts = torch.zeros(n_clusters, device=device)
    cluster_mag_sums = torch.zeros(n_clusters, device=device)
    cluster_means.scatter_add_(0, cluster_ids, flat_initial)
    cluster_counts.scatter_add_(0, cluster_ids, torch.ones(n_params, device=device))
    cluster_mag_sums.scatter_add_(0, cluster_ids, flat_initial.abs())

    non_empty = cluster_counts > 0
    cluster_means[non_empty] /= cluster_counts[non_empty]
    cluster_magnitude = torch.zeros(n_clusters, device=device)
    cluster_magnitude[non_empty] = cluster_mag_sums[non_empty] / cluster_counts[non_empty]

    # Stage 2: Build Hamming-distance cluster graph
    ids = torch.arange(n_clusters, device=device)
    hamming = torch.zeros(n_clusters, n_clusters, device=device)
    xor_matrix = ids.unsqueeze(1) ^ ids.unsqueeze(0)
    temp = xor_matrix.clone()
    for _ in range(N):
        hamming += (temp & 1).float()
        temp = temp >> 1

    edge_mask = (hamming > 0) & (hamming <= max_hamming) & non_empty.unsqueeze(0) & non_empty.unsqueeze(1)
    transition = torch.zeros_like(hamming)
    transition[edge_mask] = 1.0 / hamming[edge_mask]
    row_sums = transition.sum(dim=1, keepdim=True).clamp(min=1e-8)
    transition = transition / row_sums

    # Stage 3: Personalized PageRank diffusion on cluster means
    current = cluster_means.clone()
    for _ in range(num_diffusion_steps):
        diffused = transition @ current
        current = restart_prob * cluster_means + (1.0 - restart_prob) * diffused

    # Consensus shift
    cluster_shift = current - cluster_means
    param_shift = cluster_shift[cluster_ids]
    result = flat_initial + consensus_weight * param_shift

    # Stage 4: Soft suppression via per-cluster importance
    cluster_importance = cluster_counts * cluster_magnitude
    threshold = weighted_quantile_threshold(cluster_importance, cluster_counts, non_empty, max(0.0, 1.0 - density))
    param_importance = cluster_importance[cluster_ids]
    suppression = torch.sigmoid(steepness * (param_importance - threshold))
    result = result * suppression

    return result.reshape(orig_shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Method 2: TIES-Merging (Yadav et al. 2023)
# ---------------------------------------------------------------------------


def ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    Prunes each task tensor by magnitude, elects the majority sign per
    parameter position, then averages only values matching the elected sign.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Fraction of values to preserve via magnitude pruning, in (0, 1].
        majority_sign_method: Method for sign election ("total" or "frequency").

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    pruned = sparsify_task_tensors(task_tensors, density, "magnitude")
    stacked = torch.stack(pruned, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(stacked, method=majority_sign_method)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    weighted = stacked * weights_r
    return disjoint_merge(weighted, majority_sign_mask)


# ---------------------------------------------------------------------------
# Method 3: DARE-TIES (Yu et al. 2024)
# ---------------------------------------------------------------------------


def dare_ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """DARE-TIES: Random dropout with rescaling followed by TIES sign election.

    Each task tensor is randomly pruned with Bernoulli masking and rescaled to
    preserve the expected value. TIES sign election and disjoint merge follow.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Fraction of values to preserve via random pruning, in (0, 1].
        majority_sign_method: Method for sign election ("total" or "frequency").

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    pruned = sparsify_task_tensors(task_tensors, density, "random", default_rescale=True)
    stacked = torch.stack(pruned, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(stacked, method=majority_sign_method)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    weighted = stacked * weights_r
    return disjoint_merge(weighted, majority_sign_mask)


# ---------------------------------------------------------------------------
# Method 4: DELLA (Shen et al. 2024)
# ---------------------------------------------------------------------------


def della(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
    epsilon: float = 0.15,
) -> torch.Tensor:
    """DELLA: Magnitude-ranked probabilistic pruning with L1 rescaling.

    Each task tensor's elements survive with probability proportional to their
    magnitude rank. Surviving values are L1-rescaled, then TIES sign election
    and disjoint merge are applied.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Fraction of values to preserve, in (0, 1].
        majority_sign_method: Method for sign election ("total" or "frequency").
        epsilon: Controls rank influence on survival probability.

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    sparsified = [della_magprune_pruning(t, density, epsilon=epsilon) for t in task_tensors]
    stacked = torch.stack(sparsified, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(stacked, method=majority_sign_method)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    weighted = stacked * weights_r
    return disjoint_merge(weighted, majority_sign_mask)


# ---------------------------------------------------------------------------
# Method 5: Consensus-TIES
# ---------------------------------------------------------------------------


def consensus_ties(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    """Consensus-TIES: Conflict-aware adaptive density with TIES sign election.

    Allocates the density budget non-uniformly: high-conflict positions (where
    adapters disagree with similar magnitudes) are protected from pruning, while
    low-conflict positions are pruned more aggressively. Total parameter budget
    equals density x n_params.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Total fraction of parameters to retain, in (0, 1].
        majority_sign_method: Method for sign election ("total" or "frequency").

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    aligned = detect_sign_flips(task_tensors, correlation_threshold=-0.7)
    conflict = compute_conflict_score(aligned)

    conflict_boost_alpha = 2.0
    pruned = []
    for tensor in aligned:
        k = int(density * tensor.numel())
        if k == 0:
            pruned.append(torch.zeros_like(tensor))
            continue
        importance = tensor.abs() * (1.0 + conflict_boost_alpha * conflict)
        threshold = torch.kthvalue(importance.flatten(), importance.numel() - k + 1)[0]
        mask = (importance >= threshold).float()
        pruned.append(tensor * mask)

    stacked = torch.stack(pruned, dim=0)
    majority_sign_mask = calculate_majority_sign_mask(stacked, method=majority_sign_method)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    weighted = stacked * weights_r
    return disjoint_merge(weighted, majority_sign_mask)


# ---------------------------------------------------------------------------
# Method 6: Fisher Merging (Matena & Raffel 2022)
# ---------------------------------------------------------------------------


def fisher_merging(
    task_tensors: list[torch.Tensor],
    fisher_weights: list[torch.Tensor],
    weights: torch.Tensor,
    normalize: bool = True,
    minimal_weight: float = 1e-6,
) -> torch.Tensor:
    """Fisher-weighted merge of task vectors.

    Merges as: merged_j = sum_i(F_i,j * w_i * delta_i,j) / sum_i(F_i,j * w_i)
    where F_i,j is the diagonal Fisher information for parameter j of adapter i.

    Fisher weights must be precomputed by running calibration data through each
    adapter and collecting squared gradients (diagonal Fisher approximation).

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        fisher_weights: List of N Fisher information tensors (same shape as deltas).
        weights: Per-adapter scalar weights, shape [N].
        normalize: Whether to L2-normalize Fisher weights across adapters.
        minimal_weight: Floor value added to Fisher to prevent division by zero.

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    N = len(task_tensors)

    if normalize:
        fisher_norms = []
        for i in range(N):
            norm_sq = fisher_weights[i].float().pow(2).sum().item()
            fisher_norms.append(norm_sq ** 0.5 + minimal_weight)
        inv_norms = [1.0 / n for n in fisher_norms]
        norm_sum = sum(inv_norms)
        scale_factors = [inv_n / norm_sum for inv_n in inv_norms]
    else:
        scale_factors = [1.0 / N] * N

    numerator = torch.zeros_like(task_tensors[0], dtype=torch.float32)
    denominator = torch.zeros_like(task_tensors[0], dtype=torch.float32)

    for i in range(N):
        f_i = fisher_weights[i].float() + minimal_weight
        s_i = scale_factors[i] * weights[i].item()
        numerator.add_(s_i * f_i * task_tensors[i].float())
        denominator.add_(s_i * f_i)

    return (numerator / denominator).to(task_tensors[0].dtype)


# ---------------------------------------------------------------------------
# Method 7: Linear / Task Arithmetic (Ilharco et al. 2023)
# ---------------------------------------------------------------------------


def task_arithmetic(task_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Task Arithmetic: weighted sum of task vectors.

    Computes merged = sum_i(w_i * tau_i) without any pruning or sign election.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    stacked = torch.stack(task_tensors, dim=0)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    return (stacked * weights_r).sum(dim=0)


# ---------------------------------------------------------------------------
# Method 8: Simple Average (Wortsman et al. 2022)
# ---------------------------------------------------------------------------


def simple_average(task_tensors: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Simple Average: normalized weighted average of task vectors.

    Computes merged = sum_i(w_i * tau_i) / sum_i(w_i). With uniform weights
    this reduces to the arithmetic mean of the task vectors.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    stacked = torch.stack(task_tensors, dim=0)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    mixed = (stacked * weights_r).sum(dim=0)
    divisor = weights_r.sum(dim=0)
    divisor[divisor.abs() < 1e-8] = 1
    return mixed / divisor


# ---------------------------------------------------------------------------
# Method 9: Model Breadcrumbs (Davari & Belilovsky 2023)
# ---------------------------------------------------------------------------


def breadcrumbs(
    task_tensors: list[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    gamma: float = 0.01,
) -> torch.Tensor:
    """Model Breadcrumbs: outlier and small-magnitude pruning without sign consensus.

    Removes both the largest (gamma fraction) and smallest weights from each
    task tensor, then takes a weighted sum of the surviving deltas. No sign
    election or normalization is applied.

    Args:
        task_tensors: List of N adapter delta tensors, all of the same shape.
        weights: Per-adapter scalar weights, shape [N].
        density: Fraction of values to preserve after pruning, in (0, 1].
        gamma: Fraction of largest-magnitude weights to remove as outliers.

    Returns:
        Merged tensor with the same shape and dtype as the input task tensors.
    """
    sparsified = [magnitude_outliers_pruning(t, density, gamma=gamma) for t in task_tensors]
    stacked = torch.stack(sparsified, dim=0)
    weights_r = reshape_weight_task_tensors(stacked, weights)
    return (stacked * weights_r).sum(dim=0)
