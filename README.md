# SignGraph: Sign-Pattern Graph Diffusion for Model Merging

Official code for the paper *"SignGraph: Sign-Pattern Graph Diffusion for Model Merging"*.

## Overview

SignGraph merges multiple task-specific fine-tuned models into a single multi-task model by:
1. Clustering parameters by their N-bit sign fingerprint across task vectors
2. Building a Hamming-distance graph over clusters
3. Running personalized PageRank diffusion to propagate consensus
4. Applying soft sigmoid suppression controlled by a density parameter

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using SignGraph directly

```python
import torch
from merge_methods import sign_graph

# task_tensors: list of N delta-weight tensors (theta_ft - theta_base)
# Each tensor has the same shape (e.g., a single layer's weights)
task_tensors = [delta_1, delta_2, delta_3, delta_4, delta_5, delta_6]
weights = torch.ones(6)  # equal weights
density = 0.05  # keep ~5% of parameter signal

merged = sign_graph(task_tensors, weights, density)
# merged has the same shape as each input tensor
```

### Running full experiments

```bash
# Edit configs/example.yaml with your adapter paths
python run_merging.py --config configs/example.yaml
```

## Methods Implemented

| Method | Paper | Key Idea |
|--------|-------|----------|
| **SignGraph** (ours) | This paper | Sign-pattern clustering + PPR diffusion |
| TIES | Yadav et al. 2023 | Magnitude prune + majority sign + disjoint merge |
| DARE-TIES | Yu et al. 2024 | Random dropout + rescale + TIES |
| DELLA | Shen et al. 2024 | Rank-based probabilistic pruning + L1 rescale |
| Consensus-TIES | — | Conflict-aware density allocation + TIES |
| Fisher Merging | Matena & Raffel 2022 | Fisher-information-weighted average |
| Task Arithmetic | Ilharco et al. 2023 | Weighted sum of task vectors |
| Simple Average | Wortsman et al. 2022 | Normalized weighted average |
| Model Breadcrumbs | Davari & Belilovsky 2023 | Outlier + small-magnitude pruning |

## SignGraph Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `density` (ρ) | tuned | Suppression threshold (only tunable param) |
| `correlation_threshold` (γ) | -0.7 | Sign flip detection threshold |
| `restart_prob` (α) | 0.3 | PPR restart probability |
| `num_diffusion_steps` (T) | 3 | Diffusion iterations |
| `consensus_weight` (β) | 0.3 | Consensus shift strength |
| `max_hamming` | 2 | Max Hamming distance for edges |
| `steepness` (s) | 10 | Sigmoid steepness |

Only `density` requires tuning; all others use fixed defaults.

## File Structure

```
code/
├── merge_methods.py    # All merging algorithms (standalone, no dependencies beyond PyTorch)
├── run_merging.py      # Full experiment pipeline (load models, merge, evaluate)
├── evaluate.py         # Task-specific evaluation (classification, code, translation)
├── configs/
│   └── example.yaml    # Example configuration
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Evaluation

Results are reported as **retention**: the ratio of the merged model's task performance to the individually fine-tuned model's performance.

```
retention_t = score_merged_t / score_individual_t × 100
```

A method "wins" a slot if it achieves the highest retention across all methods and densities for that model-task pair.

## Citation

```bibtex
@inproceedings{signgraph2025,
  title={SignGraph: Sign-Pattern Graph Diffusion for Model Merging},
  author={Paul Trust},
  year={2026}
}
```
