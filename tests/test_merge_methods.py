"""Smoke tests for the merging methods in merge_methods.py.

Run from the repository root with:
    python -m pytest tests/
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import merge_methods as mm  # noqa: E402


N_TASKS = 3
SHAPE = (64, 32)


@pytest.fixture
def task_tensors():
    torch.manual_seed(0)
    return [torch.randn(SHAPE) * 0.01 for _ in range(N_TASKS)]


@pytest.fixture
def weights():
    return torch.ones(N_TASKS)


def _run(method, task_tensors, weights):
    if method in {"fisher_merging"}:
        fisher = [torch.rand(SHAPE) for _ in range(N_TASKS)]
        return mm.fisher_merging(task_tensors, fisher, weights)
    if method in {"task_arithmetic", "simple_average", "breadcrumbs", "sign_graph"}:
        return getattr(mm, method)(task_tensors, weights, *([0.05] if method in {"sign_graph", "breadcrumbs"} else []))
    return getattr(mm, method)(task_tensors, weights, 0.05)


METHODS = [
    "sign_graph",
    "ties",
    "dare_ties",
    "della",
    "consensus_ties",
    "fisher_merging",
    "task_arithmetic",
    "simple_average",
    "breadcrumbs",
]


@pytest.mark.parametrize("method", METHODS)
def test_method_returns_finite_tensor_of_input_shape(method, task_tensors, weights):
    out = _run(method, task_tensors, weights)
    assert out.shape == SHAPE
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip_graph": True},
        {"hard_suppression": True},
        {"magnitude_weighted_edges": True},
    ],
)
def test_sign_graph_ablation_flags_run_and_change_output(kwargs, task_tensors, weights):
    default = mm.sign_graph(task_tensors, weights, 0.05)
    ablated = mm.sign_graph(task_tensors, weights, 0.05, **kwargs)
    assert ablated.shape == SHAPE
    assert torch.isfinite(ablated).all()
    assert not torch.allclose(default, ablated), f"{kwargs} produced the same output as the default"


def test_sign_graph_small_tensor_path(weights):
    tiny = [torch.randn(3) for _ in range(N_TASKS)]
    out = mm.sign_graph(tiny, weights, 0.05)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()
