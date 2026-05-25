"""
SignGraph: Reproduce paper experiments.

Usage:
    python run_merging.py --config configs/example.yaml

This script:
  1. Loads a pretrained base model and N LoRA adapters (or full fine-tuned models)
  2. Computes task vectors (delta weights)
  3. Merges using all methods at specified densities
  4. Evaluates each merged model on held-out tasks
  5. Computes retention (merged score / individual score) and saves results
"""

import argparse
import json
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_methods import (
    sign_graph,
    ties,
    dare_ties,
    della,
    consensus_ties,
    task_arithmetic,
    simple_average,
    breadcrumbs,
)
from evaluate import evaluate_merged_model


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_task_vectors(
    base_model: torch.nn.Module,
    adapter_configs: List[Dict[str, Any]],
    device: str = "cpu",
) -> tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """
    Compute task vectors (delta weights) for each adapter.

    Supports two modes:
      - LoRA adapters: loads via PEFT, computes BA*scaling as the delta
      - Full fine-tuned models: computes theta_ft - theta_base

    Args:
        base_model: Pretrained base model.
        adapter_configs: List of dicts with 'name' and 'path' keys.
        device: Device for computation.

    Returns:
        (task_vectors, param_names): list of per-adapter delta dicts, and
        the ordered list of parameter names that were merged.
    """
    base_state = {n: p.data.cpu() for n, p in base_model.named_parameters()}
    task_vectors = []
    param_names = None

    for cfg in adapter_configs:
        adapter_path = cfg["path"]
        adapter_name = cfg["name"]
        mode = cfg.get("mode", "lora")

        if mode == "lora":
            from peft import PeftModel, PeftConfig

            peft_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name="default")
            delta = OrderedDict()
            for key, module in peft_model.base_model.model.named_modules():
                if not hasattr(module, "get_delta_weight"):
                    continue
                has_lora = hasattr(module, "lora_A") and "default" in module.lora_A
                if not has_lora:
                    continue
                with torch.no_grad():
                    d = module.get_delta_weight("default").cpu()
                delta[key] = d

            # Clean up
            del peft_model
            gc.collect()

        elif mode == "full":
            ft_model = AutoModelForCausalLM.from_pretrained(
                adapter_path, torch_dtype=torch.bfloat16, device_map="cpu"
            )
            delta = OrderedDict()
            exclude = cfg.get("exclude_patterns", ["embed", "lm_head"])
            for name, param in ft_model.named_parameters():
                if any(pat in name.lower() for pat in exclude):
                    continue
                if name in base_state:
                    delta[name] = (param.data.cpu() - base_state[name]).to(torch.bfloat16)

            del ft_model
            gc.collect()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if param_names is None:
            param_names = list(delta.keys())
        else:
            param_names = [n for n in param_names if n in delta]

        task_vectors.append(delta)
        print(f"  Computed task vector for '{adapter_name}': {len(delta)} layers")

    return task_vectors, param_names


def merge_task_vectors(
    task_vectors: List[Dict[str, torch.Tensor]],
    param_names: List[str],
    method: str,
    weights: List[float],
    density: float,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Merge task vectors layer by layer using the specified method.

    Args:
        task_vectors: Per-adapter delta dicts.
        param_names: Ordered parameter names to merge.
        method: One of the supported merging methods.
        weights: Per-adapter weights.
        density: Density parameter for pruning-based methods.
        device: Device for computation.

    Returns:
        Dict of merged delta tensors.
    """
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    merged = OrderedDict()

    density_independent = {"task_arithmetic", "linear", "simple_average"}

    for idx, name in enumerate(param_names):
        tensors = [tv[name].float().to(device) for tv in task_vectors]

        if method == "sign_graph":
            merged[name] = sign_graph(tensors, w, density)
        elif method == "ties":
            merged[name] = ties(tensors, w, density)
        elif method == "dare_ties":
            merged[name] = dare_ties(tensors, w, density)
        elif method == "della":
            merged[name] = della(tensors, w, density)
        elif method == "consensus_ties":
            merged[name] = consensus_ties(tensors, w, density)
        elif method in ("task_arithmetic", "linear"):
            merged[name] = task_arithmetic(tensors, w)
        elif method == "simple_average":
            merged[name] = simple_average(tensors, w)
        elif method == "breadcrumbs":
            merged[name] = breadcrumbs(tensors, w, density)
        else:
            raise ValueError(f"Unknown method: {method}")

        merged[name] = merged[name].to(torch.bfloat16).cpu()

        if (idx + 1) % 50 == 0:
            print(f"    Merged {idx + 1}/{len(param_names)} layers")

    return merged


def apply_merged_delta(
    base_model: torch.nn.Module,
    merged_delta: Dict[str, torch.Tensor],
    scaling_coefficient: float = 1.0,
) -> torch.nn.Module:
    """Add merged task vector to base model in-place."""
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in merged_delta:
                param.data.add_(
                    merged_delta[name].to(param.device, param.dtype),
                    alpha=scaling_coefficient,
                )
    return base_model


def run_experiment(config: Dict[str, Any]):
    """Run the full merging experiment."""
    base_model_name = config["base_model_name"]
    adapter_configs = config["adapters"]
    methods = config.get("methods", ["sign_graph", "ties", "dare_ties", "della",
                                      "consensus_ties", "task_arithmetic",
                                      "simple_average", "breadcrumbs"])
    densities = config.get("densities", [0.01, 0.05, 0.50, 0.99])
    scaling_coefficient = config.get("scaling_coefficient", 1.0)
    adapter_weights = config.get("adapter_weights", [1.0] * len(adapter_configs))
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Base model: {base_model_name}")
    print(f"Adapters: {[a['name'] for a in adapter_configs]}")
    print(f"Methods: {methods}")
    print(f"Densities: {densities}")
    print(f"Device: {device}")

    # Load base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Compute task vectors
    print("\nComputing task vectors...")
    task_vectors, param_names = compute_task_vectors(base_model, adapter_configs)
    print(f"  Total mergeable parameters: {len(param_names)}")

    # Free base model to save memory during merging
    del base_model
    gc.collect()

    # Run each method × density
    density_independent = {"task_arithmetic", "linear", "simple_average"}
    results = {"config": config, "results": []}

    for method in methods:
        for density in densities:
            if method in density_independent and density != densities[0]:
                continue

            print(f"\n{'='*60}")
            print(f"Method: {method}, Density: {density}")
            print(f"{'='*60}")

            try:
                merged_delta = merge_task_vectors(
                    task_vectors, param_names, method, adapter_weights, density, device
                )

                # Reload base model and apply delta
                print("  Applying merged delta to base model...")
                eval_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, torch_dtype=torch.float16, device_map="auto"
                )
                apply_merged_delta(eval_model, merged_delta, scaling_coefficient)
                del merged_delta
                gc.collect()

                # Evaluate
                print("  Evaluating...")
                eval_config = config.get("evaluation", {})
                eval_results = evaluate_merged_model(
                    eval_model, tokenizer, eval_config, device
                )

                result_entry = {
                    "method": method,
                    "density": density,
                    "metrics": eval_results,
                }
                results["results"].append(result_entry)

                print(f"  Results: {json.dumps(eval_results, indent=2, default=str)}")

                del eval_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                results["results"].append({
                    "method": method,
                    "density": density,
                    "error": str(e),
                })

    # Compute retention scores
    results = compute_retention(results, config)

    # Save results
    results_path = output_dir / "merging_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


def compute_retention(results: Dict, config: Dict) -> Dict:
    """
    Compute retention: merged_score / individual_score * 100.

    Requires individual adapter scores to be provided in config or
    computed separately.
    """
    individual_scores = config.get("individual_scores", {})
    if not individual_scores:
        print("\nNote: No individual_scores in config; skipping retention computation.")
        return results

    for entry in results.get("results", []):
        if "error" in entry:
            continue
        metrics = entry.get("metrics", {})
        retention = {}
        for task, score in metrics.items():
            if task in individual_scores and individual_scores[task] > 0:
                retention[task] = score / individual_scores[task] * 100
        entry["retention"] = retention
        if retention:
            entry["avg_retention"] = sum(retention.values()) / len(retention)

    return results


def main():
    parser = argparse.ArgumentParser(description="SignGraph: Model Merging Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
