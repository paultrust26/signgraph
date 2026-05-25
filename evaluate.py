"""
Evaluation utilities for merged models.

Supports:
  - Classification tasks (GLUE: CoLA, SST-2, MNLI, QNLI, QQP, RTE)
  - Safety classification (WildGuardMix)
  - Code generation (test pass rate)
  - Multilingual translation (ROUGE-1)
"""

import re
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def evaluate_merged_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_config: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate a merged model on configured tasks.

    Args:
        model: The merged model to evaluate.
        tokenizer: Tokenizer for the model.
        eval_config: Evaluation configuration with dataset paths and task info.
        device: Device for inference.

    Returns:
        Dict mapping task_name -> score (F1 for classification, pass_rate for code,
        ROUGE-1 for translation).
    """
    tasks = eval_config.get("tasks", [])
    if not tasks:
        print("  No evaluation tasks configured.")
        return {}

    results = {}
    for task_cfg in tasks:
        task_name = task_cfg["name"]
        task_type = task_cfg["type"]
        dataset_path = task_cfg.get("dataset_path")
        num_samples = task_cfg.get("num_samples", 100)

        print(f"    Evaluating task: {task_name} ({task_type})")

        try:
            if task_type == "classification":
                score = evaluate_classification(
                    model, tokenizer, task_cfg, device, num_samples
                )
            elif task_type == "code_generation":
                score = evaluate_code_generation(
                    model, tokenizer, task_cfg, device, num_samples
                )
            elif task_type == "translation":
                score = evaluate_translation(
                    model, tokenizer, task_cfg, device, num_samples
                )
            else:
                print(f"      Unknown task type: {task_type}, skipping")
                continue

            results[task_name] = score
            print(f"      Score: {score:.4f}")

        except Exception as e:
            print(f"      Error: {e}")
            results[task_name] = 0.0

    return results


def evaluate_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task_cfg: Dict[str, Any],
    device: str,
    num_samples: int,
) -> float:
    """
    Evaluate on a classification task. Returns macro F1.

    Expects task_cfg to contain:
      - dataset_path: path to HuggingFace dataset or local file
      - prompt_template: template with {input} placeholder
      - labels: list of valid label strings
    """
    from datasets import load_dataset

    dataset_path = task_cfg["dataset_path"]
    split = task_cfg.get("split", "test")
    prompt_col = task_cfg.get("prompt_column", "prompt")
    label_col = task_cfg.get("label_column", "label")
    labels = task_cfg.get("labels", [])
    max_new_tokens = task_cfg.get("max_new_tokens", 16)

    ds = load_dataset(dataset_path, split=split)
    if len(ds) > num_samples:
        ds = ds.shuffle(seed=42).select(range(num_samples))

    predictions = []
    references = []

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for item in ds:
        prompt = item[prompt_col]
        reference = str(item[label_col]).strip().lower()

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_label(generated, labels)

        predictions.append(predicted)
        references.append(reference)

    return compute_f1(predictions, references, labels)


def evaluate_code_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task_cfg: Dict[str, Any],
    device: str,
    num_samples: int,
) -> float:
    """Evaluate code generation via test pass rate."""
    import subprocess
    import sys
    from datasets import load_dataset

    dataset_path = task_cfg["dataset_path"]
    split = task_cfg.get("split", "test")
    prompt_col = task_cfg.get("prompt_column", "prompt")
    tests_col = task_cfg.get("tests_column", "test_list")
    max_new_tokens = task_cfg.get("max_new_tokens", 512)

    ds = load_dataset(dataset_path, split=split)
    if len(ds) > num_samples:
        ds = ds.shuffle(seed=42).select(range(num_samples))

    total_passed = 0
    total_tests = 0

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for item in ds:
        prompt = item[prompt_col]
        test_cases = item.get(tests_col, [])
        if isinstance(test_cases, str):
            import json as _json
            try:
                test_cases = _json.loads(test_cases)
            except Exception:
                test_cases = []

        if not test_cases:
            continue

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(generated)

        for test in test_cases:
            test = test.strip()
            if not test:
                continue
            total_tests += 1
            full_code = code + "\n\n" + test
            try:
                result = subprocess.run(
                    [sys.executable, "-c", full_code],
                    capture_output=True, timeout=10, text=True,
                )
                if result.returncode == 0:
                    total_passed += 1
            except (subprocess.TimeoutExpired, Exception):
                pass

    return total_passed / total_tests if total_tests > 0 else 0.0


def evaluate_translation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task_cfg: Dict[str, Any],
    device: str,
    num_samples: int,
) -> float:
    """Evaluate translation via ROUGE-1 F1."""
    from datasets import load_dataset

    dataset_path = task_cfg["dataset_path"]
    split = task_cfg.get("split", "test")
    prompt_col = task_cfg.get("prompt_column", "prompt")
    ref_col = task_cfg.get("reference_column", "reference")
    max_new_tokens = task_cfg.get("max_new_tokens", 256)

    ds = load_dataset(dataset_path, split=split)
    if len(ds) > num_samples:
        ds = ds.shuffle(seed=42).select(range(num_samples))

    rouge_scores = []

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for item in ds:
        prompt = item[prompt_col]
        reference = item[ref_col]

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        score = compute_rouge1(generated, reference)
        rouge_scores.append(score)

    return float(np.mean(rouge_scores)) if rouge_scores else 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_label(text: str, valid_labels: List[str]) -> Optional[str]:
    """Extract a valid label from generated text."""
    text_lower = text.strip().lower()
    for label in valid_labels:
        if label.lower() in text_lower:
            return label.lower()
    first_word = text_lower.split()[0] if text_lower.split() else ""
    return first_word


def extract_code(text: str) -> str:
    """Extract Python code from generated text."""
    if "```python" in text:
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in text:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


def compute_f1(
    predictions: List[Optional[str]],
    references: List[str],
    labels: List[str],
) -> float:
    """Compute macro F1 score."""
    if not predictions:
        return 0.0

    label_set = set(l.lower() for l in labels) if labels else set(references)

    class_metrics = []
    for label in label_set:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_metrics.append(f1)

    return float(np.mean(class_metrics)) if class_metrics else 0.0


def compute_rouge1(generated: str, reference: str) -> float:
    """Compute ROUGE-1 F1 between generated and reference text."""
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())

    if not ref_tokens:
        return 0.0
    if not gen_tokens:
        return 0.0

    overlap = gen_tokens & ref_tokens
    precision = len(overlap) / len(gen_tokens)
    recall = len(overlap) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
