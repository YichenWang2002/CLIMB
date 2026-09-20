"""DUCL v2 scoring: semantic Difficulty, target-mix Utility, rank DUE."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from .common import load_jsonl, prompt_text, record_id
from .scoring import ACTION_TAGS, _structure, rank01

BASE_DEFAULT = "models/llama32-1b"
EMBED_DEFAULT = "models/all-MiniLM-L12-v2"
CONDITION_TAGS = {"IsAtLocation", "IsItemAt", "IsCarrying", "SignalReady", "WaitReady"}
SEMANTIC_TAGS = ACTION_TAGS | CONDITION_TAGS
SEMANTIC_ATTRS = {
    "robot", "robot1", "robot2", "giver", "receiver", "item", "location",
    "from", "to", "station", "edge_from", "edge_to",
}
OPEN_TAG_RE = re.compile(r"<([A-Za-z_][\w.-]*)([^<>]*)>")
ATTR_RE = re.compile(r"\b([A-Za-z_][\w.-]*)\s*=\s*([\"'])(.*?)\2")
TARGET_TIER_PROFILE = {"T1": 0.20, "T2": 0.40, "T3": 0.40}
TARGET_FAULT_PROFILE = {"0": 0.40, "1": 0.40, "2": 0.20}


def _xml_complexity(row: dict) -> float:
    """Parameter-free executable-tree complexity from output XML only."""
    tags = re.findall(r"<([A-Za-z_][\w.-]*)\b", row["output"])
    semantic_actions = sum(tag in ACTION_TAGS for tag in tags)
    conditions = sum(tag in CONDITION_TAGS for tag in tags)
    recovery = sum(tag in {"Recharge", "ClearPath"} for tag in tags)
    return float(semantic_actions + conditions + recovery + tags.count("Fallback")
                 + tags.count("Parallel") + tags.count("BehaviorTree"))


def semantic_spans(xml: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in OPEN_TAG_RE.finditer(xml):
        tag = match.group(1)
        if tag not in SEMANTIC_TAGS:
            continue
        spans.append((match.start(1), match.end(1)))
        attrs = match.group(2)
        for attr in ATTR_RE.finditer(attrs):
            if attr.group(1) in SEMANTIC_ATTRS:
                value_start = match.start(2) + attr.start(3)
                spans.append((value_start, match.start(2) + attr.end(3)))
    return spans


def semantic_target_ids(text: str, tokenizer) -> tuple[list[int], list[bool]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    spans = semantic_spans(text)
    if offsets is None:
        return ids, [True] * len(ids)
    mask = []
    for start, end in offsets:
        mask.append(any(start < b and end > a for a, b in spans))
    return ids, mask


def _load_lm_bf16(base: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _semantic_sequence(row: dict, tokenizer, max_len: int):
    prompt_ids = tokenizer(prompt_text(row, tokenizer), add_special_tokens=False)["input_ids"]
    output_text = row["output"] + tokenizer.eos_token
    output_ids, semantic_mask = semantic_target_ids(output_text, tokenizer)
    keep = max_len - len(prompt_ids)
    if keep <= 0:
        return prompt_ids[:max_len], [], []
    return prompt_ids + output_ids[:keep], output_ids[:keep], semantic_mask[:keep]


@torch.no_grad()
def score_difficulty(rows: list[dict], model, tokenizer, batch_size: int, max_len: int):
    device = next(model.parameters()).device
    values, semantic_counts, output_counts = [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prepared = [_semantic_sequence(row, tokenizer, max_len) for row in chunk]
        max_width = max(len(x[0]) for x in prepared)
        ids = torch.full((len(chunk), max_width), tokenizer.pad_token_id,
                         dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (sequence, output_ids, semantic_mask) in enumerate(prepared):
            width = len(sequence)
            ids[i, :width] = torch.tensor(sequence, device=device)
            attention[i, :width] = 1
            prompt_len = width - len(output_ids)
            for j, (token_id, use) in enumerate(zip(output_ids, semantic_mask)):
                if use:
                    labels[i, prompt_len + j] = token_id
        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
        target = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(target.shape)
        valid = target != -100
        for i in range(len(chunk)):
            count = int(valid[i].sum().item())
            semantic_counts.append(count)
            output_counts.append(len(prepared[i][1]))
            values.append(float((losses[i] * valid[i]).sum().div(max(count, 1)).cpu()))
        done = min(start + batch_size, len(rows))
        if start == 0 or done == len(rows) or done % (batch_size * 25) == 0:
            print(f"semantic difficulty {done}/{len(rows)}", flush=True)
    if not all(semantic_counts):
        raise RuntimeError("some rows have no semantic target tokens")
    semantic_nll = np.asarray(values, dtype=np.float64)
    xml_complexity = np.asarray([_xml_complexity(row) for row in rows], dtype=np.float64)
    difficulty = np.sqrt(rank01(semantic_nll) * rank01(xml_complexity))
    return difficulty, {
        "difficulty_definition": "bf16_frozen_base_semantic_nll_geometric_rank_xml_complexity",
        "difficulty_semantic_nll_mean": float(np.mean(semantic_nll)),
        "difficulty_semantic_nll_std": float(np.std(semantic_nll)),
        "difficulty_xml_complexity_mean": float(np.mean(xml_complexity)),
        "difficulty_xml_complexity_std": float(np.std(xml_complexity)),
        "difficulty_rank_fusion": "geometric_mean_equal_views_no_manual_weight",
        "difficulty_semantic_token_mean": float(np.mean(semantic_counts)),
        "difficulty_semantic_token_min": int(np.min(semantic_counts)),
        "difficulty_semantic_token_max": int(np.max(semantic_counts)),
        "difficulty_output_token_mean": float(np.mean(output_counts)),
        "difficulty_semantic_fraction": float(np.sum(semantic_counts) / max(np.sum(output_counts), 1)),
    }


def _target_weights(val: list[dict]) -> tuple[np.ndarray, dict]:
    tier_counts = {tier: sum(row["meta"]["tier"] == tier for row in val)
                   for tier in TARGET_TIER_PROFILE}
    fault_counts = {fault: sum(str(len(row["meta"].get("faults", []))) == fault for row in val)
                    for fault in TARGET_FAULT_PROFILE}
    n = len(val)
    weights = []
    for row in val:
        tier = row["meta"]["tier"]
        fault = str(len(row["meta"].get("faults", [])))
        tier_ratio = TARGET_TIER_PROFILE[tier] / max(tier_counts[tier] / n, 1e-12)
        fault_ratio = TARGET_FAULT_PROFILE[fault] / max(fault_counts[fault] / n, 1e-12)
        weights.append(tier_ratio * fault_ratio)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= max(float(weights.mean()), 1e-12)
    return weights, {
        "target_tier_profile": TARGET_TIER_PROFILE,
        "target_fault_profile": TARGET_FAULT_PROFILE,
        "validation_tier_counts": tier_counts,
        "validation_fault_counts": fault_counts,
        "validation_weight_mean": float(weights.mean()),
        "validation_weight_min": float(weights.min()),
        "validation_weight_max": float(weights.max()),
    }


def _weighted_kernel(train: np.ndarray, val: np.ndarray, weights: np.ndarray,
                     metric: str) -> tuple[np.ndarray, float]:
    if metric == "cosine":
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (train @ val.T)))
    else:
        dist = np.sqrt(np.maximum(0.0, ((train[:, None] - val[None, :]) ** 2).sum(axis=-1)))
    positive = dist[dist > 1e-8]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    return (similarity * weights[None, :]).sum(axis=1) / weights.sum(), sigma


def score_utility(train: list[dict], val: list[dict], model_path: str,
                  batch_size: int, device: str):
    from sentence_transformers import SentenceTransformer

    weights, weight_report = _target_weights(val)
    encoder = SentenceTransformer(model_path, device=device)
    train_text = encoder.encode([row["input"] for row in train], normalize_embeddings=True,
                                convert_to_numpy=True, batch_size=batch_size,
                                show_progress_bar=True).astype(np.float32)
    val_text = encoder.encode([row["input"] for row in val], normalize_embeddings=True,
                              convert_to_numpy=True, batch_size=batch_size,
                              show_progress_bar=True).astype(np.float32)
    text_u, text_sigma = _weighted_kernel(train_text, val_text, weights, "cosine")
    train_struct = np.stack([_structure(row) for row in train])
    val_struct = np.stack([_structure(row) for row in val])
    mean, std = train_struct.mean(axis=0), train_struct.std(axis=0)
    std[std < 1e-6] = 1.0
    struct_u, struct_sigma = _weighted_kernel(
        (train_struct - mean) / std, (val_struct - mean) / std, weights, "euclidean",
    )
    text_rank, struct_rank = rank01(text_u), rank01(struct_u)
    utility = np.sqrt(text_rank * struct_rank)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return utility, {
        "utility_definition": "weighted_validation_text_kernel_geometric_mean_structure_kernel",
        "utility_text_sigma": text_sigma,
        "utility_structure_sigma": struct_sigma,
        "utility_text_mean": float(np.mean(text_u)),
        "utility_structure_mean": float(np.mean(struct_u)),
        "utility_text_rank_mean": float(np.mean(text_rank)),
        "utility_structure_rank_mean": float(np.mean(struct_rank)),
        **weight_report,
    }


def window_order_from_due(due: np.ndarray, batch_size: int, alpha: float, seed: int):
    due = np.asarray(due, dtype=np.float64)
    if due.ndim != 1 or not len(due):
        return []
    if batch_size <= 0 or not 0 < alpha <= 1:
        raise ValueError("invalid batch_size or alpha")
    n = len(due)
    ranking = sorted(range(n), key=lambda i: (float(due[i]), i))
    rng = random.Random(seed)
    steps = math.ceil(n / batch_size)
    reach = max(1, math.ceil(alpha * steps))
    used = set()
    order = []

    def pool_fraction(step_idx: int) -> float:
        if step_idx <= 0:
            return 0.10
        if step_idx >= reach - 1:
            return 1.0
        return 0.10 + 0.90 * step_idx / (reach - 1)

    for step in range(steps):
        need = min(batch_size, n - len(order))
        local_step = step
        while True:
            pool_size = min(n, max(1, math.floor(pool_fraction(local_step) * n)))
            pool = [i for i in ranking[:pool_size] if i not in used]
            if len(pool) >= need:
                break
            if local_step < steps - 1:
                local_step += 1
                continue
            pool = [i for i in ranking if i not in used]
            break
        chosen = pool if len(pool) <= need else rng.sample(pool, need)
        used.update(chosen)
        order.extend(chosen)
    if sorted(order) != list(range(n)):
        raise AssertionError("window order is not a permutation")
    return order


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rank01(x), rank01(y))[0, 1])


def diagnostics(rows: list[dict], difficulty: np.ndarray, utility: np.ndarray,
                due: np.ndarray, order: list[int]) -> dict:
    plan_len = np.asarray([row["meta"].get("plan_len", 0) for row in rows], dtype=np.float64)
    faults = np.asarray([len(row["meta"].get("faults", [])) for row in rows], dtype=np.float64)
    position = np.empty(len(rows), dtype=np.float64)
    position[np.asarray(order)] = np.arange(len(order), dtype=np.float64)
    first = set(order[:max(1, len(order) // 5)])
    last = set(order[-max(1, len(order) // 5):])
    def composition(indices):
        subset = [rows[i] for i in indices]
        return {
            "tier": {k: sum(row["meta"]["tier"] == k for row in subset) / len(subset)
                     for k in ("T1", "T2", "T3")},
            "faults": {str(k): sum(len(row["meta"].get("faults", [])) == k for row in subset) / len(subset)
                       for k in (0, 1, 2)},
            "mean_utility": float(np.mean(utility[list(indices)])),
            "mean_difficulty": float(np.mean(difficulty[list(indices)])),
        }
    return {
        "spearman_difficulty_plan_len": _spearman(difficulty, plan_len),
        "spearman_difficulty_faults": _spearman(difficulty, faults),
        "spearman_position_utility": _spearman(position, utility),
        "spearman_position_difficulty": _spearman(position, difficulty),
        "spearman_position_due": _spearman(position, due),
        "first_20_percent": composition(first),
        "last_20_percent": composition(last),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--embed-model", default=EMBED_DEFAULT)
    ap.add_argument("--score-batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-positive-correlations", action="store_true")
    args = ap.parse_args()

    train, val = load_jsonl(args.train), load_jsonl(args.val)
    model, tokenizer = _load_lm_bf16(args.base)
    difficulty, difficulty_report = score_difficulty(
        train, model, tokenizer, args.score_batch_size, args.max_len,
    )
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    utility, utility_report = score_utility(
        train, val, args.embed_model, max(8, args.score_batch_size), args.embed_device,
    )
    due = rank01(difficulty) / rank01(utility)
    order = window_order_from_due(due, 4, args.alpha, args.seed)
    position = {index: pos for pos, index in enumerate(order)}
    diagnostic = diagnostics(train, difficulty, utility, due, order)
    if args.require_positive_correlations:
        if diagnostic["spearman_difficulty_plan_len"] <= 0 or diagnostic["spearman_difficulty_faults"] <= 0:
            raise RuntimeError("semantic Difficulty failed positive construct-validity checks")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for index, row in enumerate(train):
        item = dict(row)
        item.update({
            "bt_record_id": record_id(row),
            "bt_difficulty": float(difficulty[index]),
            "bt_utility": float(utility[index]),
            "bt_due": float(due[index]),
            "bt_order_position": position[index],
        })
        scored.append(item)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "ordered.jsonl").open("w", encoding="utf-8") as handle:
        for index in order:
            handle.write(json.dumps(scored[index], ensure_ascii=False) + "\n")
    report = {
        "score_version": "bt_ducl_v2_semantic_bf16_rank_due_targetmix",
        "n_train": len(train), "n_val": len(val), "order_is_permutation": True,
        "alpha": args.alpha, "alpha_source": "official_DUCL_window_quantile_default",
        "due_definition": "rank01_difficulty_divided_by_rank01_utility",
        "sampler": "official_DUCL_window_quantile_10_to_100_percent",
        **difficulty_report, **utility_report, **diagnostic,
    }
    (out_dir / "score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

