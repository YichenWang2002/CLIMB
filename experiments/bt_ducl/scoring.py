"""Static Difficulty/Utility/DUE scoring for the BT generation experiment.

Difficulty is the frozen base model's length-normalized loss on the target XML
tokens. Utility is validation coverage in both language and executable BT
structure spaces. The two utility views are multiplied after independent,
data-derived kernel scaling; no hand-written mixture coefficient is used.
"""
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

BASE_DEFAULT = "models/llama32-1b"
EMBED_DEFAULT = "models/all-MiniLM-L12-v2"
ACTION_TAGS = {
    "MoveTo", "PickUp", "PlaceDown", "Recharge", "ClearPath",
    "HandoverGive", "HandoverTake", "CoPickUp", "CoMoveTo", "CoPlaceDown",
    "CatalogBook", "WaterPlants", "InspectPlants",
}
TAG_RE = re.compile(r"</?([A-Za-z_][\w.-]*)[\s/>]")


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return (ranks + 0.5) / max(len(values), 1)


def _load_lm(base: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tok


def _semantic_output_ids(row: dict, tokenizer, max_len: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt_text(row, tokenizer), add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(row["output"] + tokenizer.eos_token,
                           add_special_tokens=False)["input_ids"]
    keep = max_len - len(prompt_ids)
    if keep <= 0:
        return prompt_ids[:max_len], []
    output_ids = output_ids[:keep]
    return prompt_ids + output_ids, output_ids


@torch.no_grad()
def score_difficulty(rows: list[dict], model, tokenizer,
                     batch_size: int, max_len: int) -> tuple[np.ndarray, dict]:
    """Frozen-base semantic difficulty, using only target XML token loss."""
    device = next(model.parameters()).device
    values = []
    token_counts = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        sequences, outputs = zip(*[_semantic_output_ids(r, tokenizer, max_len)
                                   for r in chunk])
        max_width = max(map(len, sequences))
        ids = torch.full((len(chunk), max_width), tokenizer.pad_token_id,
                         dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (seq, out_ids) in enumerate(zip(sequences, outputs)):
            n = len(seq)
            ids[i, :n] = torch.tensor(seq, device=device)
            mask[i, :n] = 1
            prompt_len = n - len(out_ids)
            if out_ids:
                labels[i, prompt_len:n] = torch.tensor(out_ids, device=device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        target = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(target.shape)
        valid = target != -100
        for i in range(len(chunk)):
            denom = valid[i].sum().clamp_min(1)
            values.append(float((losses[i] * valid[i]).sum().div(denom).cpu()))
            token_counts.append(int(valid[i].sum().item()))
        done = min(start + batch_size, len(rows))
        if start == 0 or done == len(rows) or done % (batch_size * 25) == 0:
            print(f"difficulty {done}/{len(rows)}", flush=True)
    if not all(token_counts):
        raise RuntimeError("some records have no target XML tokens after truncation")
    return np.asarray(values, dtype=np.float64), {
        "difficulty_definition": "frozen_base_length_normalized_xml_target_nll",
        "difficulty_token_mean": float(np.mean(token_counts)),
        "difficulty_token_min": int(np.min(token_counts)),
    }


def _structure(row: dict) -> np.ndarray:
    meta = row["meta"]
    tags = TAG_RE.findall(row["output"])
    counts = {tag: tags.count(tag) for tag in ACTION_TAGS}
    return np.asarray([
        float(meta.get("plan_len", 0)), float(meta.get("n_agents", 0)),
        float(len(meta.get("faults", []))), float(counts["HandoverGive"] + counts["HandoverTake"]),
        float(counts["CoPickUp"] + counts["CoMoveTo"] + counts["CoPlaceDown"]),
        float(counts["Recharge"] + counts["ClearPath"]),
        float(tags.count("Fallback")), float(tags.count("Parallel")),
        float(len(tags)),
    ], dtype=np.float32)


def _kernel_coverage(train: np.ndarray, val: np.ndarray, metric: str) -> tuple[np.ndarray, float]:
    if metric == "cosine":
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (train @ val.T)))
    else:
        dist = np.sqrt(np.maximum(0.0, ((train[:, None] - val[None, :]) ** 2).sum(axis=-1)))
    positive = dist[dist > 1e-8]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    return similarity.mean(axis=1), sigma


def score_utility(train: list[dict], val: list[dict], model_path: str,
                  batch_size: int, device: str) -> tuple[np.ndarray, dict]:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_path, device=device)
    tr_text = encoder.encode([r["input"] for r in train], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=batch_size,
                             show_progress_bar=True).astype(np.float32)
    va_text = encoder.encode([r["input"] for r in val], normalize_embeddings=True,
                             convert_to_numpy=True, batch_size=batch_size,
                             show_progress_bar=True).astype(np.float32)
    text_u, text_sigma = _kernel_coverage(tr_text, va_text, "cosine")
    tr_struct = np.stack([_structure(r) for r in train])
    va_struct = np.stack([_structure(r) for r in val])
    mean, std = tr_struct.mean(axis=0), tr_struct.std(axis=0)
    std[std < 1e-6] = 1.0
    struct_u, struct_sigma = _kernel_coverage((tr_struct - mean) / std,
                                               (va_struct - mean) / std, "euclidean")
    utility = rank01(text_u) * rank01(struct_u)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return utility, {
        "utility_definition": "validation_text_kernel_times_executable_structure_kernel",
        "utility_text_sigma": text_sigma,
        "utility_structure_sigma": struct_sigma,
        "utility_text_mean": float(np.mean(text_u)),
        "utility_structure_mean": float(np.mean(struct_u)),
    }


def due_order(difficulty: np.ndarray, utility: np.ndarray,
              batch_size: int, alpha: float, seed: int) -> tuple[list[int], np.ndarray]:
    """Official DUCL quantile-window ordering over raw DUE = D / U.

    This mirrors DUCL-main/scripts/data_order.py::window_quantile: the eligible
    pool grows linearly from the easiest 10% to the full dataset and reaches the
    full dataset after alpha of the ordering steps.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    difficulty = np.asarray(difficulty, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if difficulty.shape != utility.shape or difficulty.ndim != 1:
        raise ValueError("difficulty and utility must be same-length vectors")
    if not np.all(np.isfinite(difficulty)) or np.any(difficulty < 0):
        raise ValueError("difficulty must be finite and non-negative")
    if not np.all(np.isfinite(utility)) or np.any(utility <= 0):
        raise ValueError("utility must be finite and strictly positive")

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    n = len(difficulty)
    if n == 0:
        return [], np.asarray([], dtype=np.float64)
    due = difficulty / utility
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
        progress = step_idx / (reach - 1)
        return 0.10 + progress * 0.90

    for step in range(steps):
        need = min(batch_size, n - len(order))
        step_local = step
        while True:
            pool_size = min(n, max(1, math.floor(pool_fraction(step_local) * n)))
            pool = [i for i in ranking[:pool_size] if i not in used]
            if len(pool) >= need:
                break
            if step_local < steps - 1:
                step_local += 1
                continue
            pool = [i for i in ranking if i not in used]
            break
        chosen = pool if len(pool) <= need else rng.sample(pool, need)
        used.update(chosen)
        order.extend(chosen)
    if len(order) != n or len(set(order)) != n:
        raise AssertionError("DUE order is not a permutation")
    return order, due


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--embed-model", default=EMBED_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--score-batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train, val = load_jsonl(args.train), load_jsonl(args.val)
    model, tok = _load_lm(args.base)
    diff, diff_report = score_difficulty(train, model, tok, args.score_batch_size, args.max_len)
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    util, util_report = score_utility(train, val, args.embed_model,
                                      max(8, args.score_batch_size), args.embed_device)
    order, due = due_order(diff, util, args.batch_size, args.alpha, args.seed)
    position = {idx: pos for pos, idx in enumerate(order)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path, ordered_path = out_dir / "scores.jsonl", out_dir / "ordered.jsonl"
    with scored_path.open("w", encoding="utf-8") as sf, ordered_path.open("w", encoding="utf-8") as of:
        scored = []
        for i, row in enumerate(train):
            item = dict(row)
            item.update({"bt_record_id": record_id(row), "bt_difficulty": float(diff[i]),
                         "bt_utility": float(util[i]), "bt_due": float(due[i]),
                         "bt_order_position": position[i]})
            scored.append(item)
            sf.write(json.dumps(item, ensure_ascii=False) + "\n")
        for i in order:
            of.write(json.dumps(scored[i], ensure_ascii=False) + "\n")
    report = {"n_train": len(train), "n_val": len(val), "order_is_permutation": True,
              "alpha": args.alpha,
              "alpha_source": "official_DUCL_window_quantile_default",
              "due_definition": "raw_difficulty_divided_by_raw_utility",
              "sampler": "official_DUCL_window_quantile_10_to_100_percent",
              **diff_report, **util_report}
    (out_dir / "score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
