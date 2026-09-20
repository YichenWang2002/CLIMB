"""Model-target DUCL scoring for the fair 3 x 1-epoch experiment.

The implementation follows the DUCL decomposition while keeping every
combination data-adaptive:

* Difficulty is an information-weighted semantic XML score.  It combines
  teacher-forced NLL and top-1 error on opening-tag tokens.  Token weights are
  corpus IDF weights and the two signals are fused with the first PCA loading;
  there is no hand-written 0.7/0.3 mixture.
* Utility combines language-space validation coverage with executable BT
  structure coverage.  The two views are fused by the same adaptive PCA rule.
* DUE is the rank-normalised D/U ratio.  Window Ordering samples uniformly
  without replacement from the expanding DUE window, as in DUCL.

The validation set is used only for Utility.  Test data never enters scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from curriculum.structural import structural_features
from experiments.bt_ducl.common import apply_chat_template_compat

BASE = "models/llama32-1b"
ENCODER = "models/all-MiniLM-L12-v2"
OPENING_TAG = re.compile(r"<(?![/!?])[^>]+>")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def prompt_for(record: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": record["instruction"]},
        {"role": "user", "content": record["input"]},
    ]
    return apply_chat_template_compat(tokenizer, messages, add_generation_prompt=True)


def record_id(record: dict) -> str:
    payload = json.dumps(
        {k: record.get(k) for k in ("instruction", "input", "output")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_model(base: str, adapter: str | None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def midrank01(values: np.ndarray) -> np.ndarray:
    """Stable percentile ranks in (0, 1), with no distributional prior."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return (ranks + 0.5) / n


def adaptive_fuse(columns: np.ndarray, mode: str = "pca") -> tuple[np.ndarray, list[float]]:
    """Fuse equally-scaled signals using data-derived PCA loadings.

    Absolute loadings are used because all input columns are oriented so that
    larger means more difficult or more useful.  A constant signal receives a
    uniform loading only when the data itself contains no variance.
    """
    x = np.asarray(columns, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[1] == 1:
        return x[:, 0], [1.0]
    if mode == "mean":
        return x.mean(axis=1), [float(1.0 / x.shape[1])] * x.shape[1]
    if mode != "pca":
        raise ValueError(f"unknown fusion mode: {mode}")
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(x), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    loading = np.abs(eigvecs[:, int(np.argmax(eigvals))])
    if not np.isfinite(loading).all() or loading.sum() <= 1e-12:
        loading = np.ones(x.shape[1], dtype=np.float64)
    loading /= loading.sum()
    return x @ loading, [float(v) for v in loading]


def _idf_vector(records: list[dict], tokenizer) -> np.ndarray:
    """Corpus-derived information weights for output tokens."""
    df: dict[int, int] = {}
    for record in records:
        ids = tokenizer(record["output"], add_special_tokens=False)["input_ids"]
        for token_id in set(ids):
            df[int(token_id)] = df.get(int(token_id), 0) + 1
    vocab = int(getattr(tokenizer, "vocab_size", 0))
    if vocab <= 0:
        vocab = max(df.keys(), default=0) + 1
    vec = np.zeros(vocab, dtype=np.float32)
    n = len(records)
    for token_id, count in df.items():
        if token_id >= len(vec):
            continue
        vec[token_id] = math.log((n + 1.0) / (count + 1.0))
    return vec


@torch.no_grad()
def score_difficulty(records: list[dict], model, tokenizer,
                     batch_size: int, max_len: int,
                     fusion_mode: str = "pca") -> tuple[list[float], list[float], list[float], list[int], dict]:
    """Return semantic NLL, semantic error, and semantic token counts."""
    device = next(model.parameters()).device
    idf = _idf_vector(records, tokenizer)
    nll_scores: list[float] = []
    error_scores: list[float] = []
    token_counts: list[int] = []
    fallback_count = 0
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prompts = [prompt_for(r, tokenizer) for r in chunk]
        fulls = [p + r["output"] + tokenizer.eos_token
                 for p, r in zip(prompts, chunk)]
        raw = tokenizer(
            fulls, return_tensors="pt", padding=True, truncation=True,
            max_length=max_len, add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = raw.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in raw.items()}
        labels = enc["input_ids"].clone()
        semantic = torch.zeros_like(labels, dtype=torch.bool, device=device)
        for j, (prompt, full) in enumerate(zip(prompts, fulls)):
            plen = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[j, :min(plen, labels.shape[1])] = -100
            output_start = len(prompt)
            intervals = [(output_start + m.start(), output_start + m.end())
                         for m in OPENING_TAG.finditer(full[output_start:])]
            for pos, (a, b) in enumerate(offsets[j].tolist()):
                if b <= a or pos >= labels.shape[1]:
                    continue
                if any(a < end and b > begin for begin, end in intervals):
                    semantic[j, pos] = True
        labels[enc["attention_mask"] == 0] = -100
        # Keep the model output in bf16 and materialize fp32 only one row at
        # a time.  DeepSeek/Qwen's ~152k vocabulary makes a full-batch cast
        # exceed a 24GB card even though the model itself fits comfortably.
        logits = model(**enc).logits[:, :-1, :]
        target = labels[:, 1:]
        sem = semantic[:, 1:] & (target != -100)
        for row in range(len(chunk)):
            mask = sem[row]
            if not bool(mask.any()):
                mask = target[row] != -100
                fallback_count += 1
            row_logits = logits[row].float()
            row_target = target[row]
            predictions = row_logits.argmax(dim=-1)
            token_loss = torch.nn.functional.cross_entropy(
                row_logits, row_target, reduction="none", ignore_index=-100,
            )
            ids = target[row].clamp_min(0).detach().cpu().numpy()
            weights_np = np.asarray(
                [idf[int(t)] if int(t) < len(idf) else 0.0 for t in ids],
                dtype=np.float32,
            )
            weights = torch.as_tensor(weights_np, device=device)
            weights = weights * mask.float()
            if float(weights.sum()) <= 1e-12:
                weights = mask.float()
            denom = weights.sum().clamp_min(1.0)
            nll_scores.append(float((token_loss * weights).sum().div(denom).cpu()))
            errors = (predictions != row_target).float()
            error_scores.append(float((errors * weights).sum().div(denom).cpu()))
            token_counts.append(int(mask.sum().item()))
            del row_logits, predictions, token_loss
        del logits
        done = min(start + batch_size, len(records))
        if start == 0 or done == len(records) or done % (batch_size * 25) == 0:
            print(f"semantic difficulty {done}/{len(records)}", flush=True)
    ranked, weights = adaptive_fuse(np.column_stack(
        [midrank01(np.asarray(nll_scores)), midrank01(np.asarray(error_scores))]),
        mode=fusion_mode)
    report = {
        "difficulty_components": ["semantic_idf_weighted_nll", "semantic_idf_weighted_error"],
        "difficulty_fusion": "uniform_mean" if fusion_mode == "mean" else "PCA_abs_loading_on_midrank_components",
        "difficulty_adaptive_weights": weights,
        "semantic_tokens_mean": float(np.mean(token_counts)),
        "semantic_tokens_p10": float(np.percentile(token_counts, 10)),
        "semantic_tokens_zero_fallback": fallback_count,
    }
    return [float(x) for x in ranked], nll_scores, error_scores, token_counts, report


def _struct_vector(record: dict) -> list[float]:
    meta = record.get("meta", {})
    try:
        feat = structural_features(record["output"])
    except Exception:
        feat = {"n_sync": 0, "n_co": 0, "n_fb": 0, "depth": 0, "n_nodes": 0}
    return [
        float(meta.get("plan_len", 0)), float(meta.get("n_agents", 0)),
        float(len(meta.get("faults", []))), float(feat["n_sync"]),
        float(feat["n_co"]), float(feat["n_fb"]), float(feat["depth"]),
        float(feat["n_nodes"]),
    ]


def _knn_utility(train_emb: np.ndarray, val_emb: np.ndarray, k: int,
                 metric: str) -> tuple[np.ndarray, float]:
    k = min(max(1, k), len(val_emb))
    nearest = []
    for start in range(0, len(train_emb), 512):
        chunk = train_emb[start:start + 512]
        if metric == "cosine":
            sim = chunk @ val_emb.T
            dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sim))
        elif metric == "euclidean":
            delta = chunk[:, None, :] - val_emb[None, :, :]
            dist = np.sqrt(np.maximum(0.0, np.sum(delta * delta, axis=2)))
        else:
            raise ValueError(f"unknown kNN metric: {metric}")
        nearest.append(np.partition(dist, kth=k - 1, axis=1)[:, :k])
    knn = np.concatenate(nearest, axis=0)
    positive = knn[knn > 1e-8]
    if len(positive) == 0:
        return np.ones(len(train_emb), dtype=np.float64), 0.0
    sigma = float(np.median(positive))
    return np.exp(-(knn ** 2) / (2.0 * sigma ** 2)).mean(axis=1), sigma


def score_utility(train: list[dict], val: list[dict], device: str,
                  k: int, batch_size: int,
                  fusion_mode: str = "pca") -> tuple[np.ndarray, dict]:
    """Fuse text coverage and executable behavior-tree structure coverage."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER, device=device)
    train_emb = encoder.encode(
        [r["input"] for r in train], normalize_embeddings=True,
        convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True,
    ).astype(np.float32)
    val_emb = encoder.encode(
        [r["input"] for r in val], normalize_embeddings=True,
        convert_to_numpy=True, batch_size=batch_size, show_progress_bar=True,
    ).astype(np.float32)
    text_utility, text_sigma = _knn_utility(train_emb, val_emb, k, "cosine")
    train_struct = np.asarray([_struct_vector(r) for r in train], dtype=np.float32)
    val_struct = np.asarray([_struct_vector(r) for r in val], dtype=np.float32)
    mean = train_struct.mean(axis=0)
    std = train_struct.std(axis=0)
    std[std < 1e-6] = 1.0
    struct_utility, struct_sigma = _knn_utility(
        (train_struct - mean) / std, (val_struct - mean) / std, k, "euclidean")
    utility, weights = adaptive_fuse(np.column_stack(
        [midrank01(text_utility), midrank01(struct_utility)]), mode=fusion_mode)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return utility, {
        "utility_components": ["text_validation_coverage", "executable_bt_structure_coverage"],
        "utility_fusion": "uniform_mean" if fusion_mode == "mean" else "PCA_abs_loading_on_midrank_components",
        "utility_adaptive_weights": weights,
        "utility_text_sigma": text_sigma,
        "utility_struct_sigma": struct_sigma,
        "utility_text_mean": float(np.mean(text_utility)),
        "utility_struct_mean": float(np.mean(struct_utility)),
        "utility_text": text_utility,
        "utility_struct": struct_utility,
    }


def load_utility_source(path: str, train: list[dict]) -> tuple[np.ndarray, dict]:
    source = load_jsonl(path)
    if len(source) != len(train):
        raise ValueError(f"utility source length {len(source)} != train length {len(train)}")
    for i, (a, b) in enumerate(zip(source, train)):
        if a.get("mt_record_id") and a["mt_record_id"] != record_id(b):
            raise ValueError(f"utility source provenance mismatch at row {i}")
        if "mt_utility" not in a:
            raise ValueError(f"utility source row {i} has no mt_utility")
    return np.asarray([float(r["mt_utility"]) for r in source]), {
        "utility_reused_from": str(path),
        "utility_components": ["text_validation_coverage", "executable_bt_structure_coverage"],
        "utility_adaptive_weights": None,
    }


def ordered_indices(difficulty: list[float], utility: list[float],
                    batch_size: int, alpha: float, seed: int) -> tuple[list[int], np.ndarray]:
    """True DUCL D/U ranking followed by uniform expanding-window ordering."""
    rng = np.random.default_rng(seed)
    n = len(difficulty)
    d_rank = midrank01(np.asarray(difficulty))
    u_rank = midrank01(np.asarray(utility))
    due = d_rank / np.maximum(u_rank, 1e-12)
    ranking = sorted(range(n), key=lambda i: (float(due[i]), i))
    steps = math.ceil(n / batch_size)
    reach = max(1, math.ceil(alpha * steps))
    used = np.zeros(n, dtype=bool)
    order: list[int] = []
    for step in range(steps):
        frac = min(1.0, (step + 1) / reach)
        end = max(1, math.ceil(frac * n))
        pool = [i for i in ranking[:end] if not used[i]]
        if len(pool) < min(batch_size, n - len(order)):
            pool.extend(i for i in ranking[end:] if not used[i] and i not in pool)
        take = min(batch_size, n - len(order))
        chosen = pool if len(pool) <= take else rng.choice(pool, size=take, replace=False).tolist()
        for i in chosen:
            used[i] = True
        order.extend(chosen)
    assert len(order) == n and len(set(order)) == n
    return order, due


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ordered-out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--utility-source", default=None,
                    help="reuse round-0 utility after verifying record IDs")
    args = ap.parse_args()

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    model, tokenizer = load_model(args.base, args.adapter)
    difficulty, nll_scores, error_scores, token_counts, diff_report = score_difficulty(
        train, model, tokenizer, args.batch_size, args.max_len)
    if args.utility_source:
        utility, util_report = load_utility_source(args.utility_source, train)
        util_report["utility_reused"] = True
        text_utility = [None] * len(train)
        struct_utility = [None] * len(train)
    else:
        utility, util_report = score_utility(
            train, val, args.embed_device, args.knn, max(args.batch_size, 32))
        text_utility = util_report.pop("utility_text")
        struct_utility = util_report.pop("utility_struct")
        util_report["utility_reused"] = False
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    order, due = ordered_indices(difficulty, utility, args.batch_size, args.alpha, args.seed)
    positions = {sample_index: position for position, sample_index in enumerate(order)}
    scored = []
    for i, record in enumerate(train):
        row = dict(record)
        row["mt_record_id"] = record_id(record)
        row["mt_difficulty"] = float(difficulty[i])
        row["mt_difficulty_nll"] = float(nll_scores[i])
        row["mt_difficulty_error"] = float(error_scores[i])
        row["mt_semantic_tokens"] = int(token_counts[i])
        row["mt_utility_text"] = None if text_utility[i] is None else float(text_utility[i])
        row["mt_utility_struct"] = None if struct_utility[i] is None else float(struct_utility[i])
        row["mt_utility"] = float(utility[i])
        row["mt_due"] = float(due[i])
        row["mt_order_position"] = int(positions[i])
        scored.append(row)
    scored_path = Path(args.out)
    ordered_path = Path(args.ordered_out)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with ordered_path.open("w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(scored[i], ensure_ascii=False) + "\n")
    report = {
        "n_train": len(train), "n_val": len(val), "adapter": args.adapter,
        "difficulty_mean": float(np.mean(difficulty)),
        "difficulty_p50": float(np.percentile(difficulty, 50)),
        "difficulty_p90": float(np.percentile(difficulty, 90)),
        "difficulty_error_mean": float(np.mean(error_scores)),
        "utility_mean": float(np.mean(utility)),
        "utility_p50": float(np.percentile(utility, 50)),
        "due_p50": float(np.percentile(due, 50)),
        "due_p90": float(np.percentile(due, 90)),
        "order_is_permutation": len(order) == len(set(order)) == len(train),
        "sampler": "DUCL_D_over_U_rank + uniform_expanding_window",
        "alpha": args.alpha,
        **diff_report,
        **util_report,
    }
    scored_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
