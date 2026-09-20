"""v4 DUCL scoring: semantic frozen-base Difficulty plus input-only Sinkhorn Utility."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from .common import load_jsonl, prompt_text, record_id
from .semantic import encode_semantic_example

BASE_DEFAULT = "models/llama32-1b"
EMBED_DEFAULT = "models/all-MiniLM-L12-v2"
TARGET_TIER_PROFILE = {"T1": 0.20, "T2": 0.40, "T3": 0.40}
TARGET_FAULT_PROFILE = {"0": 0.35, "1": 0.40, "2": 0.25}


def _load_lm(base: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def semantic_base_nll(rows: list[dict], model, tokenizer, batch_size: int,
                      max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return mean semantic NLL, semantic token count, and completion count."""
    device = next(model.parameters()).device
    nlls, counts, completion_counts = [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prepared = [encode_semantic_example(row, tokenizer, max_len) for row in chunk]
        width = max(len(x.input_ids) for x in prepared)
        ids = torch.full((len(chunk), width), tokenizer.pad_token_id,
                         dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, item in enumerate(prepared):
            size = len(item.input_ids)
            ids[i, :size] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
            attention[i, :size] = 1
            labels[i, :size] = torch.tensor(item.labels, dtype=torch.long, device=device)
        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
        target = labels[:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(target.shape)
        valid = target != -100
        for i in range(len(chunk)):
            count = int(valid[i].sum().item())
            if count <= 0:
                raise RuntimeError(f"row {start + i} has no semantic targets")
            nlls.append(float((token_loss[i] * valid[i]).sum().div(count).cpu()))
            counts.append(count)
            completion_counts.append(prepared[i].completion_tokens)
        done = min(start + batch_size, len(rows))
        if start == 0 or done == len(rows) or done % (batch_size * 25) == 0:
            print(f"semantic base NLL {done}/{len(rows)}", flush=True)
    nll = np.asarray(nlls, dtype=np.float64)
    count = np.asarray(counts, dtype=np.float64)
    completion = np.asarray(completion_counts, dtype=np.float64)
    # Mean surprise alone underweights long coordinated trees. This is still
    # frozen-base semantic NLL, with a transparent information-burden factor.
    burden = nll * count
    report = {
        "difficulty_definition": "frozen_base_total_semantic_nll",
        "semantic_nll_mean": float(nll.mean()),
        "semantic_nll_std": float(nll.std()),
        "semantic_burden_mean": float(burden.mean()),
        "semantic_burden_std": float(burden.std()),
        "semantic_token_mean": float(count.mean()),
        "semantic_token_min": int(count.min()),
        "semantic_token_max": int(count.max()),
        "completion_token_mean": float(completion.mean()),
        "semantic_fraction": float(count.sum() / max(completion.sum(), 1.0)),
    }
    return burden, nll, count, report


def _target_weights(val: list[dict]) -> tuple[np.ndarray, dict]:
    """Reweight held-out validation to the frozen deployment target mix."""
    n = len(val)
    tier_counts = {key: sum(row["meta"]["tier"] == key for row in val)
                   for key in TARGET_TIER_PROFILE}
    fault_counts = {key: sum(str(len(row["meta"].get("faults", []))) == key for row in val)
                    for key in TARGET_FAULT_PROFILE}
    weights = []
    for row in val:
        tier = row["meta"]["tier"]
        fault = str(len(row["meta"].get("faults", [])))
        weights.append(
            TARGET_TIER_PROFILE[tier] / max(tier_counts[tier] / n, 1e-12)
            * TARGET_FAULT_PROFILE[fault] / max(fault_counts[fault] / n, 1e-12)
        )
    weights = np.asarray(weights, dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    return weights, {
        "target_tier_profile": TARGET_TIER_PROFILE,
        "target_fault_profile": TARGET_FAULT_PROFILE,
        "validation_tier_counts": tier_counts,
        "validation_fault_counts": fault_counts,
        "target_weight_min": float(weights.min()),
        "target_weight_max": float(weights.max()),
    }


def _input_embeddings(rows: list[dict], model_path: str, device: str,
                      batch_size: int) -> tuple[np.ndarray, dict]:
    """Embed only the natural-language input, retaining long-input chunks."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_path, device=device)
    tok = encoder.tokenizer
    max_tokens = max(8, int(getattr(encoder, "max_seq_length", 128)) - 2)
    pieces, owners, counts = [], [], []
    for i, row in enumerate(rows):
        ids = tok.encode(row["input"], add_special_tokens=False)
        n = max(1, math.ceil(len(ids) / max_tokens))
        counts.append(n)
        for j in range(n):
            part = ids[j * max_tokens:(j + 1) * max_tokens]
            pieces.append(tok.decode(part, clean_up_tokenization_spaces=False))
            owners.append(i)
    vectors = encoder.encode(
        pieces, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=batch_size, show_progress_bar=True,
    ).astype(np.float32)
    result = np.zeros((len(rows), vectors.shape[1]), dtype=np.float32)
    owners_arr = np.asarray(owners)
    for i in range(len(rows)):
        value = vectors[owners_arr == i].mean(axis=0)
        result[i] = value / max(float(np.linalg.norm(value)), 1e-12)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, {
        "utility_definition": "input_only_minilm_chunk_mean_sinkhorn_target_potential",
        "embedding_max_tokens_per_chunk": max_tokens,
        "embedding_chunks_total": len(pieces),
        "embedding_chunks_mean": float(np.mean(counts)),
        "embedding_chunks_max": int(np.max(counts)),
    }


def sinkhorn_utility(train_emb: np.ndarray, val_emb: np.ndarray,
                     semantic_nll: np.ndarray, val_weights: np.ndarray,
                     temperature: float = 0.05, epsilon: float = 0.2,
                     blur: float = 0.1, device: str = "cuda") -> tuple[np.ndarray, dict]:
    if len(train_emb) != len(semantic_nll):
        raise ValueError("embedding/loss length mismatch")
    if not temperature > 0 or not 0 < epsilon < 1:
        raise ValueError("invalid Sinkhorn parameters")
    from geomloss import SamplesLoss

    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available()
                       else "cpu")
    source = torch.as_tensor(train_emb, dtype=torch.float32, device=dev)
    target = torch.as_tensor(val_emb, dtype=torch.float32, device=dev)
    n = source.shape[0]
    source_w = torch.softmax(torch.as_tensor(-semantic_nll / temperature,
                                             dtype=torch.float32, device=dev), dim=0)
    # The first copy represents the current model proxy; the second copy is
    # the candidate sample being considered for utility.
    candidates = torch.cat((source, source), dim=0)
    candidate_w = torch.cat(((1.0 - epsilon) * source_w,
                             torch.full((n,), epsilon / n, device=dev)))
    target_w = torch.as_tensor(val_weights, dtype=torch.float32, device=dev)
    target_w = target_w / target_w.sum()
    ot = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=False,
                     verbose=False, potentials=True)
    with torch.no_grad():
        potential, _ = ot(candidate_w, candidates, target_w, target)
    raw_utility = -np.asarray(potential.detach().cpu(), dtype=np.float64).reshape(-1)[n:]
    utility = raw_utility - float(raw_utility.min()) + 1e-9
    source_np = source_w.detach().cpu().numpy()
    report = {
        "mu_p_temperature": temperature,
        "mu_p_effective_sample_size": float(1.0 / np.sum(source_np ** 2)),
        "mu_p_max_weight": float(source_np.max()),
        "epsilon": epsilon,
        "blur": blur,
        "utility_raw_mean": float(raw_utility.mean()),
        "utility_raw_std": float(raw_utility.std()),
        "utility_mean": float(utility.mean()),
        "utility_std": float(utility.std()),
    }
    return utility, report


def tie_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=np.float64), method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    a, b = tie_rank(x), tie_rank(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def diagnostics(rows: list[dict], difficulty: np.ndarray, utility: np.ndarray,
                due: np.ndarray, semantic_counts: np.ndarray) -> dict:
    plan_len = np.asarray([row["meta"].get("plan_len", 0) for row in rows], dtype=np.float64)
    faults = np.asarray([len(row["meta"].get("faults", [])) for row in rows], dtype=np.float64)
    return {
        "spearman_difficulty_plan_len": spearman(difficulty, plan_len),
        "spearman_difficulty_faults": spearman(difficulty, faults),
        "spearman_difficulty_utility": spearman(difficulty, utility),
        "spearman_due_utility": spearman(due, utility),
        "semantic_count_plan_len": spearman(semantic_counts, plan_len),
        "diagnostic_acceptance": {
            "difficulty_plan_len_positive": spearman(difficulty, plan_len) > 0,
            "difficulty_faults_positive": spearman(difficulty, faults) > 0,
            "du_not_collapsed_abs_spearman_lt_0_8": abs(spearman(difficulty, utility)) < 0.8,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--embed-model", default=EMBED_DEFAULT)
    ap.add_argument("--score-batch-size", type=int, default=2)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--ot-device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--blur", type=float, default=0.1)
    ap.add_argument("--allow-diagnostic-failure", action="store_true")
    args = ap.parse_args()
    train, val = load_jsonl(args.train), load_jsonl(args.val)
    model, tokenizer = _load_lm(args.base)
    burden, semantic_nll, semantic_counts, diff_report = semantic_base_nll(
        train, model, tokenizer, args.score_batch_size, args.max_len)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_emb, train_embed_report = _input_embeddings(
        train, args.embed_model, args.embed_device, args.embed_batch_size)
    val_emb, val_embed_report = _input_embeddings(
        val, args.embed_model, args.embed_device, args.embed_batch_size)
    val_weights, target_report = _target_weights(val)
    utility, utility_report = sinkhorn_utility(
        train_emb, val_emb, semantic_nll, val_weights, args.temperature,
        args.epsilon, args.blur, args.ot_device)
    d_rank, u_rank = tie_rank(burden), tie_rank(utility)
    due = d_rank / np.maximum(u_rank, 1e-12)
    diagnostics_report = diagnostics(train, burden, utility, due, semantic_counts)
    ess_ok = 100.0 <= utility_report["mu_p_effective_sample_size"] <= 1000.0
    diagnostics_report["diagnostic_acceptance"]["mu_p_ess_between_100_and_1000"] = ess_ok
    accepted = all(diagnostics_report["diagnostic_acceptance"].values())
    if not accepted and not args.allow_diagnostic_failure:
        raise RuntimeError("v4 scoring construct-validity checks failed: "
                           + json.dumps(diagnostics_report["diagnostic_acceptance"]))
    # Store scores in original row order. The sampler consumes this file and
    # therefore keeps record identity independent of the curriculum schedule.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for i, row in enumerate(train):
        item = dict(row)
        item.update({"v4_record_id": record_id(row), "v4_semantic_nll": float(semantic_nll[i]),
                     "v4_semantic_tokens": int(semantic_counts[i]),
                     "v4_difficulty": float(burden[i]), "v4_utility": float(utility[i]),
                     "v4_due": float(due[i]), "v4_d_rank": float(d_rank[i]),
                     "v4_u_rank": float(u_rank[i])})
        scores.append(item)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for item in scores:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {"score_version": "bt_ducl_v4_semantic_nll_input_sinkhorn_rank_due",
              "n_train": len(train), "n_val": len(val), "batch_size": args.batch_size,
              "alpha": args.alpha, "order_is_permutation": True,
              **diff_report, **train_embed_report, **val_embed_report,
              **target_report, **utility_report, **diagnostics_report,
              "diagnostic_accepted": accepted}
    (out_dir / "score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
