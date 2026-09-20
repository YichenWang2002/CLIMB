"""Official DUCL-style Wasserstein DUE scoring for the mBT task.

This module deliberately keeps the curriculum score independent of the held-out
test split.  A frozen base-model completion loss is used only to form the
proxy distribution ``mu_p``; Sinkhorn potentials then provide the per-example
Difficulty and target-validation Utility values, following DUCL-main.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .common import load_jsonl, prompt_text, record_id
from .scoring_v2 import window_order_from_due

BASE_DEFAULT = "models/llama32-1b"
EMBED_DEFAULT = "models/all-MiniLM-L12-v2"


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
def base_completion_losses(rows: list[dict], model, tokenizer,
                           batch_size: int, max_len: int) -> np.ndarray:
    """Length-normalized NLL of the XML completion under the frozen base."""
    device = next(model.parameters()).device
    losses: list[float] = []
    counts: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [prompt_text(r, tokenizer) for r in chunk]
        fulls = [p + r["output"] + tokenizer.eos_token
                 for p, r in zip(prompts, chunk)]
        enc = tokenizer(fulls, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len,
                        add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, :-1].float()
        labels = enc["input_ids"][:, 1:].clone()
        for j, p in enumerate(prompts):
            plen = len(tokenizer(p, add_special_tokens=False,
                                 truncation=True, max_length=max_len)["input_ids"])
            labels[j, :max(0, plen - 1)] = -100
        labels[enc["attention_mask"][:, 1:] == 0] = -100
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(labels.shape)
        valid = labels != -100
        for j in range(len(chunk)):
            n = int(valid[j].sum().item())
            if n == 0:
                raise RuntimeError(f"row {start + j} has no completion tokens")
            losses.append(float((token_loss[j] * valid[j]).sum().div(n).cpu()))
            counts.append(n)
        done = min(start + batch_size, len(rows))
        if start == 0 or done == len(rows) or done % (batch_size * 25) == 0:
            print(f"base completion loss {done}/{len(rows)}", flush=True)
    return np.asarray(losses, dtype=np.float64)


def _full_text(row: dict) -> str:
    return (row["instruction"] + "\n" + row["input"] + "\n" + row["output"])


def _chunked_embeddings(rows: list[dict], model_path: str, device: str,
                        batch_size: int) -> tuple[np.ndarray, dict]:
    """Embed the complete prompt-response pair, chunking past MiniLM's limit."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_path, device=device)
    tok = encoder.tokenizer
    max_tokens = max(8, int(getattr(encoder, "max_seq_length", 128)) - 2)
    pieces: list[str] = []
    owners: list[int] = []
    chunk_counts: list[int] = []
    for i, row in enumerate(rows):
        ids = tok.encode(_full_text(row), add_special_tokens=False)
        n_chunks = max(1, math.ceil(len(ids) / max_tokens))
        chunk_counts.append(n_chunks)
        for j in range(n_chunks):
            part = ids[j * max_tokens:(j + 1) * max_tokens]
            pieces.append(tok.decode(part, clean_up_tokenization_spaces=False))
            owners.append(i)
    embeddings = encoder.encode(
        pieces, normalize_embeddings=True, convert_to_numpy=True,
        batch_size=batch_size, show_progress_bar=True,
    ).astype(np.float32)
    out = np.zeros((len(rows), embeddings.shape[1]), dtype=np.float32)
    for i in range(len(rows)):
        take = embeddings[np.asarray(owners) == i]
        vec = take.mean(axis=0)
        norm = np.linalg.norm(vec)
        out[i] = vec / max(norm, 1e-12)
    del encoder
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, {
        "embedding_definition": "instruction_input_output_chunk_mean_normalized",
        "embedding_max_tokens_per_chunk": max_tokens,
        "embedding_chunks_total": len(pieces),
        "embedding_chunks_mean": float(np.mean(chunk_counts)),
        "embedding_chunks_max": int(np.max(chunk_counts)),
    }


def sinkhorn_due(train_emb: np.ndarray, val_emb: np.ndarray,
                 base_loss: np.ndarray, temperature: float = 1.0,
                 epsilon: float = 0.2, blur: float = 0.1,
                 device: str = "cuda") -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute per-sample D and U from DUCL Sinkhorn potentials."""
    if len(train_emb) != len(base_loss):
        raise ValueError("embedding/loss length mismatch")
    if not temperature > 0 or not 0 < epsilon < 1:
        raise ValueError("temperature must be >0 and epsilon in (0,1)")
    from geomloss import SamplesLoss

    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available()
                       else "cpu")
    data = torch.as_tensor(train_emb, dtype=torch.float32, device=dev)
    target = torch.as_tensor(val_emb, dtype=torch.float32, device=dev)
    n = len(data)
    source_w = torch.softmax(torch.as_tensor(-base_loss / temperature,
                                             dtype=torch.float32, device=dev), dim=0)
    combined = torch.cat((data, data), dim=0)
    weights = torch.cat(((1.0 - epsilon) * source_w,
                         torch.full((n,), epsilon / n, device=dev)))
    source_target_w = source_w
    target_w = torch.full((len(target),), 1.0 / len(target), device=dev)
    ot = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=False,
                     verbose=False, potentials=True)
    with torch.no_grad():
        source_potential, _ = ot(weights, combined, source_target_w, data)
        target_potential, _ = ot(weights, combined, target_w, target)
    d = np.asarray(source_potential.detach().cpu(), dtype=np.float64).reshape(-1)[n:]
    u = -np.asarray(target_potential.detach().cpu(), dtype=np.float64).reshape(-1)[n:]
    d += max(0.0, -float(d.min())) + 1e-8
    u += max(0.0, -float(u.min())) + 1e-3
    due = d / u
    report = {
        "due_definition": "official_ducl_sinkhorn_potentials",
        "mu_p_definition": "train_embeddings_weighted_by_softmax_negative_base_completion_nll",
        "target_definition": "uniform_same_domain_validation_embeddings",
        "temperature": temperature, "epsilon": epsilon, "blur": blur,
        "difficulty_mean": float(d.mean()), "difficulty_std": float(d.std()),
        "utility_mean": float(u.mean()), "utility_std": float(u.std()),
        "due_mean": float(due.mean()), "due_std": float(due.std()),
        "mu_p_effective_sample_size": float(1.0 / (source_w.detach().cpu().numpy() ** 2).sum()),
    }
    return d, u, report


def main():
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
    ap.add_argument("--embed-chunks", type=int, default=0,
                    help="reserved for compatibility; all chunks are retained")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--blur", type=float, default=0.1)
    args = ap.parse_args()
    train, val = load_jsonl(args.train), load_jsonl(args.val)
    model, tok = _load_lm(args.base)
    base_loss = base_completion_losses(train, model, tok, args.score_batch_size, args.max_len)
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_emb, emb_report = _chunked_embeddings(train, args.embed_model,
                                                args.embed_device, args.embed_batch_size)
    val_emb, _ = _chunked_embeddings(val, args.embed_model,
                                      args.embed_device, args.embed_batch_size)
    difficulty, utility, due_report = sinkhorn_due(
        train_emb, val_emb, base_loss, args.temperature, args.epsilon, args.blur,
        args.ot_device)
    due = difficulty / utility
    order = window_order_from_due(due, args.batch_size, args.alpha, args.seed)
    if sorted(order) != list(range(len(train))):
        raise AssertionError("DUCL order is not a complete permutation")
    position = {idx: pos for pos, idx in enumerate(order)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as sf:
        for i, row in enumerate(train):
            item = dict(row)
            item.update({"ot_record_id": record_id(row), "ot_base_loss": float(base_loss[i]),
                         "ot_difficulty": float(difficulty[i]), "ot_utility": float(utility[i]),
                         "ot_due": float(due[i]), "ot_order_position": int(position[i])})
            scored.append(item)
            sf.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "ordered.jsonl").open("w", encoding="utf-8") as of:
        for i in order:
            of.write(json.dumps(scored[i], ensure_ascii=False) + "\n")
    report = {"score_version": "bt_ducl_v3_official_sinkhorn", "n_train": len(train),
              "n_val": len(val), "order_is_permutation": True, "batch_size": args.batch_size,
              "alpha": args.alpha, "window": "quantile_10_to_100_percent",
              **emb_report, **due_report,
              "base_loss_mean": float(base_loss.mean()),
              "base_loss_std": float(base_loss.std())}
    (out_dir / "score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
