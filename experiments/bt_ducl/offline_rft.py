"""Offline single-round verified self-training (STaR-style) for BT generation.

Stage ``generate``: sample K rollouts per prompt from the frozen warm-start
actor, verify each with the strict executor, and keep up to
``--max-per-prompt`` deduplicated verified completions per prompt.  Data is
collected ONCE from a FIXED model, so the training target never moves — the
online sample-train-resample loop that destabilized three previous attempts
does not exist here.

Stage ``train``: continue plain full-completion SFT from the warm start on a
static pool.  The method arm trains on gold ∪ verified; the matched control
trains on gold only, with identical optimizer steps, batch shape, LR
schedule, and seed.  The only difference is whether the verified
completions are in the pool.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .common import load_jsonl, record_id
from .exec_ac_sft import _load_actor, _sample_rollouts, _sft_update

BASE_DEFAULT = "models/llama32-1b"


def generate(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.train)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = _load_actor(args.base, args.warm_start, torch.bfloat16,
                                   args.load_mode)
    model.config.use_cache = False
    torch.manual_seed(args.seed)
    rollout_rows = [row for row in rows for _ in range(args.rollouts)]
    started = time.perf_counter()
    batch = _sample_rollouts(model, tokenizer, rollout_rows, args.max_new,
                             args.temperature, args.top_p,
                             logprob_batch=args.logprob_batch,
                             generation_batch=args.generation_batch,
                             compute_logprob=False)
    rng = random.Random(args.seed + 77)
    kept, n_prompts_with_verified = [], 0
    for draw, row in enumerate(rows):
        start = draw * args.rollouts
        texts = batch.texts[start:start + args.rollouts]
        rewards = batch.rewards[start:start + args.rollouts]
        verified = sorted({text for text, reward in zip(texts, rewards)
                           if reward == 1.0})
        if not verified:
            continue
        n_prompts_with_verified += 1
        take = min(args.max_per_prompt, len(verified))
        for text in rng.sample(verified, take):
            kept.append({"instruction": row["instruction"],
                         "input": row["input"], "output": text,
                         "meta": row["meta"], "source": "verified",
                         "parent_record_id": record_id(row)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "prompts": len(rows), "rollouts_per_prompt": args.rollouts,
        "rollout_success": sum(batch.rewards),
        "prompts_with_verified": n_prompts_with_verified,
        "verified_rows_written": len(kept),
        "generation_seconds": time.perf_counter() - started,
        "seed": args.seed, "warm_start": args.warm_start,
    }
    (out.parent / (out.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def train(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.train)
    n_gold = len(rows)
    if args.verified:
        rows = rows + load_jsonl(args.verified)
    rng = random.Random(args.seed + 9137)
    model, tokenizer = _load_actor(args.base, args.warm_start, torch.bfloat16,
                                   args.load_mode)
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    warmup_updates = max(1, int(args.updates * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_updates))
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for step in range(1, args.updates + 1):
        batch_rows = [rows[rng.randrange(len(rows))]
                      for _ in range(args.batch)]
        loss, tokens, _ = _sft_update(model, tokenizer, batch_rows,
                                      args.max_len, args.micro_batch,
                                      optimizer, scheduler)
        reports.append({"step": step, "loss": loss,
                        "train_completion_tokens": tokens})
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(reports[-1]), flush=True)
    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")
    metadata = {
        "protocol": "offline_rft_v1", "seed": args.seed,
        "warm_start": args.warm_start, "n_gold": n_gold,
        "n_verified": len(rows) - n_gold, "pool_size": len(rows),
        "updates": args.updates, "batch": args.batch,
        "micro_batch": args.micro_batch, "lr": args.lr,
        "max_len": args.max_len, "verified_source": args.verified or None,
        "reports": reports,
    }
    (out_dir / "train_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in
                      ("n_gold", "n_verified", "updates", "batch")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--train", required=True)
    gen.add_argument("--warm-start", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument("--base", default=BASE_DEFAULT)
    gen.add_argument("--load-mode", choices=("bf16", "4bit"), default="4bit")
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--rollouts", type=int, default=4)
    gen.add_argument("--max-per-prompt", type=int, default=2)
    gen.add_argument("--max-new", type=int, default=1400)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--logprob-batch", type=int, default=8)
    gen.add_argument("--generation-batch", type=int, default=64)
    gen.add_argument("--seed", type=int, default=42)
    tr = sub.add_parser("train")
    tr.add_argument("--train", required=True)
    tr.add_argument("--verified", default="")
    tr.add_argument("--warm-start", required=True)
    tr.add_argument("--run-dir", required=True)
    tr.add_argument("--base", default=BASE_DEFAULT)
    tr.add_argument("--load-mode", choices=("bf16", "4bit"), default="4bit")
    tr.add_argument("--updates", type=int, default=100)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--micro-batch", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1e-5)
    tr.add_argument("--warmup-ratio", type=float, default=0.1)
    tr.add_argument("--max-len", type=int, default=2560)
    tr.add_argument("--log-every", type=int, default=5)
    tr.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.command == "generate":
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
