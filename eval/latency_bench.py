#!/usr/bin/env python3
"""Latency benchmark: greedy vs SCD-masked decoding, same adapter/hardware.

Two measurement modes, both real wall-clock on this machine:

  * bs=1  -- single-request latency (deployment semantics; comparable to the
             per-request API latencies). Run on a tier-stratified subset
             (--per-scenario N) to bound wall time; the subset is the first N
             tasks of each scenario family in test.jsonl order (deterministic).
  * bs=B>1 -- batched amortized latency (batch wall-clock / batch size),
             the protocol of the frozen evals (batch_size=64); run on the
             full 600-task split to cross-check the recorded 0.839 s.

Integrity notes:
  * same model load (4-bit NF4, bf16) and adapter for every arm;
  * one warmup chunk per (batch-size, arm) is executed but NOT timed;
  * per-task wall-clock via time.perf_counter around model.generate only;
  * per-task execution success is recorded for every timed task and must
    track the frozen eval rates before any latency number may be cited
    (success verification is exact on the full-600 batched arms).

Usage:
  python -m eval.latency_bench \
      --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
      --data outputs/dataset/test.jsonl \
      --out outputs/results/outputs/latency_bench_scd_vs_greedy.json \
      --configs bs1_subset120,bs64_full600
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

import sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from datagen.executor import execute
from eval.evaluate import reconstruct_task, load_jsonl
from eval.eval_constrained import ConstrainedState, PlatformMap, _Proc
from common.data import record_id, apply_chat_template_compat


def load_model(base, adapter):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def scenario_group(meta):
    return ("relay" if str(meta.get("scenario", "")).startswith("relay")
            else "joint-heavy")


def stratify(recs, per_scenario):
    """First `per_scenario` tasks of each scenario family, in file order."""
    if per_scenario <= 0:
        return list(recs)
    counts = defaultdict(int)
    out = []
    for r in recs:
        g = scenario_group(r["meta"])
        if counts[g] < per_scenario:
            out.append(r)
            counts[g] += 1
    return out


def run_arm(model, tok, recs, arm, batch_size, max_new):
    parsed = []
    for r in recs:
        pm = PlatformMap.from_meta(r["meta"])
        edges = defaultdict(set)
        for a_, b_ in pm["edges"]:
            edges[a_].add(b_)
        locs = {x for e in pm["edges"] for x in e} | set(pm["charge_stations"])
        parsed.append((set(pm["robots"]), set(pm["items"]), locs, dict(edges)))

    prompts, states = [], []
    for i, r in enumerate(recs):
        msgs = [{"role": "system", "content": r["instruction"]},
                {"role": "user", "content": r["input"]}]
        prompts.append(apply_chat_template_compat(tok, msgs,
                                                  add_generation_prompt=True))
        states.append(ConstrainedState(tok, *parsed[i], level="topology")
                      if arm == "scd" else None)

    lat, toks, success = [], [], 0
    n_chunks = 0
    warmup_chunks = 8 if batch_size == 1 else 1
    for s in range(0, len(recs), batch_size):
        chunk = range(s, min(s + batch_size, len(recs)))
        enc = tok([prompts[j] for j in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=2048).to(model.device)
        plen = enc["input_ids"].shape[1]
        active = [states[j] for j in chunk if states[j] is not None]
        kw = {"logits_processor": [_Proc(active, plen)]} if active else {}
        t0 = time.perf_counter()
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id, **kw)
        dt = time.perf_counter() - t0
        n_chunks += 1
        if n_chunks <= warmup_chunks:      # warmup chunks, not timed
            continue
        m = len(list(chunk))
        lat.extend([dt / m] * m)
        for j in chunk:
            text = tok.decode(gen[j - s][plen:], skip_special_tokens=True)
            toks.append(int((gen[j - s][plen:] != tok.pad_token_id).sum()))
            try:
                res = execute(reconstruct_task(recs[j]["meta"]), text)
                success += res["success"]
            except Exception:
                pass
        done = len(lat)
        if done % (batch_size * 4) < batch_size or done >= len(recs) - batch_size:
            print(f"  [{arm} bs{batch_size}] {done}/{len(recs)} "
                  f"acc {success}/{done} amort_lat "
                  f"{sum(lat)/len(lat):.3f}s", flush=True)

    n = len(lat)
    return {
        "arm": arm, "batch_size": batch_size, "n_timed": n,
        "latency_amortized_mean_s": sum(lat) / n if batch_size > 1 else None,
        "latency_mean_s": sum(lat) / n,
        "latency_median_s": sorted(lat)[n // 2],
        "latency_p90_s": sorted(lat)[int(n * 0.9)],
        "tokens_mean": sum(toks) / n,
        "success_rate": success / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="models/llama32-1b")
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--configs", default="bs1_subset120,bs64_full600",
                    help="comma list of <bs=1|64>_subset<N>|full600")
    a = ap.parse_args()

    all_recs = load_jsonl(a.data)
    model, tok = load_model(a.base, a.adapter)

    smi = __import__("subprocess").run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True)

    out = {"base": a.base, "adapter": a.adapter, "data": a.data,
           "max_new": a.max_new, "quant": "nf4-bf16",
           "nvidia_smi_at_start": smi.stdout.strip(),
           "runs": {}}
    for cfg in a.configs.split(","):
        bs_s, scope = cfg.split("_")
        bs = int(bs_s.replace("bs", ""))
        recs = (stratify(all_recs, int(scope.replace("subset", "")))
                if scope.startswith("subset") else list(all_recs))
        for arm in ("greedy", "scd"):
            print(f"=== {cfg} arm {arm} "
                  f"({len(recs)} tasks, bs={bs}) ===", flush=True)
            ConstrainedState._GLOBAL_TOKEN_CACHE.clear()
            key = f"{cfg}/{arm}"
            out["runs"][key] = run_arm(model, tok, recs, arm, bs, a.max_new)
            out["runs"][key]["n_tasks_in_scope"] = len(recs)
            r = out["runs"][key]
            print(f"{key}: n={r['n_timed']} lat_mean={r['latency_mean_s']:.3f}s "
                  f"median={r['latency_median_s']:.3f}s "
                  f"tokens={r['tokens_mean']:.0f} "
                  f"success={r['success_rate']:.4f}", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))

    print("wrote", a.out)


if __name__ == "__main__":
    main()
