"""Evaluate a model (fine-tuned 1B / base / DeepSeek API) on a dataset split.

Headline metric: symbolic execution success rate — the generated XML is
executed by datagen.executor against the task's world model; success means
the mission goal is reached (with faults active).

Usage:
  python -m eval.evaluate --data outputs/dataset/test.jsonl \
      --adapter outputs/checkpoints/ours/stage3 --out outputs/results/ours_test.json
  python -m eval.evaluate --data outputs/dataset/test.jsonl \
      --deepseek deepseek-v4-pro --out outputs/results/dsv4pro_test.json
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from datagen.executor import execute, re_extract
from datagen.strips.domains import build_domain

BASE = os.environ.get("CLIMB_BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stable_record_id(record: dict) -> str:
    """Identity shared by evaluation writers and strict paired consumers."""
    identity = json.dumps({
        "instruction": record["instruction"], "input": record["input"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def reconstruct_task(meta: dict) -> dict:
    static = [("can_reach", r, l) for r, l in meta["can_reach"]]
    edges = [("connected", a, b) for a, b in meta["connected"]]
    domain = build_domain(meta["domain"], robots=meta["robots"],
                          items=meta["items"], extra_static=static, edges=edges)
    return {
        "domain": meta["domain"], "domain_obj": domain,
        "init_dynamic": [tuple(f) for f in meta["init_dynamic"]],
        "goal": [tuple(f) for f in meta["goal"]],
        "faults": meta["faults"],
        "skill_aliases": meta.get("skill_aliases", {}),
    }


# ------------------------------------------------------------ generators ---

def gen_local(records: list, adapter: str = None, base: str = BASE,
              max_new: int = 1400, batch_size: int = 64) -> list:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outs = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        prompts = []
        for r in chunk:
            msgs = [{"role": "system", "content": r["instruction"]},
                    {"role": "user", "content": r["input"]}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2560).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j, g in enumerate(gen):
            outs.append(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(records))}/{len(records)}", flush=True)
    return outs


def gen_deepseek(records: list, model: str, max_workers: int = 12) -> list:
    from common.llm import chat_batch
    jobs = [[{"role": "system", "content": r["instruction"]},
             {"role": "user", "content": r["input"]}] for r in records]
    return chat_batch(jobs, max_workers=max_workers, model=model, temperature=0.0)


# ---------------------------------------------------------------- metrics ---

def score(records: list, generations: list) -> dict:
    if len(records) != len(generations):
        raise ValueError(
            f"generation count {len(generations)} != record count {len(records)}")
    per_domain = defaultdict(Counter)
    per_scenario = defaultdict(Counter)
    per_primitive = defaultdict(Counter)  # shared vs new_primitive (test-domain unique actions)
    reasons = Counter()
    n_xml_ok = 0
    details = []
    for r, g in zip(records, generations):
        meta = r["meta"]
        try:
            re_extract(g)
            xml_ok = True
        except Exception:  # noqa: BLE001
            xml_ok = False
        n_xml_ok += xml_ok
        if xml_ok:
            res = execute(reconstruct_task(meta), g)
        else:
            res = {"success": False, "reason": "no_xml_block"}
        ok = res["success"]
        reasons[res["reason"]] += 0 if ok else 1
        per_domain[meta["domain"]]["n"] += 1
        per_domain[meta["domain"]]["ok"] += ok
        per_scenario[meta["scenario"]]["n"] += 1
        per_scenario[meta["scenario"]]["ok"] += ok
        bucket = "new_primitive" if "+service" in meta["scenario"] else "shared_primitive"
        per_primitive[bucket]["n"] += 1
        per_primitive[bucket]["ok"] += ok
        details.append({"record_id": stable_record_id(r),
                        "domain": meta["domain"], "scenario": meta["scenario"],
                        "scenario": meta["scenario"],
                        "success": ok, "reason": res["reason"],
                        "recoveries": res.get("recoveries", 0)})
    n = len(records)
    return {
        "n": n,
        "exec_success_rate": sum(d["success"] for d in details) / max(n, 1),
        "xml_wellformed_rate": n_xml_ok / max(n, 1),
        "per_domain": {k: dict(v) for k, v in per_domain.items()},
        "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
        "per_primitive": {k: dict(v) for k, v in per_primitive.items()},
        "fail_reasons": dict(reasons),
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=BASE,
                    help="local base model the adapter was trained on")
    ap.add_argument("--deepseek", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-generations", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.data)
    if args.limit:
        records = records[:args.limit]

    if args.deepseek:
        gens = gen_deepseek(records, args.deepseek)
    else:
        gens = gen_local(records, adapter=args.adapter, base=args.base,
                         batch_size=args.batch_size)

    result = score(records, gens)
    result["model"] = args.deepseek or args.adapter or "base"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.save_generations:
        result["generations"] = gens
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("details", "generations")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
