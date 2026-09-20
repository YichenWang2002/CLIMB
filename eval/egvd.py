"""Executor-Guided Verified Decoding (EGVD) evaluation.

For each task: sample k candidates (do_sample=True, independent draws),
parse + execute each against a freshly reconstructed world (full reset per
candidate), accept the first that executes AND satisfies the goal.
Reports pass@k / selected@k (= first-success), executor-call stats,
candidate diversity, per tier/domain/primitive breakdowns.
"""
import argparse, json, os, time
from collections import Counter, defaultdict

import torch
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict as _dd
from eval.eval_constrained import ConstrainedState, _Proc, PlatformMap
import sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from eval.evaluate import (load_jsonl, reconstruct_task, BASE)
from experiments.bt_ducl.common import record_id, apply_chat_template_compat
from datagen.executor import execute, re_extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base", default=BASE,
                    help="base causal LM identifier or local snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-generations", action="store_true")
    ap.add_argument("--constrained", action="store_true",
                    help="map-grounded constrained decoding (platform map vocab)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint jsonl for incremental save / resume (default: <out>.ckpt.jsonl)")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    records = load_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]
    torch.manual_seed(args.seed)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    parsed = None
    if args.constrained:
        parsed = {}
        for r in records:
            pm = PlatformMap.from_meta(r["meta"])
            edges = _dd(set)
            for a_, b_ in pm["edges"]:
                edges[a_].add(b_)  # keep original directionality semantics
            locs = {x for e in pm["edges"] for x in e} | set(pm["charge_stations"])
            parsed[record_id(r)] = (set(pm["robots"]), set(pm["items"]), locs, dict(edges))

    k = args.k
    ckpt_path = args.ckpt or (args.out + ".ckpt.jsonl")
    details, done_ids = [], set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as _f:
            for _ln in _f:
                _ln = _ln.strip()
                if _ln:
                    _d = json.loads(_ln)
                    details.append(_d)
                    done_ids.add(_d["record_id"])
        if done_ids:
            print(f"[resume] {len(done_ids)} records loaded from {ckpt_path}", flush=True)
    todo = [r for r in records if record_id(r) not in done_ids]
    _ck = open(ckpt_path, "a")
    t0 = time.time()
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        prompts = [apply_chat_template_compat(tok,
            [{"role": "system", "content": r["instruction"]},
             {"role": "user", "content": r["input"]}],
            add_generation_prompt=True) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2560).to(model.device)
        plen = enc["input_ids"].shape[1]
        lp = None
        if args.constrained:
            states = [ConstrainedState(tok, *parsed[record_id(chunk[j])])
                      for j in range(len(chunk)) for _ in range(k)]
            lp = [_Proc(states, plen)]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=True, temperature=args.temperature,
                                 top_p=args.top_p, num_return_sequences=k,
                                 pad_token_id=tok.eos_token_id,
                                 logits_processor=lp)
        for j, r in enumerate(chunk):
            cands = [tok.decode(g[plen:], skip_special_tokens=True)
                     for g in gen[j * k:(j + 1) * k]]
            meta = r["meta"]
            results, n_xml_ok = [], 0
            for cand in cands:
                try:
                    re_extract(cand)
                    xml_ok = True
                except Exception:
                    xml_ok = False
                n_xml_ok += xml_ok
                if xml_ok:
                    res = execute(reconstruct_task(meta), cand)  # fresh world per candidate
                else:
                    res = {"success": False, "reason": "no_xml_block"}
                results.append(res)
            succ = [bool(x["success"]) for x in results]
            first = succ.index(True) if any(succ) else None
            n_unique = len(set(c.strip() for c in cands))
            details.append({
                "record_id": record_id(r), "domain": meta["domain"],
                "tier": meta["tier"], "scenario": meta["scenario"],
                "succ_at": succ,                       # per-candidate outcomes
                "first_success_idx": first,            # None if all fail
                "executor_calls": (first + 1) if first is not None else k,
                "n_unique_candidates": n_unique,
                "n_xml_ok": n_xml_ok,
                "gen_tokens": [int((g[plen:] != tok.eos_token_id).sum()) for g in gen[j*k:(j+1)*k]],
                "candidates": cands if args.save_generations else None,
            })
        for _d in details[-len(chunk):]:
            _ck.write(json.dumps(_d, ensure_ascii=False) + "\n")
        _ck.flush()
        print(f"  {len(details)}/{len(records)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    ks = [x for x in (1, 2, 4, 8) if x <= k]
    def agg(sub):
        out = {"n": len(sub)}
        for kk in ks:
            out[f"pass@{kk}"] = sum(any(d["succ_at"][:kk]) for d in sub) / max(len(sub), 1)
        return out
    report = {
        "n": len(records), "k": k, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed, "adapter": args.adapter,
        "base": args.base,
        "wall_time_s": round(time.time() - t0, 1),
        "overall": agg(details),
        "avg_executor_calls_selected": sum(d["executor_calls"] for d in details) / len(details),
        "avg_unique_candidates": sum(d["n_unique_candidates"] for d in details) / len(details),
        "candidate_dedup_rate": 1 - sum(d["n_unique_candidates"] for d in details) / (len(details) * k),
        "xml_wellformed_rate": sum(d["n_xml_ok"] for d in details) / (len(details) * k),
        "avg_gen_tokens": sum(sum(d["gen_tokens"]) for d in details) / (len(details) * k),
        "per_tier": {t: agg([d for d in details if d["tier"] == t]) for t in ("T1", "T2", "T3")},
        "per_domain": {dom: agg([d for d in details if d["domain"] == dom])
                       for dom in sorted({d["domain"] for d in details})},
        "per_primitive": {b: agg([d for d in details
                                  if ("+service" in d["scenario"]) == (b == "new_primitive")])
                          for b in ("shared_primitive", "new_primitive")},
        "note": "selected@k == pass@k because verifier is the final success criterion; "
                "early-stop executor calls reported separately",
    }
    if not args.save_generations:
        for d in details:
            d.pop("candidates", None)
    _ck.close()
    report["details"] = details
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({kk: report["overall"][f"pass@{kk}"] for kk in ks}, indent=1))
    print("written ->", args.out)


if __name__ == "__main__":
    main()
