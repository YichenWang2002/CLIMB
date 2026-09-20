"""Build the full dataset: sample tasks -> compile BT XML -> executor-validate
-> generate NL input via DeepSeek -> write jsonl splits.

Sampling/validation is CPU-bound and runs in parallel processes.

Usage:
  python -m datagen.build_dataset --dry-run
  python -m datagen.build_dataset --full --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .executor import execute
from .nl_gen import INSTRUCTION, generate_nl_inputs
from .strips.compiler import BTCompiler
from .strips.sampler import Sampler

OUT = Path(__file__).resolve().parents[1] / "data"

TRAIN_DOMAINS = ["warehouse", "hospital", "search_rescue", "office"]
TEST_DOMAINS = ["library", "greenhouse"]
TIER_MIX_TRAIN = {"T1": 0.30, "T2": 0.50, "T3": 0.20}
TIER_MIX_TEST = {"T1": 0.20, "T2": 0.40, "T3": 0.40}


def task_key(t: dict) -> str:
    blob = json.dumps({
        "d": t["domain"], "i": t["init_dynamic"], "g": t["goal"],
        "f": t["faults"], "r": t["robots"],
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def serialize_task(task: dict, xml: str) -> dict:
    """Picklable, executor-reconstructable task record (no domain_obj)."""
    connected = sorted([[f[1], f[2]] for f in task["domain_obj"].static_facts
                        if f[0] == "connected"])
    stations = sorted(f[1] for f in task["domain_obj"].static_facts
                      if f[0] == "charge_station_at")
    return {
        "domain": task["domain"], "tier": task["tier"], "scenario": task["scenario"],
        "robots": task["robots"], "items": task["items"],
        "init_dynamic": [list(f) for f in task["init_dynamic"]],
        "goal": [list(f) for f in task["goal"]],
        "faults": task["faults"],
        "zones": {r: list(z) for r, z in task["zones"].items()},
        "connected": connected, "charge_stations": stations,
        "can_reach": sorted([list(t) for t in task["can_reach"]]),
        "plan": [a.label() for a in task["plan"]],
        "plan_len": len(task["plan"]), "n_agents": len(task["robots"]),
        "_xml": xml,
    }


def worker(domain: str, tier: str, count: int, seed: int) -> list:
    """Runs in a child process. Returns validated serialized task records."""
    sampler = Sampler(seed=seed)
    out, seen = [], set()
    tries = 0
    while len(out) < count and tries < count * 40:
        tries += 1
        task = sampler.sample(domain, tier)
        if task is None:
            continue
        st = serialize_task(task, BTCompiler(task).compile())
        k = task_key(st)
        if k in seen:
            continue
        res = execute(task, st["_xml"])
        if not res["success"]:
            continue
        seen.add(k)
        out.append(st)
    return out


def to_record(st: dict, nl: str) -> dict:
    meta = {k: v for k, v in st.items() if k != "_xml"}
    return {"instruction": INSTRUCTION, "input": nl, "output": st["_xml"], "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--train-per-domain", type=int, default=1500)
    ap.add_argument("--val-per-domain", type=int, default=150)
    ap.add_argument("--test-per-domain", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seed_seq = args.seed

    if args.dry_run:
        quotas = {"train": {d: {"T1": 2, "T2": 3, "T3": 2} for d in TRAIN_DOMAINS},
                  "val": {d: {"T1": 1, "T2": 1, "T3": 1} for d in TRAIN_DOMAINS},
                  "test": {d: {"T1": 1, "T2": 2, "T3": 2} for d in TEST_DOMAINS}}
    else:
        quotas = {
            "train": {d: {t: round(args.train_per_domain * f) for t, f in TIER_MIX_TRAIN.items()}
                      for d in TRAIN_DOMAINS},
            "val": {d: {t: round(args.val_per_domain * f) for t, f in TIER_MIX_TRAIN.items()}
                    for d in TRAIN_DOMAINS},
            "test": {d: {t: round(args.test_per_domain * f) for t, f in TIER_MIX_TEST.items()}
                     for d in TEST_DOMAINS},
        }

    for split, domain_quotas in quotas.items():
        t0 = time.time()
        print(f"=== split {split} ===", flush=True)
        jobs = []
        for d, q in domain_quotas.items():
            for tier, count in q.items():
                seed_seq += 1
                jobs.append((d, tier, count, seed_seq * 1000 + zlib.crc32(f"{d}/{tier}".encode()) % 997))
        all_tasks = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(worker, d, t, c, s): (d, t, c) for d, t, c, s in jobs}
            for fut in as_completed(futs):
                d, t, c = futs[fut]
                got = fut.result()
                print(f"  {d}/{t}: {len(got)}/{c}", flush=True)
                all_tasks += got
        print(f"  sampled+validated {len(all_tasks)} tasks in {time.time()-t0:.0f}s", flush=True)

        nls = generate_nl_inputs(all_tasks, max_workers=args.workers)
        records = [to_record(st, nl) for st, nl in zip(all_tasks, nls) if nl is not None]
        dropped = len(all_tasks) - len(records)
        out = OUT / f"{split}.jsonl"
        with out.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"split {split}: wrote {len(records)} records to {out} "
              f"(dropped {dropped} NL failures, total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
