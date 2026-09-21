#!/usr/bin/env python3
"""CPU-only smoke tests for the CLIMB release -- no GPU, no model downloads.

Covers the three core mechanisms without any training:
  1. corpus pipeline: STRIPS sample -> BT compile -> symbolic execution
  2. SPCL scoring math: midranks, PCA fusion, structural difficulty, buckets
  3. SCD: platform-map whitelisting + attribute-value masking
  4. stats: exact McNemar paired test

Usage:  python tests/test_core_smoke.py     (from anywhere)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, []


def check(name, cond, info=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL.append(name)
        print(f" FAIL {name} {info}")


print("[1] datagen: STRIPS sample -> BT compile -> executor")
from datagen.strips.sampler import Sampler
from datagen.strips.compiler import BTCompiler
from datagen.executor import execute

task = None
for seed in range(20):
    task = Sampler(seed=seed).sample("warehouse", "T3")
    if task is not None:
        break
check("sampler produced a faulted warehouse task", task is not None)
xml = BTCompiler(task).compile()
check("BT compiler emitted BTCPP v4 XML", 'BTCPP_format="4"' in xml)
res = execute(task, xml)
check("gold XML executes to success", res.get("success") is True, str(res))

print("[1b] benchmark composition (paper Table 1 protocol)")
relay = heavy = faulted = 0
test_n = 0
tier_leaks = 0
for line in (ROOT / "data/test.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    m = json.loads(line)["meta"]
    test_n += 1
    relay += str(m["scenario"]).startswith("relay")
    heavy += str(m["scenario"]).startswith("heavy")
    faulted += len(m.get("faults", [])) > 0
    tier_leaks += "tier" in m
check("test split is the 480-task multi-agent suite", test_n == 480, f"n={test_n}")
check("320 relay-transport missions", relay == 320, str(relay))
check("160 joint-heavy-transport missions", heavy == 160, str(heavy))
check("357 faulted missions", faulted == 357, str(faulted))
check("no tier labels in any split",
      tier_leaks == 0 and all(
          "tier" not in json.loads((ROOT / f"data/{s}.jsonl").read_text().splitlines()[0])["meta"]
          for s in ("train", "val")))

print("[2] curriculum: SPCL scoring math + structural difficulty")
import numpy as np
from curriculum.score_mt_ducl import midrank01, adaptive_fuse
from curriculum.structural import structural_features, structural_stage
from curriculum.build_spcl import struct_difficulty

r = midrank01(np.array([3.0, 1.0, 2.0]))
check("midrank01 is a stable percentile rank",
      np.allclose(r, [(2 + .5) / 3, (0 + .5) / 3, (1 + .5) / 3]))
fused, w = adaptive_fuse(np.column_stack([np.linspace(0, 1, 50),
                                          np.linspace(0, 1, 50) + 1.0]), mode="pca")
check("adaptive_fuse handles collinear columns",
      np.allclose(w, [0.5, 0.5]) and np.allclose(fused, np.linspace(.5, 1.5, 50)))

feat = structural_features(xml)
check("structural features extracted",
      all(k in feat for k in ("n_sync", "n_co", "n_fb", "depth", "n_nodes"))
      and feat["n_nodes"] > 0 and feat["n_fb"] > 0, str(feat))
check("faulted task lands in the recovery stage", structural_stage(feat) == 3)

train = [json.loads(l) for l in (ROOT / "data/train.jsonl").read_text().splitlines()[:60]]
d, report = struct_difficulty(train)
check("structural difficulty scores 60 records",
      len(d) == 60 and np.isfinite(d).all() and report["struct_parse_failures"] == 0)

print("[3] eval: SCD platform map + masking, failure stats")
from eval.eval_constrained import PlatformMap, ConstrainedState
from eval.paired_test import compare, exact_mcnemar

meta = train[0]["meta"]
pm = PlatformMap.from_meta(meta)
check("PlatformMap whitelist extraction works",
      set(pm["robots"]) <= {"alpha", "beta", "gamma", "delta", "epsilon"})
try:
    pm["goal"]
    check("PlatformMap blocks forbidden fields", False)
except AssertionError:
    check("PlatformMap blocks forbidden fields", True)

tok = None
if not os.environ.get("CLIMB_SMOKE_SKIP_TOKENIZER"):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.environ.get("CLIMB_BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    except Exception as exc:  # noqa: BLE001 -- offline box / hub unreachable
        print(f"  (skip SCD masking check: tokenizer unavailable: {exc})")
else:
    print("  (skip SCD masking check: CLIMB_SMOKE_SKIP_TOKENIZER set)")
if tok is not None:
    edges = {a: {b} for a, b in meta["connected"][:1]}
    for a, b in meta["connected"][:1]:
        edges.setdefault(b, set()).add(a)
    locs = {x for s in edges.values() for x in s} | set(meta.get("charge_stations", []))
    state = ConstrainedState(tok, set(meta["robots"]), set(meta["items"]), locs,
                             {k: set(v) for k, v in edges.items()}, level="topology")
    ids = tok(' robot="alp', add_special_tokens=False)["input_ids"]
    masked = state.mask(ids)
    check("ConstrainedState.mask returns a token allowlist or None",
          masked is None or (isinstance(masked, list) and len(masked) > 0))

a = {"details": [{"success": True, "tier": "T2"}, {"success": False, "tier": "T2"}]}
b = {"details": [{"success": True, "tier": "T2"}, {"success": True, "tier": "T2"}]}
with tempfile.TemporaryDirectory() as td:
    fa, fb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
    Path(fa).write_text(json.dumps(a))
    Path(fb).write_text(json.dumps(b))
    out = compare(fa, fb)
check("exact McNemar on a 2x2 toy table",
      out["A_success"] == 1 and out["B_success"] == 2
      and abs(out["overall"]["mcnemar_p_exact"] - 1.0) < 1e-9)
check("mcnemar edge cases", exact_mcnemar(0, 0) == 1.0
      and abs(exact_mcnemar(0, 5) - 2 * (0.5 ** 5)) < 1e-12)

print(f"\n{PASS} passed, {len(FAIL)} failed" + (f": {FAIL}" if FAIL else ""))
sys.exit(1 if FAIL else 0)
