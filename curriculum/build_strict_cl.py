"""Strict online capability-incremental CL builder (protocol-fixed).

Two protocol fixes vs the offline builder:

1. ONLINE BUFFER UPDATES. Stage 3 must not re-select replay samples from the
   full historical data D1 u D2. It may only use the buffer actually retained:
       M1 = select(D1, B1)
       stage2 trains on D2 u M1
       M2 = select(M1 u D2, B2)
       stage3 trains on D3 u M2
   The offline builder violated this (pool3 = stages[1]+stages[2]).

2. STAGED SKILL-LIBRARY PROMPTS. Records carry the instruction of their
   NATIVE stage (the skill library available when the data arrived):
       L1 = basic nodes only
       L2 = L1 + coordination (SignalReady/WaitReady, Handover*, Co*)
       L3 = L2 + recovery (Fallback, ClearPath, Recharge)
   A buffered sample keeps its original (x, y) pair including the instruction
   it was learned with. Native prompts are used to measure forgetting; the full
   L3 prompt is used separately to measure final deployment performance.

Selection modes (--selection):
   bottommix : 50% lowest E_DUE2_OTD + 50% random from remainder (E3a-bottom)
   reservoir : uniform random (reservoir sampling baseline)

--repackage (prompt-aligned replay, PAR): the buffer CONTENT is unchanged
(same (x, y) pairs are retained), but when a buffered sample re-enters
training at a later stage its instruction is re-wrapped with that stage's
CURRENT skill library (stage2 file -> L2, stage3 file -> L3). This teaches
the model to execute old simple tasks under the expanded library, targeting
the interface-interference failure mode (REPORT.md §13.4/§15). Buffer
provenance assertions still apply to the retained pairs.

Also emits noreplay files and a compute-matched naive variant (current-stage
upsampling to match the replay version's optimizer steps, no history access).

Usage:
  python -m curriculum.build_strict_cl --selection bottommix --out outputs/curriculum_strict
  python -m curriculum.build_strict_cl --selection reservoir --out outputs/curriculum_strict_rsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from curriculum.build_curriculum import stage_of, window_quantile  # noqa: E402

SCORED = "outputs/curriculum_edue2/train_edue2_scored.jsonl"
KEY = "E_DUE2_OTD"
BATCH, ALPHA, SEED = 16, 0.8, 42
B1, B2 = 270, 484  # budgets, identical to the offline diagnostic version

_ORIG = ("You are a helpful assistant that creates behavior trees for multi-robot teams.\n"
         "Your task:\n"
         "- Convert the provided natural-language mission description into an XML-formatted multi-agent behavior tree.\n"
         "- The behavior tree must be compatible with the BehaviorTree.CPP library (<root BTCPP_format=\"4\">), with one <BehaviorTree> per robot coordinated by a main tree.\n"
         "{skills}\n\n"
         "Output Requirements:\n"
         "- Output only the XML representation of the behavior tree. Do not include explanations, comments, or any additional text.\n"
         "- Every robot mentioned in the mission must have its own <BehaviorTree ID=\"Agent_<name>\">.")

L1 = _ORIG.format(skills=(
    "- Available skill library (L1): Sequence, Parallel, MoveTo, PickUp, PlaceDown, "
    "and condition nodes (e.g. IsAtLocation, IsItemAt)."))

L2 = _ORIG.format(skills=(
    "- Available skill library (L2): Sequence, Parallel, MoveTo, PickUp, PlaceDown, "
    "condition nodes, and inter-robot coordination nodes: SignalReady, WaitReady, "
    "HandoverGive, HandoverTake, CoPickUp, CoMoveTo, CoPlaceDown."))

L3 = _ORIG.format(skills=(
    "- Available skill library (L3): L2 nodes plus recovery nodes: Fallback, ClearPath, Recharge.\n"
    "- Use Fallback nodes to recover from the failures mentioned in the mission."))

INSTR = {1: L1, 2: L2, 3: L3}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} -> {path}")


def sample_id(record: dict) -> str:
    """Stable content identity for buffer provenance across files/processes."""
    blob = json.dumps({
        "input": record["input"],
        "output": record["output"],
        "meta": record["meta"],
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ids_digest(records: list) -> str:
    ids = "\\n".join(sorted(r["sample_id"] for r in records))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def select(pool: list, budget: int, mode: str, rng: random.Random) -> list:
    if budget >= len(pool):
        return list(pool)
    if mode == "reservoir":
        return rng.sample(pool, budget)
    # bottommix: 50% lowest score + 50% random from remainder
    srt = sorted(pool, key=lambda r: r[KEY])
    half = budget // 2
    return srt[:half] + rng.sample(srt[half:], budget - half)


def with_instr(records, stage):
    out = []
    for r in records:
        r = dict(r)
        r["instruction"] = INSTR[stage]
        r["sample_id"] = sample_id(r)
        out.append(r)
    return out


def build(selection, out_dir, repackage=False):
    rng = random.Random(SEED)
    records = load_jsonl(SCORED)
    stages = {1: [], 2: [], 3: []}
    for r in records:
        stages[stage_of(r)].append(r)
    D = {s: with_instr(stages[s], s) for s in (1, 2, 3)}
    print(f"stage sizes: {{k: len(v) for k, v in D.items()}}")

    out = Path(out_dir)
    meta = {"selection": selection, "budgets": {"M1": B1, "M2": B2},
            "repackage": repackage,
            "protocol": "online buffer: M1=select(D1,B1); "
                        "stage2=D2+M1; M2=select(M1+D2,B2); stage3=D3+M2"}

    # ---- stage 1: no history ----
    s1 = window_quantile(D[1], BATCH, ALPHA, SEED + 1, key=KEY)
    write_jsonl(out / "stage1.jsonl", s1)

    # ---- online buffer after stage 1 ----
    M1 = select(D[1], B1, selection, rng)
    assert all(r["instruction"] == L1 for r in M1)
    meta["M1_comp"] = dict(Counter(stage_of(r) for r in M1))

    # ---- stage 2: D2 + M1 ----
    s2 = window_quantile(D[2] + M1, BATCH, ALPHA, SEED + 2, key=KEY)
    if repackage:
        # PAR: re-wrap every sample (incl. buffered L1 ones) with the L2 library
        s2 = with_instr(s2, 2)
    write_jsonl(out / "stage2.jsonl", s2)

    # ---- online buffer update: M2 from (M1 u D2) ONLY ----
    M2 = select(M1 + D[2], B2, selection, rng)
    meta["M2_comp"] = dict(Counter(stage_of(r) for r in M2))
    meta["M2_C1_from_M1"] = sum(1 for r in M2 if stage_of(r) == 1)
    # Protocol guarantee: every C1 sample in M2 must already be in M1.
    # Stable content IDs make the check auditable after JSON serialization.
    m1_ids = {r["sample_id"] for r in M1}
    leaked = [r for r in M2
              if stage_of(r) == 1 and r["sample_id"] not in m1_ids]
    assert not leaked, f"protocol violation: {len(leaked)} C1 samples not in M1"
    meta["protocol_check"] = "PASS: all C1 in M2 were retained in M1"
    meta["buffer_provenance"] = {
        "M1_count": len(M1),
        "M2_count": len(M2),
        "M1_ids_sha256": ids_digest(M1),
        "M2_ids_sha256": ids_digest(M2),
        "M2_C1_ids_sha256": ids_digest(
            [r for r in M2 if stage_of(r) == 1]),
        "M2_C1_subset_M1": True,
    }

    # ---- stage 3: D3 + M2 ----
    s3 = window_quantile(D[3] + M2, BATCH, ALPHA, SEED + 3, key=KEY)
    if repackage:
        # PAR: re-wrap every sample (incl. buffered L1/L2 ones) with L3 library
        s3 = with_instr(s3, 3)
    write_jsonl(out / "stage3.jsonl", s3)

    # ---- noreplay variants (staged prompts, no history access) ----
    write_jsonl(out / "stage1_noreplay.jsonl",
                window_quantile(D[1], BATCH, ALPHA, SEED + 1, key=KEY))
    write_jsonl(out / "stage2_noreplay.jsonl",
                window_quantile(D[2], BATCH, ALPHA, SEED + 2, key=KEY))
    write_jsonl(out / "stage3_noreplay.jsonl",
                window_quantile(D[3], BATCH, ALPHA, SEED + 3, key=KEY))

    # ---- compute-matched naive: upsample CURRENT stage to replay step counts
    # stage2 replay file has len(D2)+B1 samples; stage3 has len(D3)+B2.
    cm2 = D[2] + rng.sample(D[2], B1)
    cm3 = D[3] + rng.sample(D[3], B2)
    write_jsonl(out / "stage2_naivecm.jsonl",
                window_quantile(cm2, BATCH, ALPHA, SEED + 2, key=KEY))
    write_jsonl(out / "stage3_naivecm.jsonl",
                window_quantile(cm3, BATCH, ALPHA, SEED + 3, key=KEY))
    meta["naivecm"] = {"stage2_extra": B1, "stage3_extra": B2,
                       "note": "current-stage upsampling, matches replay "
                               "optimizer steps without any history access"}

    (out / "strict_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["bottommix", "reservoir"],
                    default="bottommix")
    ap.add_argument("--repackage", action="store_true",
                    help="prompt-aligned replay: re-wrap buffered samples with "
                         "the current stage's skill library at training time")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.selection, args.out, repackage=args.repackage)


if __name__ == "__main__":
    main()
