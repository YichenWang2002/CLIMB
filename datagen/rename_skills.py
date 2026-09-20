"""Primitive-rename augmentation for the training split.

Motivation: the held-out test domains (library/greenhouse) introduce primitives
(CatalogBook/WaterPlants/InspectPlants) the model never saw in training; their
call signature is only described in the NL prose ("Extra skill: ..." line, see
names.UNIQUE_ACTION_EN). Without practice, the meta-skill "read an unseen skill
signature in prose -> ground it into a same-named XML leaf" may score ~0.

This script takes ~frac of the training records, renames ONE or TWO simple
single-robot action leaves (e.g. PickUp -> FetchItem) throughout the output
XML, regenerates the NL input with the renamed skill's signature described in
prose (same "Extra skill:" injection as the test-domain unique actions), and
records meta["skill_aliases"] = {new_name: standard_name} so the symbolic
executor can still validate the renamed tree. With --include-faulted, renamed
skills also reach curriculum stage 3 (faulted samples), keeping the meta-skill
fresh at the end of training. No test-domain primitive is ever used, so
nothing leaks.

Usage:
  python -m datagen.rename_skills --train outputs/dataset/train.jsonl \
      --out outputs/dataset/train_aug10.jsonl --frac 0.10 --seed 123 \
      --workers 16 --include-faulted --double-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import nl_gen
from .executor import BINDING_MAP, JOINT_PAIRS, execute
from .names import RENAME_POOL
from .strips.domains import build_domain

# simple single-robot leaf tags that may be renamed (never coordination or
# control nodes); Recharge/ClearPath only appear in faulty trees, but keep the
# pool complete in case a fault-free tree still contains them.
SIMPLE_ACTIONS = {"MoveTo", "PickUp", "PlaceDown", "Recharge", "ClearPath"}

_RESERVED = (set(BINDING_MAP) | set(JOINT_PAIRS)
             | {"IsAtLocation", "IsItemAt", "IsCarrying", "SignalReady", "WaitReady",
                "Sequence", "Fallback", "Parallel", "SubTree", "BehaviorTree", "root"})
for _std, _cands in RENAME_POOL.items():
    assert _std in SIMPLE_ACTIONS, _std
    for _new, _desc in _cands:
        assert _new not in _RESERVED and re.fullmatch(r"[A-Za-z_][\w.-]*", _new), _new

TAG_RE = re.compile(r"</?([A-Za-z_][\w.-]*)[\s/>]")


def rename_xml(xml: str, old: str, new: str) -> str:
    """Rename every occurrence of a leaf tag (open/close/self-closing).
    Attributes are untouched."""
    return re.sub(rf"<(/?){old}(?=[\s/>])", rf"<\1{new}", xml)


def reconstruct_task(meta: dict) -> dict:
    """Same as eval.evaluate.reconstruct_task, without importing torch."""
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


def select_records(records: list, frac: float, seed: int,
                   include_faulted: bool = False) -> list:
    """Indices of records to rename, stratified by tier so the meta-skill
    appears in all curriculum stages. Faulted records are included when
    include_faulted=True -- renamed skills then also land in stage 3, keeping
    the meta-skill fresh at the end of the curriculum (forgetting curves show
    stage-1/2-only skills decay by the end)."""
    rng = random.Random(seed)
    by_tier = defaultdict(list)
    for i, r in enumerate(records):
        if not include_faulted and r["meta"].get("faults"):
            continue
        by_tier[r["meta"].get("tier", "?")].append(i)
    selected = []
    for tier in sorted(by_tier):
        idxs = by_tier[tier]
        rng.shuffle(idxs)
        k = max(1, round(len(idxs) * frac)) if frac > 0 else 0
        selected += idxs[:k]
    return sorted(selected)


def plan_renames(records: list, selected: list, seed: int,
                 double_frac: float = 0.0) -> dict:
    """idx -> list of (std_action, new_name, description), 1 or 2 entries.
    Round-robins the candidate names per standard action so different samples
    get different new names. MoveTo is de-prioritized (it is always a
    candidate, which would otherwise crowd out richer-arg actions); double
    renames force reading each skill description instead of pattern-matching
    a single odd name."""
    rng = random.Random(seed + 1)
    rr = Counter()  # std_action -> rotation offset
    plans = {}
    for i in selected:
        tags = set(TAG_RE.findall(records[i]["output"]))
        cands = sorted(tags & SIMPLE_ACTIONS & set(RENAME_POOL))
        if not cands:
            continue
        n_rename = 2 if (len(cands) >= 2 and rng.random() < double_frac) else 1
        chosen = []
        pool_cands = list(cands)
        for _ in range(n_rename):
            pick_from = pool_cands
            if len(pick_from) > 1 and "MoveTo" in pick_from and rng.random() < 0.5:
                pick_from = [c for c in pick_from if c != "MoveTo"]
            std = rng.choice(pick_from)
            new, desc = RENAME_POOL[std][rr[std] % len(RENAME_POOL[std])]
            rr[std] += 1
            chosen.append((std, new, desc))
            pool_cands = [c for c in pool_cands if c != std]
        plans[i] = chosen
    return plans


def regenerate_nl(records: list, plans: dict, workers: int) -> dict:
    """idx -> new NL string (only where generation passed the leak filter).

    Reuses nl_gen.generate_nl_inputs; _facts_summary is wrapped to append the
    same 'Extra skill:' line used for test-domain unique actions. The prompt
    text differs from the original, so the llm disk cache cannot hit stale
    entries."""
    tasks = []
    for i in plans:
        t = dict(records[i]["meta"])
        t["_extra_skill_descs"] = [desc for _, _, desc in plans[i]]
        tasks.append(t)
    orig_summary = nl_gen._facts_summary

    def summary_with_extra(task: dict) -> str:
        s = orig_summary(task)
        for d in task.get("_extra_skill_descs", []):
            s += f"\nExtra skill: the robots have {d}."
        return s

    nl_gen._facts_summary = summary_with_extra
    try:
        nls = nl_gen.generate_nl_inputs(tasks, max_workers=workers)
    finally:
        nl_gen._facts_summary = orig_summary
    return {i: nl for i, nl in zip(plans, nls) if nl is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-nl", action="store_true",
                    help="self-test only: keep the old NL input (no DeepSeek calls)")
    ap.add_argument("--include-faulted", action="store_true",
                    help="allow faulted records (renamed skills then reach stage 3)")
    ap.add_argument("--double-frac", type=float, default=0.0,
                    help="fraction of selected records getting TWO renamed actions")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.train).read_text().splitlines() if l.strip()]
    selected = select_records(records, args.frac, args.seed, args.include_faulted)
    plans = plan_renames(records, selected, args.seed, args.double_frac)
    print(f"selected {len(selected)}/{len(records)} records "
          f"({len(plans)} with a renamable simple action)", flush=True)

    nls = {} if args.skip_nl else regenerate_nl(records, plans, args.workers)

    stats = {"renamed": 0, "nl_failed": 0, "exec_failed": 0}
    per_tier, per_name = Counter(), Counter()
    for i in selected:
        if i not in plans:
            continue
        if not args.skip_nl and i not in nls:
            stats["nl_failed"] += 1
            continue  # keep the original record
        rec = records[i]
        meta = dict(rec["meta"])
        xml = rec["output"]
        aliases = {}
        names = []
        for std, new, _desc in plans[i]:
            xml = rename_xml(xml, std, new)
            aliases[new] = std
            names.append(new)
        meta["skill_aliases"] = aliases
        meta["renamed_skill"] = ",".join(names)
        res = execute(reconstruct_task(meta), xml)
        if not res["success"]:
            stats["exec_failed"] += 1
            print(f"  exec failed after rename ({meta['renamed_skill']}): {res['reason']}", flush=True)
            continue  # keep the original record
        records[i] = {"instruction": rec["instruction"],
                      "input": rec["input"] if args.skip_nl else nls[i],
                      "output": xml, "meta": meta}
        stats["renamed"] += 1
        per_tier[meta.get("tier", "?")] += 1
        for nm in names:
            per_name[nm] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_sel = max(len(plans), 1)
    print(f"wrote {len(records)} records to {out}", flush=True)
    print(f"renamed: {stats['renamed']}  nl_failed: {stats['nl_failed']}  "
          f"exec_failed: {stats['exec_failed']}  "
          f"exec pass rate: {stats['renamed'] / n_sel:.1%} of renamable", flush=True)
    print(f"per tier: {dict(per_tier)}", flush=True)
    print(f"per new name: {dict(per_name)}", flush=True)


if __name__ == "__main__":
    main()
