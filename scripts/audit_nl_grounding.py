#!/usr/bin/env python3
"""Grounding audit: every entity id mentioned in a mission text must exist
in the executor-reconstructable platform configuration (meta).

A mission "hallucinates" if it cites an id outside {robots, items,
locations (connected endpoints), charge stations}. Coverage is also
reported: the fraction of missions whose text mentions every goal entity
id. Run over the released splits before citing the numbers.

Usage:  python scripts/audit_nl_grounding.py
"""
import json
import re
from pathlib import Path

DS = Path(__file__).resolve().parents[1] / "outputs" / "dataset"
ID_RE = re.compile(r"\(([a-z0-9_]+)\)")  # ids appear as 'shelf A (shelf_a)'


def known_entities(meta: dict) -> set:
    return (set(meta["robots"]) | set(meta["items"])
            | {l for pair in meta["connected"] for l in pair}
            | set(meta["charge_stations"]))


def audit(split: str) -> dict:
    n = bad = cov = 0
    examples = []
    for line in (DS / f"{split}.jsonl").open():
        r = json.loads(line)
        meta, n = r["meta"], n + 1
        known = known_entities(meta)
        ids = set(ID_RE.findall(r["input"]))
        unknown = ids - known
        if unknown:
            bad += 1
            if len(examples) < 5:
                examples.append({"line": n, "unknown": sorted(unknown)})
        goal_ids = {t for f in meta["goal"] for t in f[1:] if t in known}
        cov += goal_ids <= ids
    print(f"{split}: {n} missions | unknown-id citations: {bad} "
          f"({bad / n:.2%}) | goal-entity coverage: {cov} ({cov / n:.2%})")
    if examples:
        print("  examples:", examples)
    return {"n": n, "bad": bad, "coverage": cov}


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        audit(s)
