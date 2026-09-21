"""Structural features of a gold behavior tree -- no generation-time labels.

Features (mechanically computed from the output XML):
  n_sync   handshake synchronization pairs (SignalReady/WaitReady, HandoverGive/Take)
           -- "blackboard sync": points where two agents' timelines couple
  n_co     tightly-coupled joint action nodes (CoPickUp/CoMoveTo/CoPlaceDown)
  n_fb     Fallback recovery structures (failure-branch planning)
  depth    max tree depth; n_nodes: tree size

structural_stage maps the features to a coarse coordination level used for
reporting only (1 = single-agent skill, 2 = coordination, 3 = coordination
with fault recovery); SPCL itself consumes the raw features, not the stage.

Usage:
  python -m curriculum.structural --data data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datagen.executor import re_extract  # noqa: E402

SYNC_TAGS = {"SignalReady", "WaitReady", "HandoverGive", "HandoverTake"}
CO_TAGS = {"CoPickUp", "CoMoveTo", "CoPlaceDown"}


def structural_features(xml_str: str) -> dict:
    root = ET.fromstring(re_extract(xml_str))
    tags = Counter(e.tag for e in root.iter())
    depth = _max_depth(root)
    return {
        "n_sync": sum(tags[t] for t in SYNC_TAGS),
        "n_co": sum(tags[t] for t in CO_TAGS),
        "n_fb": tags["Fallback"],
        "depth": depth,
        "n_nodes": sum(tags.values()),
    }


def _max_depth(node, d: int = 0) -> int:
    return max([d] + [_max_depth(c, d + 1) for c in node])


def structural_stage(feat: dict) -> int:
    if feat["n_fb"] > 0:
        return 3
    if feat["n_sync"] > 0 or feat["n_co"] > 0:
        return 2
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.data).read_text().splitlines()
               if l.strip()]
    if args.limit:
        records = records[: args.limit]

    stage_size = Counter()
    feat_by_stage = defaultdict(Counter)
    for r in records:
        feat = structural_features(r["output"])
        s = structural_stage(feat)
        stage_size[s] += 1
        feat_by_stage[s].update(feat)

    print(f"n={len(records)}")
    for s in sorted(feat_by_stage):
        tot = sum(feat_by_stage[s].values())
        means = {k: round(v / tot, 2) for k, v in feat_by_stage[s].items()}
        print(f"  stage {s}: n={stage_size[s]}  mean features {means}")


if __name__ == "__main__":
    main()
