"""Structural curriculum stages: assign stages purely from the target
behavior tree's structural features -- no generation-time labels.

Features (mechanically computed from the output XML):
  n_sync   handshake synchronization pairs (SignalReady/WaitReady, HandoverGive/Take)
           -- "blackboard sync": points where two agents' timelines couple
  n_co     tightly-coupled joint action nodes (CoPickUp/CoMoveTo/CoPlaceDown)
  n_fb     Fallback recovery structures (failure-branch planning)
  depth    max tree depth; n_nodes: tree size; plan_len not needed here

Stage predicate (deterministic, domain-agnostic):
  Stage 1: n_sync=0 and n_co=0 and n_fb=0   (single-agent compositional skills)
  Stage 2: (n_sync>0 or n_co>0) and n_fb=0  (coordination, no recovery)
  Stage 3: n_fb>0                           (failure-recovery structures)

Usage:
  python -m curriculum.structural --data outputs/dataset/train_aug10.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from curriculum.build_curriculum import stage_of  # noqa: E402
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
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    conf = Counter()          # (label_stage, structural_stage) -> count
    feat_by_stage = {1: Counter(), 2: Counter(), 3: Counter()}
    mismatches = []
    for i, r in enumerate(records):
        feat = structural_features(r["output"])
        s_struct = structural_stage(feat)
        s_label = stage_of(r)
        conf[(s_label, s_struct)] += 1
        for k, v in feat.items():
            feat_by_stage[s_struct][k] += v
        if s_struct != s_label and len(mismatches) < 10:
            mismatches.append((i, s_label, s_struct, feat))

    n = len(records)
    agree = sum(conf[(s, s)] for s in (1, 2, 3))
    print(f"n={n}  agreement={agree}/{n} = {agree / n:.2%}\n")
    print("confusion (rows=label stage_of, cols=structural):")
    print("          S1    S2    S3")
    for sl in (1, 2, 3):
        row = [conf[(sl, sc)] for sc in (1, 2, 3)]
        print(f"  label{sl} {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    print("\nmean features per structural stage:")
    for s in (1, 2, 3):
        tot = sum(conf[(sl, s)] for sl in (1, 2, 3))
        if tot:
            means = {k: round(v / tot, 2) for k, v in feat_by_stage[s].items()}
            print(f"  S{s}: n={tot}  {means}")
    if mismatches:
        print("\nsample mismatches (idx, label, structural, features):")
        for m in mismatches:
            print(" ", m)


if __name__ == "__main__":
    main()
