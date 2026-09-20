#!/usr/bin/env python3
"""Compare Qwen Flat/SPCL with and without topology-SCD from frozen eval JSON files."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

METHODS = ("flat", "spcl")
MODES = ("unconstrained", "topology_scd")


def load_details(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details", [])
    if not details:
        raise ValueError(f"No details in {path}")
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(details):
        key = row.get("record_id") or f"idx:{index}"
        if key in result:
            raise ValueError(f"Duplicate record key {key} in {path}")
        result[key] = row
    return result


def exact_mcnemar_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pair(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(f"Record mismatch: missing_left={missing_left}, missing_right={missing_right}")
    buckets: dict[str, list[str]] = defaultdict(list)
    buckets["overall"] = sorted(left)
    for record_id in sorted(left):
        tier = left[record_id].get("tier", "unknown")
        buckets[tier].append(record_id)
        if tier in {"T2", "T3"}:
            buckets["coordination"] .append(record_id)
    results: dict[str, Any] = {}
    for name, record_ids in buckets.items():
        left_ok = sum(bool(left[r].get("success")) for r in record_ids)
        right_ok = sum(bool(right[r].get("success")) for r in record_ids)
        wins = sum((not bool(left[r].get("success"))) and bool(right[r].get("success")) for r in record_ids)
        losses = sum(bool(left[r].get("success")) and (not bool(right[r].get("success"))) for r in record_ids)
        results[name] = {
            "n": len(record_ids),
            "left_success": left_ok,
            "right_success": right_ok,
            "left_rate": round(100 * left_ok / len(record_ids), 3),
            "right_rate": round(100 * right_ok / len(record_ids), 3),
            "delta_pp": round(100 * (right_ok - left_ok) / len(record_ids), 3),
            "right_only_wins": wins,
            "left_only_wins": losses,
            "mcnemar_exact_p": exact_mcnemar_p(wins, losses),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-unconstrained", type=Path, required=True)
    parser.add_argument("--spcl-unconstrained", type=Path, required=True)
    parser.add_argument("--flat-scd", type=Path, required=True)
    parser.add_argument("--spcl-scd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    flat_un = load_details(args.flat_unconstrained)
    spcl_un = load_details(args.spcl_unconstrained)
    flat_scd = load_details(args.flat_scd)
    spcl_scd = load_details(args.spcl_scd)
    payload = {
        "flat_scd_vs_flat_unconstrained": pair(flat_un, flat_scd),
        "spcl_scd_vs_spcl_unconstrained": pair(spcl_un, spcl_scd),
        "spcl_vs_flat_unconstrained": pair(flat_un, spcl_un),
        "spcl_vs_flat_topology_scd": pair(flat_scd, spcl_scd),
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "out"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Author: Manus AI
# License: MIT
# Created: 2026-08-23
# Purpose: Compute paired topology-SCD effects for the frozen Qwen replication.
# End of file.
