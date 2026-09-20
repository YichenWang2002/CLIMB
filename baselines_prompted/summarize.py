#!/usr/bin/env python3
"""Print a compact result table and exact paired McNemar comparisons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    print("shots\tsuccess\tn\trate\txml_rate\tfile")
    for path, report in zip(args.results, reports):
        success = sum(bool(row["success"]) for row in report["details"])
        print(
            f"{report.get('shots')}\t{success}\t{report['n']}\t"
            f"{report['exec_success_rate']:.4f}\t{report['xml_wellformed_rate']:.4f}\t{path}"
        )
    print("\npaired comparisons (right minus left)")
    for i, left in enumerate(reports):
        for j in range(i + 1, len(reports)):
            right = reports[j]
            left_by_id = {row["record_id"]: bool(row["success"]) for row in left["details"]}
            right_by_id = {row["record_id"]: bool(row["success"]) for row in right["details"]}
            if set(left_by_id) != set(right_by_id):
                raise ValueError("result record IDs do not align")
            left_only = sum(left_by_id[key] and not right_by_id[key] for key in left_by_id)
            right_only = sum(right_by_id[key] and not left_by_id[key] for key in left_by_id)
            difference = right["exec_success_rate"] - left["exec_success_rate"]
            print(
                f"{left.get('shots')}-shot -> {right.get('shots')}-shot: "
                f"diff={difference:+.4f}, left_only={left_only}, right_only={right_only}, "
                f"McNemar p={exact_mcnemar(left_only, right_only):.6g}"
            )


if __name__ == "__main__":
    main()
