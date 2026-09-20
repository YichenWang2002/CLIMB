#!/usr/bin/env python3
"""Audit frozen paired Flat-vs-SPCL strict-evaluation outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_id(record: dict[str, Any]) -> str:
    payload = json.dumps(
        {key: record[key] for key in ("instruction", "input", "output")},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_primitive(record: dict[str, Any]) -> bool:
    return "+service" in str(record["meta"].get("scenario", ""))


def paired_state(flat: dict[str, Any], spcl: dict[str, Any]) -> str:
    if flat["success"] and spcl["success"]:
        return "both_success"
    if flat["success"]:
        return "flat_only"
    if spcl["success"]:
        return "spcl_only"
    return "both_fail"


def avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat_chars = [row["flat"]["generated_chars"] for row in rows]
    spcl_chars = [row["spcl"]["generated_chars"] for row in rows]
    states = Counter(row["state"] for row in rows)
    return {
        "n": len(rows),
        "flat_success": sum(row["flat"]["success"] for row in rows),
        "spcl_success": sum(row["spcl"]["success"] for row in rows),
        "transitions": dict(sorted(states.items())),
        "mean_generated_chars": {
            "flat": avg(flat_chars), "spcl": avg(spcl_chars),
            "spcl_minus_flat": round(mean(spcl_chars) - mean(flat_chars), 3) if rows else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", required=True, type=Path)
    parser.add_argument("--spcl", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    flat = {row["record_id"]: row for row in read_json(args.flat)["details"]}
    spcl = {row["record_id"]: row for row in read_json(args.spcl)["details"]}
    test_rows = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = {record_id(row): row for row in test_rows}
    if set(flat) != set(spcl) or set(flat) != set(test):
        raise ValueError("Flat, SPCL, and test record IDs must match exactly")

    rows: list[dict[str, Any]] = []
    for rid in sorted(test):
        source = test[rid]
        f, s = flat[rid], spcl[rid]
        rows.append({
            "record_id": rid, "tier": source["meta"]["tier"],
            "domain": source["meta"]["domain"], "scenario": source["meta"].get("scenario", ""),
            "new_primitive": new_primitive(source), "flat": f, "spcl": s, "state": paired_state(f, s),
        })

    groups: dict[str, list[dict[str, Any]]] = {
        "overall": rows,
        "T1": [r for r in rows if r["tier"] == "T1"],
        "T2": [r for r in rows if r["tier"] == "T2"],
        "T3": [r for r in rows if r["tier"] == "T3"],
        "coordination": [r for r in rows if r["tier"] in {"T2", "T3"}],
        "new_primitive": [r for r in rows if r["new_primitive"]],
        "shared_primitive": [r for r in rows if not r["new_primitive"]],
    }
    for tier in ("T1", "T2", "T3"):
        for label, value in (("new", True), ("shared", False)):
            groups[f"{tier}_{label}_primitive"] = [r for r in rows if r["tier"] == tier and r["new_primitive"] == value]
    for domain in sorted({r["domain"] for r in rows}):
        groups[f"domain_{domain}"] = [r for r in rows if r["domain"] == domain]

    reason_by_state: dict[str, dict[str, Counter[str]]] = {
        state: {"flat": Counter(), "spcl": Counter()} for state in ("flat_only", "spcl_only", "both_fail")
    }
    chars_by_state: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        chars_by_state[row["state"]]["flat"].append(row["flat"]["generated_chars"])
        chars_by_state[row["state"]]["spcl"].append(row["spcl"]["generated_chars"])
        if row["state"] in reason_by_state:
            for arm in ("flat", "spcl"):
                if not row[arm]["success"]:
                    reason_by_state[row["state"]][arm][row[arm]["reason"]] += 1

    def net_records(state: str, failed_arm: str) -> list[dict[str, Any]]:
        return [{
            "record_id": r["record_id"], "tier": r["tier"], "domain": r["domain"],
            "scenario": r["scenario"], "new_primitive": r["new_primitive"],
            "failure_reason": r[failed_arm]["reason"],
            "flat_generated_chars": r["flat"]["generated_chars"],
            "spcl_generated_chars": r["spcl"]["generated_chars"],
        } for r in rows if r["state"] == state]

    output = {
        "groups": {name: group_summary(group) for name, group in groups.items()},
        "failure_reasons": {
            arm: dict(Counter(r[arm]["reason"] for r in rows if not r[arm]["success"]).most_common())
            for arm in ("flat", "spcl")
        },
        "failure_reasons_by_transition": {
            state: {arm: dict(counter.most_common()) for arm, counter in arms.items()}
            for state, arms in reason_by_state.items()
        },
        "mean_chars_by_transition": {
            state: {arm: avg(values) for arm, values in arms.items()} for state, arms in chars_by_state.items()
        },
        "net_loss_records": net_records("flat_only", "spcl"),
        "net_gain_records": net_records("spcl_only", "flat"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "overall": output["groups"]["overall"],
        "T3_new_primitive": output["groups"]["T3_new_primitive"],
        "reason_by_transition": output["failure_reasons_by_transition"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Author: Manus AI
# License: MIT
# Created: 2026-08-23
# Purpose: Reproducible audit of frozen paired strict-evaluation outputs.
# End of file.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#】【。】【”】【} is not valid JSON. The previous Write call made no changes. кеүarnissaaassistant to=functions.file  天天爱彩票appանչjson  北京pk赛车{
