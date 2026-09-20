"""Pre-training data, XML, provenance, and compute audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import load_jsonl, record_id
from .strict_eval import reconstruct, validate_xml
from datagen.executor import execute


def task_id(row: dict) -> str:
    m = row["meta"]
    payload = {k: m.get(k) for k in (
        "domain", "tier", "scenario", "robots", "items", "init_dynamic",
        "goal", "faults", "zones", "connected", "charge_stations", "can_reach",
    )}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def audit_file(path: str) -> dict:
    rows = load_jsonl(path)
    counts = Counter()
    task_ids = set()
    record_ids = set()
    errors = []
    for i, row in enumerate(rows):
        meta = row["meta"]
        counts[(meta.get("domain"), meta.get("tier"))] += 1
        tid = task_id(row)
        if tid in task_ids and len(errors) < 20:
            errors.append({"row": i, "error": "duplicate_task_signature"})
        task_ids.add(tid)
        rid = record_id(row)
        if rid in record_ids and len(errors) < 20:
            errors.append({"row": i, "error": "duplicate_record_id"})
        record_ids.add(rid)
        clean, schema_error = validate_xml(row["output"], meta)
        if schema_error:
            if len(errors) < 20:
                errors.append({"row": i, "error": schema_error})
            continue
        result = execute(reconstruct(meta), clean)
        if not result.get("success"):
            if len(errors) < 20:
                errors.append({"row": i, "error": result.get("reason", "execution_failed")})
    return {"path": path, "n": len(rows), "domain_tier": {
        f"{d}/{t}": n for (d, t), n in sorted(counts.items())
    }, "unique_task_signatures": len(task_ids), "unique_record_ids": len(record_ids),
            "errors_sample": errors, "error_count_lower_bound": len(errors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()
    reports = [audit_file(p) for p in (args.train, args.val, args.test)]
    all_ids = []
    for p in (args.train, args.val, args.test):
        all_ids.extend(task_id(r) for r in load_jsonl(p))
    cross_dup = len(all_ids) - len(set(all_ids))
    if any(r["errors_sample"] for r in reports) or cross_dup:
        raise SystemExit(json.dumps({"reports": reports, "cross_split_task_duplicates": cross_dup}, indent=2))
    result = {"reports": reports, "cross_split_task_duplicates": cross_dup}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
