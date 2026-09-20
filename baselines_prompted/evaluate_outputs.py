#!/usr/bin/env python3
"""Score saved LLM responses with the bundled symbolic executor."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# one frozen executor, shared with corpus construction and all paper evals
sys.path.insert(0, str(ROOT.parent))

from datagen.executor import execute, re_extract  # noqa: E402
from datagen.strips.domains import build_domain  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reconstruct_task(meta: dict) -> dict:
    static = [("can_reach", robot, location) for robot, location in meta["can_reach"]]
    edges = [("connected", left, right) for left, right in meta["connected"]]
    domain = build_domain(
        meta["domain"],
        robots=meta["robots"],
        items=meta["items"],
        extra_static=static,
        edges=edges,
    )
    return {
        "domain": meta["domain"],
        "domain_obj": domain,
        "init_dynamic": [tuple(fact) for fact in meta["init_dynamic"]],
        "goal": [tuple(fact) for fact in meta["goal"]],
        "faults": meta["faults"],
        "skill_aliases": meta.get("skill_aliases", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbolically evaluate an API response JSONL.")
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--eval-data", type=Path, default=ROOT / "data" / "test_eval.jsonl")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    eval_rows = load_jsonl(args.eval_data)
    response_rows = load_jsonl(args.responses)
    by_index = {int(row["index"]): row for row in response_rows}
    if len(by_index) != len(response_rows):
        raise ValueError("response file contains duplicate indices; rerun run_api.py to normalize it")
    expected = set(range(len(eval_rows)))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(f"response coverage mismatch: missing={missing[:20]} extra={extra[:20]}")

    per_domain = defaultdict(Counter)
    per_scenario = defaultdict(Counter)
    per_primitive = defaultdict(Counter)
    fail_reasons = Counter()
    details = []
    xml_ok_total = 0
    for expected_index, eval_row in enumerate(eval_rows):
        if int(eval_row["index"]) != expected_index:
            raise ValueError("evaluation indices are not contiguous")
        response = by_index[expected_index]
        if response.get("record_id") != eval_row["record_id"]:
            raise ValueError(f"record ID mismatch at index {expected_index}")
        generation = response.get("response", "")
        meta = eval_row["meta"]
        try:
            re_extract(generation)
            xml_ok = True
        except Exception:  # noqa: BLE001
            xml_ok = False
        xml_ok_total += xml_ok
        result = (
            execute(reconstruct_task(meta), generation)
            if xml_ok
            else {"success": False, "reason": "no_xml_block", "recoveries": 0}
        )
        success = bool(result["success"])
        if not success:
            fail_reasons[result["reason"]] += 1
        per_domain[meta["domain"]]["n"] += 1
        per_domain[meta["domain"]]["ok"] += success
        per_scenario[meta["scenario"]]["n"] += 1
        per_scenario[meta["scenario"]]["ok"] += success
        primitive = "new_primitive" if "+service" in meta["scenario"] else "shared_primitive"
        per_primitive[primitive]["n"] += 1
        per_primitive[primitive]["ok"] += success
        details.append(
            {
                "index": expected_index,
                "record_id": eval_row["record_id"],
                "domain": meta["domain"],
                "scenario": meta["scenario"],
                "scenario": meta["scenario"],
                "success": success,
                "reason": result["reason"],
                "recoveries": result.get("recoveries", 0),
            }
        )

    n = len(eval_rows)
    first = response_rows[0]
    report = {
        "n": n,
        "exec_success_rate": sum(row["success"] for row in details) / n,
        "xml_wellformed_rate": xml_ok_total / n,
        "per_domain": {key: dict(value) for key, value in per_domain.items()},
        "per_scenario": {key: dict(value) for key, value in per_scenario.items()},
        "per_primitive": {key: dict(value) for key, value in per_primitive.items()},
        "fail_reasons": dict(fail_reasons),
        "model": first.get("model"),
        "shots": first.get("shots"),
        "responses": str(args.responses.resolve()),
        "details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
