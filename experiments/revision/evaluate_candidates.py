"""Score externally generated candidate lists with the same executor oracle.

This is the fair frontier+EGVD interface: a frontier model may generate k
candidates per task, but all systems are then scored by this one deterministic
executor. The input JSONL must contain ``record_id`` and ``candidates`` (a list
of XML strings) for every row in ``--data``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datagen.executor import execute
from experiments.bt_ducl.strict_eval import reconstruct, validate_xml
from .protocol import load_jsonl, row_id


def score(data_rows: list[dict], candidate_rows: list[dict], max_k: int | None = None) -> dict:
    expected = {row_id(row): row for row in data_rows}
    provided = {row.get("record_id"): row for row in candidate_rows}
    if None in provided or len(provided) != len(candidate_rows) or set(expected) != set(provided):
        raise ValueError("candidate file IDs do not exactly match the evaluation split")
    details = []
    for rid, row in expected.items():
        values = provided[rid].get("candidates")
        if not isinstance(values, list) or not values:
            raise ValueError(f"empty candidates for record {rid}")
        if max_k:
            values = values[:max_k]
        if not values:
            raise ValueError(f"max-k removed all candidates for record {rid}")
        if details and len(values) != len(details[0]["succ_at"]):
            raise ValueError("all candidate rows must contain the same k")
        outcomes = []
        for value in values:
            try:
                clean, schema_error = validate_xml(str(value), row["meta"])
                if schema_error:
                    outcomes.append(False)
                    continue
                result = execute(reconstruct(row["meta"]), clean)
                outcomes.append(bool(result.get("success")))
            except Exception:
                outcomes.append(False)
        first = next((index for index, ok in enumerate(outcomes) if ok), None)
        meta = row["meta"]
        details.append({"record_id": rid, "domain": meta["domain"],
                        "tier": meta["tier"], "scenario": meta["scenario"],
                        "succ_at": outcomes,
                        "success": first is not None,
                        "first_success_idx": first,
                        "executor_calls": (first + 1) if first is not None else len(outcomes)})
    ks = sorted({1, 2, 4, len(details[0]["succ_at"])})
    ks = [k for k in ks if k <= len(details[0]["succ_at"])]

    def subset(rows: list[dict]) -> dict:
        return {f"pass@{k}": sum(any(d["succ_at"][:k]) for d in rows) / max(1, len(rows))
                for k in ks}

    per_domain = defaultdict(list)
    per_tier = defaultdict(list)
    for detail in details:
        per_domain[detail["domain"]].append(detail)
        per_tier[detail["tier"]].append(detail)
    return {
        "n": len(details), "k": len(details[0]["succ_at"]),
        "overall": subset(details),
        "per_domain": {key: subset(value) for key, value in sorted(per_domain.items())},
        "per_tier": {key: subset(value) for key, value in sorted(per_tier.items())},
        "mean_executor_calls": sum(d["executor_calls"] for d in details) / len(details),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-k", type=int, default=0)
    args = parser.parse_args()
    data = load_jsonl(args.data)
    candidates = [json.loads(line) for line in Path(args.candidates).read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    result = score(data, candidates, args.max_k or None)
    result["data"] = str(args.data)
    result["candidates"] = str(args.candidates)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print(json.dumps({"n": result["n"], "k": result["k"],
                      "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
