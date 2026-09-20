"""Select a checkpoint using validation only.

Each candidate is supplied as ``ADAPTER|VALIDATION_JSON``. Test outputs are
intentionally rejected by filename, making it harder to accidentally tune on
the held-out test while assembling a revision result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rate(payload: dict) -> float:
    for key in ("strict_success_rate", "exec_success_rate"):
        if key in payload:
            return float(payload[key])
    raise ValueError("validation result has no success-rate field")


def select(candidates: list[tuple[str, str]]) -> dict:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    rows = []
    for order, (adapter, eval_path) in enumerate(candidates):
        path = Path(eval_path)
        if "test" in path.name.lower():
            raise ValueError(f"refusing to select from a test result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"adapter": adapter, "eval": str(path), "rate": _rate(payload),
                     "n": int(payload.get("n", 0)), "order": order})
    # Stable tie-breaking: earliest checkpoint in the explicitly supplied
    # sequence, which is normally chronological training order.
    winner = max(rows, key=lambda row: (row["rate"], -row["order"]))
    return {"criterion": "validation_success_rate", "selected": winner,
            "candidates": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        help="ADAPTER|VALIDATION_JSON; repeat in chronological order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parsed = []
    for value in args.candidate:
        if "|" not in value:
            raise ValueError(f"candidate must be ADAPTER|VALIDATION_JSON: {value}")
        adapter, eval_path = value.split("|", 1)
        parsed.append((adapter, eval_path))
    result = select(parsed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

