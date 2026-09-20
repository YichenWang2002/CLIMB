"""Align repeated API generations with the benchmark and build candidate JSONL.

Each source is a JSONL emitted by ``LLMbase/run_api.py`` (``index`` and
``response`` fields). Sources must cover the same contiguous task indices. The
benchmark row, rather than an API-side hash, supplies the stable record ID so
the resulting file can be scored by ``evaluate_candidates.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import load_jsonl, row_id


def _source(path: str | Path, n: int) -> list[str]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if "index" not in value or "response" not in value:
            raise ValueError(f"{path}:{line_no} needs index and response")
        rows.append(value)
    if len(rows) != n:
        raise ValueError(f"{path} has {len(rows)} rows; expected {n}")
    rows.sort(key=lambda item: int(item["index"]))
    if [int(item["index"]) for item in rows] != list(range(n)):
        raise ValueError(f"{path} indices are not exactly 0..{n - 1}")
    responses = [str(item.get("response") or "") for item in rows]
    if any(not response.strip() for response in responses):
        raise ValueError(f"{path} contains an empty API response")
    return responses


def build(data_path: str | Path, source_paths: list[str | Path], out_path: str | Path) -> dict:
    data = load_jsonl(data_path)
    if not source_paths:
        raise ValueError("at least one --source is required")
    sources = [_source(path, len(data)) for path in source_paths]
    rows = []
    for index, record in enumerate(data):
        # Keep repeated generations: pass@k and executor-call accounting are
        # defined over the requested k draws, and the scorer requires a fixed
        # candidate count for every task.
        candidates = [responses[index] for responses in sources]
        rows.append({"record_id": row_id(record), "candidates": candidates})
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "data": str(data_path), "sources": [str(path) for path in source_paths],
        "n": len(rows), "requested_k": len(source_paths),
        "candidate_count": len(source_paths),
        "out": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="one complete API JSONL; repeat for candidate draws")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
