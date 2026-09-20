"""Attach split metadata to legacy evaluator details without rerunning models.

Older constrained-evaluation files predate the ``scenario`` field in each
detail.  This utility joins only on the stable benchmark ``record_id`` and
copies non-outcome metadata from the declared split.  It refuses missing or
extra IDs and never changes ``success``, ``reason``, or generations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import load_jsonl, row_id


def enrich(result_path: str | Path, data_path: str | Path,
           out_path: str | Path) -> dict:
    result_file, data_file, out_file = map(Path, (result_path, data_path, out_path))
    result = json.loads(result_file.read_text(encoding="utf-8"))
    details = result.get("details")
    if not isinstance(details, list) or not details:
        raise ValueError("result lacks non-empty details")
    rows = load_jsonl(data_file)
    metadata = {}
    for row in rows:
        rid = row_id(row)
        if rid in metadata:
            raise ValueError(f"duplicate data record_id {rid}")
        meta = row.get("meta", {})
        metadata[rid] = {"domain": meta.get("domain"), "tier": meta.get("tier"),
                         "scenario": meta.get("scenario", "")}
    detail_ids = [item.get("record_id") for item in details]
    if any(rid is None for rid in detail_ids) or len(detail_ids) != len(set(detail_ids)):
        raise ValueError("result details lack unique record IDs")
    if set(detail_ids) != set(metadata):
        raise ValueError("result/data record IDs do not match exactly")
    for item in details:
        item.update(metadata[item["record_id"]])
    result["metadata_enrichment"] = {
        "data": str(data_file), "fields": ["domain", "tier", "scenario"],
        "record_ids_verified": True,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return {"out": str(out_file), "n": len(details), "record_ids_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(enrich(args.result, args.data, args.out), indent=2))


if __name__ == "__main__":
    main()
