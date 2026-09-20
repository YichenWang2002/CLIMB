#!/usr/bin/env python3
"""Audit SFT sequence truncation with the project's actual chat-template logic."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(record: dict[str, Any]) -> str:
    return record.get("meta", {}).get("tier", "unknown")


def summarize(values: list[dict[str, int]]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "truncated": 0, "truncation_rate": 0.0, "mean_full_tokens": 0.0,
                "p95_full_tokens": 0, "mean_prompt_tokens": 0.0, "mean_target_tokens_lost": 0.0}
    fulls = sorted(v["full_tokens"] for v in values)
    index = max(0, min(len(fulls) - 1, int(0.95 * (len(fulls) - 1))))
    return {
        "n": len(values),
        "truncated": sum(v["truncated"] for v in values),
        "truncation_rate": round(sum(v["truncated"] for v in values) / len(values), 6),
        "mean_full_tokens": round(sum(v["full_tokens"] for v in values) / len(values), 3),
        "p95_full_tokens": fulls[index],
        "mean_prompt_tokens": round(sum(v["prompt_tokens"] for v in values) / len(values), 3),
        "mean_target_tokens_lost": round(sum(v["target_tokens_lost"] for v in values) / len(values), 3),
    }


def audit(path: Path, tokenizer: Any, max_len: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, int]]] = defaultdict(list)
    records = load_records(path)
    for record in records:
        messages = [
            {"role": "system", "content": record["instruction"]},
            {"role": "user", "content": record["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = prompt + record["output"] + tokenizer.eos_token
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        full_tokens = len(tokenizer(full, add_special_tokens=False)["input_ids"])
        target_tokens = max(0, full_tokens - prompt_tokens)
        retained_target = max(0, min(full_tokens, max_len) - min(prompt_tokens, max_len))
        groups[group_key(record)].append({
            "prompt_tokens": prompt_tokens,
            "full_tokens": full_tokens,
            "truncated": int(full_tokens > max_len),
            "target_tokens_lost": max(0, target_tokens - retained_target),
        })
    return {
        "path": str(path), "max_len": max_len, "groups": {key: summarize(rows) for key, rows in sorted(groups.items())},
        "overall": summarize([item for rows in groups.values() for item in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-len", required=True, type=int)
    parser.add_argument("--data", required=True, action="append", help="label=path; repeatable")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    payload: dict[str, Any] = {"model": args.model, "max_len": args.max_len, "datasets": {}}
    for item in args.data:
        label, raw_path = item.split("=", 1)
        payload["datasets"][label] = audit(Path(raw_path), tokenizer, args.max_len)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        label: {"overall": value["overall"], "T3": value["groups"].get("T3", {})}
        for label, value in payload["datasets"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# Author: Manus AI
# License: MIT
# Created: 2026-08-23
# Purpose: Compare actual tokenizer-dependent truncation under frozen SFT streams.
# End of file.
