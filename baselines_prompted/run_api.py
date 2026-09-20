#!/usr/bin/env python3
"""Run deterministic 0/1/5-shot baselines through an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPTS = ROOT / "data" / "test_prompts.jsonl"
DEFAULT_DEMOS = ROOT / "data" / "fixed_5shot_demos.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def build_messages(record: dict, demos: list[dict], shots: int) -> list[dict]:
    messages = [{"role": "system", "content": record["instruction"]}]
    for demo in demos[:shots]:
        messages.append({"role": "user", "content": demo["input"]})
        messages.append({"role": "assistant", "content": demo["output"]})
    messages.append({"role": "user", "content": record["input"]})
    return messages


def fingerprint(model: str, messages: list[dict], request_args: dict) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "request": request_args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    existing = {}
    for row in rows:
        if "index" not in row:
            raise ValueError(f"existing output has no index: {path}")
        existing[int(row["index"])] = row
    return existing


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed 0/1/5-shot behavior-tree baseline."
    )
    parser.add_argument("--model", required=True, help="API model identifier")
    parser.add_argument("--shots", required=True, type=int, choices=(0, 1, 5))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI-compatible base URL; defaults to OPENAI_BASE_URL/provider default",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key; defaults to OPENAI_API_KEY (use EMPTY for a local server)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument(
        "--token-param",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
        help="Token-limit parameter expected by the selected provider",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0 or args.max_attempts <= 0:
        parser.error("--workers and --max-attempts must be positive")

    prompts = load_jsonl(args.prompts)
    demos = load_jsonl(args.demos)
    if args.limit:
        prompts = prompts[: args.limit]
    if len(demos) < args.shots:
        raise ValueError(f"need {args.shots} demos, found {len(demos)}")
    for expected, row in enumerate(prompts):
        if int(row["index"]) != expected:
            raise ValueError("prompt indices must be contiguous and ordered")

    request_args = {
        "temperature": args.temperature,
        args.token_param: args.max_tokens,
    }
    jobs = []
    for row in prompts:
        messages = build_messages(row, demos, args.shots)
        jobs.append((row, messages, fingerprint(args.model, messages, request_args)))

    if args.dry_run:
        preview = {
            "model": args.model,
            "shots": args.shots,
            "n_tasks": len(jobs),
            "messages_for_first_task": jobs[0][1] if jobs else [],
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install -r requirements.txt") from exc

    api_key = args.api_key
    if not api_key and args.base_url and any(
        marker in args.base_url for marker in ("localhost", "127.0.0.1", "0.0.0.0")
    ):
        api_key = "EMPTY"
    if not api_key:
        raise SystemExit("set OPENAI_API_KEY or pass --api-key")

    client_kwargs = {"api_key": api_key, "timeout": args.timeout, "max_retries": 0}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(args.out)
    pending = []
    for row, messages, request_fp in jobs:
        old = existing.get(int(row["index"]))
        if old and old.get("response") and old.get("request_fingerprint") == request_fp:
            continue
        if old and old.get("response") and old.get("request_fingerprint") != request_fp:
            raise ValueError(
                f"output {args.out} contains an incompatible completed request at "
                f"index {row['index']}; use a different output file"
            )
        pending.append((row, messages, request_fp))

    print(
        f"model={args.model} shots={args.shots} tasks={len(jobs)} "
        f"resume_hits={len(jobs) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    def call_one(job):
        row, messages, request_fp = job
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            started = time.time()
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    **request_args,
                )
                content = response.choices[0].message.content or ""
                return {
                    "index": int(row["index"]),
                    "record_id": row["record_id"],
                    "model": args.model,
                    "shots": args.shots,
                    "request_fingerprint": request_fp,
                    "response": content,
                    "usage": usage_dict(response),
                    "latency_s": round(time.time() - started, 3),
                    "attempts": attempt,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_attempts:
                    time.sleep(min(2 ** attempt, 30))
        return {
            "index": int(row["index"]),
            "record_id": row["record_id"],
            "model": args.model,
            "shots": args.shots,
            "request_fingerprint": request_fp,
            "response": "",
            "usage": {},
            "latency_s": None,
            "attempts": args.max_attempts,
            "error": last_error,
        }

    completed = 0
    failures = 0
    with args.out.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(call_one, job): job[0]["index"] for job in pending}
            for future in as_completed(futures):
                result = future.result()
                existing[result["index"]] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                failures += not bool(result["response"])
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        f"completed={completed}/{len(pending)} failures={failures}",
                        flush=True,
                    )

    final_rows = [existing[i] for i in range(len(jobs)) if i in existing]
    atomic_write_jsonl(args.out, final_rows)
    missing = [i for i in range(len(jobs)) if i not in existing or not existing[i].get("response")]
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in final_rows:
        for key in totals:
            value = row.get("usage", {}).get(key)
            if isinstance(value, int):
                totals[key] += value
    summary = {
        "model": args.model,
        "shots": args.shots,
        "expected": len(jobs),
        "completed": len(jobs) - len(missing),
        "missing_or_failed_indices": missing,
        "usage": totals,
        "output": str(args.out.resolve()),
    }
    args.out.with_suffix(args.out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing:
        print("rerun the same command to retry failed rows", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
