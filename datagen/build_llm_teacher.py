"""LLM-teacher dataset ablation: replace planner-compiled gold BTs with
API-LLM 5-shot BTs on the SAME natural-language inputs, keep only
executor-verified trees, and write a drop-in train file for flat SFT.

Isolation property: every written row shares instruction/input/meta with
train.jsonl; only `output` differs (who wrote the XML). The fixed 5-shot
demos are the same ones used by the API baselines (all drawn from train).

Default generation target is train_aug10.jsonl (the canonical 6,000-input
training set behind the paper's flat/SPCL numbers), so the teacher file is a
drop-in replacement whose inputs match the planner arm byte-for-byte.

Round 1 samples every task at temperature 0; each later round resamples
only still-unverified tasks at a higher temperature. Disk-cached, so
reruns are free and never regenerate paid content.

Usage:
  OPENAI_API_KEY=... python3 -m datagen.build_llm_teacher \
      --train outputs/dataset/train_aug10.jsonl \
      --out outputs/dataset/train_llmteacher.jsonl

  # offline wiring check: planner gold outputs must pass ~100%
  python3 -m datagen.build_llm_teacher --selftest --limit 200 \
      --train outputs/dataset/train_aug10.jsonl --out /tmp/unused.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from common.llm import chat_batch, stats as llm_stats
from datagen.executor import ExecutionError, execute, re_extract
from datagen.strips.domains import build_domain

DEMOS = Path(__file__).resolve().parents[1] / "data" / "fixed_5shot_demos.jsonl"


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def record_id(record: dict) -> str:
    """Same identity hash as eval.evaluate.stable_record_id / LLMbase demos."""
    blob = json.dumps({
        "instruction": record["instruction"], "input": record["input"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reconstruct_task(meta: dict) -> dict:
    # Mirrors eval.evaluate.reconstruct_task (kept local so this module
    # stays torch-free and importable on CPU-only nodes).
    static = [("can_reach", r, l) for r, l in meta["can_reach"]]
    edges = [("connected", a, b) for a, b in meta["connected"]]
    domain = build_domain(meta["domain"], robots=meta["robots"],
                          items=meta["items"], extra_static=static, edges=edges)
    return {
        "domain": meta["domain"], "domain_obj": domain,
        "init_dynamic": [tuple(f) for f in meta["init_dynamic"]],
        "goal": [tuple(f) for f in meta["goal"]],
        "faults": meta["faults"],
        "skill_aliases": meta.get("skill_aliases", {}),
    }


def build_messages(record: dict, demos: list, shots: int) -> list:
    # Mirrors LLMbase.run_api.build_messages (the API-baseline interface).
    messages = [{"role": "system", "content": record["instruction"]}]
    for demo in demos[:shots]:
        messages.append({"role": "user", "content": demo["input"]})
        messages.append({"role": "assistant", "content": demo["output"]})
    messages.append({"role": "user", "content": record["input"]})
    return messages


def verify(record: dict, content: str) -> tuple[bool, str | None, str | None]:
    """(ok, xml, reason) for one LLM completion against the record's task."""
    if not content or not content.strip():
        return False, None, "empty_response"
    try:
        xml = re_extract(content)
    except ExecutionError:
        return False, None, "no_root_block"
    res = execute(reconstruct_task(record["meta"]), xml)
    if res.get("success"):
        return True, xml, None
    return False, None, (res.get("reason") or "exec_fail")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="outputs/dataset/train_aug10.jsonl")
    ap.add_argument("--out", default="outputs/dataset/train_llmteacher.jsonl")
    ap.add_argument("--demos", type=Path, default=DEMOS)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--model", default=None,
                    help="defaults to OPENAI_MODEL / deepseek-v4-flash")
    ap.add_argument("--max-rounds", type=int, default=4,
                    help="round 1 at --temp-first, rounds 2+ resample failures "
                         "at --temp-resample; stop early when all verified")
    ap.add_argument("--temp-first", type=float, default=0.0)
    ap.add_argument("--temp-resample", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--thinking", action="store_true",
                    help="leave DeepSeek thinking mode on (default: disabled "
                         "via extra_body; with it on, reasoning shares the "
                         "completion budget and truncates the XML)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N tasks")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: run planner gold outputs through the same "
                         "verify() path; expect ~100%% pass; no API calls")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.train))
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} tasks from {args.train}", flush=True)

    if args.selftest:
        n_ok, reasons = 0, Counter()
        for i, r in enumerate(rows):
            ok, _, reason = verify(r, r["output"])
            n_ok += ok
            if not ok:
                reasons[reason.split(":")[0][:60]] += 1
        print(f"selftest: {n_ok}/{len(rows)} planner gold outputs pass verify()")
        for reason, n in reasons.most_common():
            print(f"  FAIL {reason}: {n}")
        return

    demos = load_jsonl(args.demos)
    train_ids = {record_id(r) for r in rows}
    n_demo_in_train = sum(1 for d in demos[: args.shots]
                          if d["record_id"] in train_ids)
    print(f"{len(demos)} demos loaded; {n_demo_in_train}/{args.shots} demo rows "
          f"appear in this train split (their teacher output will be a "
          f"near-copy of the in-context demo; negligible at this scale)",
          flush=True)

    verified: dict[int, tuple[str, int]] = {}   # idx -> (xml, round)
    reasons_total: Counter = Counter()
    rounds_report = []
    pending = list(range(len(rows)))
    t0 = time.time()

    for round_no in range(1, args.max_rounds + 1):
        if not pending:
            break
        temp = args.temp_first if round_no == 1 else args.temp_resample
        jobs = [build_messages(rows[i], demos, args.shots) for i in pending]
        print(f"round {round_no}: {len(jobs)} tasks, T={temp}, "
              f"thinking={'on' if args.thinking else 'off'}, "
              f"{llm_stats()['prompt_tokens']} prompt tokens so far", flush=True)
        extra = None if args.thinking else {"thinking": {"type": "disabled"}}
        contents = chat_batch(jobs, max_workers=args.workers, model=args.model,
                              temperature=temp, max_tokens=args.max_tokens,
                              **({"extra_body": extra} if extra else {}))

        still, newly = [], 0
        for i, content in zip(pending, contents):
            ok, xml, reason = verify(rows[i], content)
            if ok:
                verified[i] = (xml, round_no)
                newly += 1
            else:
                reasons_total[reason.split(":")[0][:60]] += 1
                still.append(i)
        pending = still
        rounds_report.append({"round": round_no, "temperature": temp,
                              "attempted": len(jobs), "newly_verified": newly,
                              "remaining": len(pending)})
        print(f"round {round_no}: +{newly} verified, {len(pending)} remaining "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    # Write in original train order; inputs are byte-identical to the
    # planner arm, only `output` (and a provenance field) differ.
    out_rows = []
    for i in sorted(verified):
        xml, round_no = verified[i]
        r = rows[i]
        meta = dict(r["meta"])
        meta["llm_teacher"] = {"model": args.model or "deepseek-v4-flash",
                               "round": round_no}
        out_rows.append({"instruction": r["instruction"], "input": r["input"],
                         "output": xml, "meta": meta})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_tier = Counter(r["meta"]["tier"] for r in rows)
    acc_tier = Counter(rows[i]["meta"]["tier"] for i in verified)
    by_dom = Counter(r["meta"]["domain"] for r in rows)
    acc_dom = Counter(rows[i]["meta"]["domain"] for i in verified)
    report = {
        "train": args.train, "model": args.model or "deepseek-v4-flash",
        "shots": args.shots, "n_tasks": len(rows),
        "n_verified": len(verified), "n_written": len(out_rows),
        "acceptance_rate": round(len(verified) / len(rows), 4),
        "acceptance_by_tier": {t: [acc_tier[t], by_tier[t]] for t in by_tier},
        "acceptance_by_domain": {d: [acc_dom[d], by_dom[d]] for d in by_dom},
        "rounds": rounds_report,
        "fail_reasons": dict(reasons_total.most_common()),
        "demo_rows_in_train": n_demo_in_train,
        "llm_usage": llm_stats(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {len(out_rows)} rows -> {out_path}")
    print(f"acceptance {len(verified)}/{len(rows)} = "
          f"{report['acceptance_rate']:.1%}; report -> {report_path}")


if __name__ == "__main__":
    main()
