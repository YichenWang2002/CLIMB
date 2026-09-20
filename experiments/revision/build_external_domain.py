"""Generate a genuinely held-out external-domain evaluation split.

The kitchen domain is registered in ``datagen.strips.domains`` but is not in
the normal train/validation/test quotas. This script samples valid STRIPS
tasks, compiles and executor-checks gold BTs, then renders either a
deterministic prose template (default, reproducible) or the same NL generation
service used by the original benchmark (``--nl-mode api``).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datagen.build_dataset import serialize_task, task_key, worker
from datagen.nl_gen import INSTRUCTION, generate_nl_inputs
from datagen.names import (SCENE_EN, item_en, loc_en, robot_en,
                           UNIQUE_ACTION_EN)
from .protocol import load_jsonl, sha256_file, validate_external_split, write_json


def _template_input(task: dict) -> str:
    domain = task["domain"]
    robots = task["robots"]
    starts = [f"{robot_en(f[1])} ({f[1]}) starts at {loc_en(domain, f[2])} ({f[2]})"
              for f in task["init_dynamic"] if f[0] == "at"]
    objects = [f"{item_en(domain, f[1])} ({f[1]}) is at {loc_en(domain, f[2])} ({f[2]})"
               for f in task["init_dynamic"] if f[0] == "item_at"]
    goals = []
    for fact in task["goal"]:
        if fact[0] == "item_at":
            goals.append(f"bring {item_en(domain, fact[1])} ({fact[1]}) to "
                         f"{loc_en(domain, fact[2])} ({fact[2]})")
        elif fact[0] == "sanitized":
            goals.append(f"sanitize the counter at {loc_en(domain, fact[1])} ({fact[1]})")
    layout = []
    seen = set()
    for source, target in task["connected"]:
        key = tuple(sorted((source, target)))
        if key in seen:
            continue
        seen.add(key)
        layout.append(f"{loc_en(domain, source)} ({source}) connects to "
                      f"{loc_en(domain, target)} ({target})")
    fault_text = []
    for fault in task.get("faults", []):
        if fault["type"] == "blocked_edge":
            a, b = fault["edge"]
            fault_text.append(f"the passage between {a} and {b} may be blocked and should be cleared")
        elif fault["type"] == "battery":
            fault_text.append(f"{fault['robot']} has a battery limit of {fault['budget']} moves and may need to recharge")
        elif fault["type"] == "handover_fail":
            fault_text.append("the first handover may fail and should be retried")
    extra = ""
    if any(fact[0] == "sanitized" for fact in task["goal"]):
        extra = " The team also has " + UNIQUE_ACTION_EN["SanitizeCounter"] + "."
    faults = (" Hazards: " + "; ".join(fault_text) + ".") if fault_text else ""
    return (
        f"This mission takes place in {SCENE_EN[domain]}. "
        f"The robot team consists of {', '.join(robot_en(r) + ' (' + r + ')' for r in robots)}. "
        f"{'. '.join(starts)}. {'. '.join(objects)}. "
        f"Their goals are to {'; '.join(goals)}. "
        f"The layout is described by: {'; '.join(layout)}. "
        f"The charging station is {task['charge_stations'][0]}."
        f"{faults}{extra}"
    )


def _records(tasks: list[dict], nl_mode: str, workers: int) -> list[dict]:
    if nl_mode == "api":
        nls = generate_nl_inputs(tasks, max_workers=workers)
    else:
        nls = [_template_input(task) for task in tasks]
    result = []
    for task, nl in zip(tasks, nls):
        if not nl:
            raise RuntimeError("external-domain NL generation returned an empty item")
        meta = {key: value for key, value in task.items() if key != "_xml"}
        result.append({"instruction": INSTRUCTION, "input": nl,
                       "output": task["_xml"], "meta": meta})
    return result


def _quotas(total: int) -> dict[str, int]:
    raw = {"T1": total * 0.20, "T2": total * 0.40, "T3": total * 0.40}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: raw[item] - result[item], reverse=True):
        if sum(result.values()) < total:
            result[key] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/revision/external_domain/kitchen.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--train", default="outputs/dataset/train_aug10.jsonl")
    parser.add_argument("--val", default="outputs/dataset/val.jsonl")
    parser.add_argument("--n", type=int, default=300,
                        help="total external tasks; tier mix is 20/40/40")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--workers", type=int, default=1,
                        help="sampling workers; NL API workers are separate")
    parser.add_argument("--nl-workers", type=int, default=8)
    parser.add_argument("--nl-mode", choices=("template", "api"), default="template")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    quotas = _quotas(args.n)
    out = Path(args.out)
    if out.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to overwrite existing external split {out}")
    print(json.dumps({"domain": "kitchen", "quotas": quotas, "seed": args.seed,
                      "nl_mode": args.nl_mode, "out": str(out),
                      "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return

    tasks = []
    seen = set()
    for offset, (tier, count) in enumerate(quotas.items()):
        got = worker("kitchen", tier, count, args.seed * 1000 + offset * 97)
        if len(got) != count:
            raise RuntimeError(f"sampler produced {len(got)}/{count} kitchen/{tier} tasks")
        for task in got:
            key = task_key(task)
            if key in seen:
                raise RuntimeError("duplicate task key in external split")
            seen.add(key)
            tasks.append(task)
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    records = _records(tasks, args.nl_mode, args.nl_workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    validation = validate_external_split(out, args.train, args.val, expected_domain="kitchen")
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    report = {
        "protocol_version": "iclr2027_revision_v1",
        "domain": "kitchen", "unique_primitive": "SanitizeCounter",
        "n": len(records), "quotas": quotas, "seed": args.seed,
        "nl_mode": args.nl_mode, "sha256": sha256_file(out),
        "split_validation": validation,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

