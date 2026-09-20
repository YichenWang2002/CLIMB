#!/usr/bin/env python3
"""Aggregate the reviewer-requested Llama P0 evaluations when they exist.

The attribution reruns live under outputs/revision/attribution_v2. Historical
files are kept separate from reruns because their curriculum sidecars are not
bit-identical. EGVD stores pass@k in its top-level aggregate and uses
``succ_at`` in per-task details, unlike the strict evaluator's ``success``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from typing import Any


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def strict_metric(data: dict[str, Any]) -> dict[str, Any]:
    tiers = data.get("per_tier", {})
    out: dict[str, Any] = {
        "n": int(data.get("n", 0)),
        "overall": float(data.get("exec_success_rate", data.get("strict_success_rate", math.nan))),
    }
    for tier, label in (("T1", "single"), ("T2", "seq"), ("T3", "joint")):
        item = tiers.get(tier, {})
        n = int(item.get("n", 0))
        ok = int(item.get("ok", item.get("success", 0)))
        out[label] = {"n": n, "ok": ok, "rate": (ok / n if n else math.nan)}
    return out


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for metric in ("overall", "single", "seq", "joint"):
        vals = [float(r[key][metric] if metric == "overall" else r[key][metric]["rate"]) for r in rows]
        result[metric] = {
            "mean": statistics.mean(vals) if vals else None,
            "std": sample_std(vals),
            "values": vals,
        }
    return result


def aggregate_controls(root: str) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {"anti": [], "hardtail": []}
    for path in sorted(glob.glob(os.path.join(root, "seed*", "eval", "*_test.json"))):
        name = os.path.basename(path)
        arm = name.removesuffix("_test.json")
        if arm not in arms:
            continue
        seed_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = int(seed_name.removeprefix("seed"))
        arms[arm].append({"seed": seed, "path": path, "metrics": strict_metric(read_json(path))})
    output: dict[str, Any] = {}
    for arm, rows in arms.items():
        output[arm] = {
            "runs": rows,
            "summary": summarize(rows, "metrics") if rows else {"n_seeds": 0, "seeds": []},
        }
    return output


def egvd_metric(data: dict[str, Any]) -> dict[str, Any]:
    overall = data.get("overall", {})
    per_tier = data.get("per_tier", {})
    out: dict[str, Any] = {
        "n": int(data.get("n", 0)),
        "pass_at_1": float(overall.get("pass@1", math.nan)),
        "pass_at_2": float(overall.get("pass@2", math.nan)),
        "pass_at_4": float(overall.get("pass@4", math.nan)),
        "pass_at_8": float(overall.get("pass@8", math.nan)),
    }
    for tier, label in (("T1", "single"), ("T2", "seq"), ("T3", "joint")):
        item = per_tier.get(tier, {})
        out[label] = {f"pass_at_{k}": float(item.get(f"pass@{k}", math.nan)) for k in (1, 2, 4, 8)}
    return out


def aggregate_egvd(results_dir: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    patterns = [
        os.path.join(results_dir, "egvd_v2_s*_rerun_constrained_k8_t07.json"),
        os.path.join(results_dir, "egvd_v2_constrained_k8_t07.json"),
        os.path.join(results_dir, "egvd_v2_s43_constrained_k8_t07.json"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    for path in sorted(set(paths)):
        data = read_json(path)
        basename = os.path.basename(path)
        if "_rerun_" in basename:
            seed_text = basename.split("_rerun_", 1)[0].removeprefix("egvd_v2_s")
            source = "rerun"
        elif basename.startswith("egvd_v2_s43_"):
            seed_text, source = "43", "historical_s43"
        else:
            seed_text, source = str(data.get("seed", "42")), "historical_s42"
        rows.append({"seed": int(seed_text), "source": source, "path": path, "metrics": egvd_metric(data)})
    # Summaries are intentionally split: historical and rerun sources are not
    # treated as paired observations.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["source"], []).append(row)
    summary: dict[str, Any] = {}
    for source, source_rows in groups.items():
        values: dict[str, Any] = {"n_seeds": len(source_rows), "seeds": [r["seed"] for r in source_rows]}
        for metric in ("pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8"):
            vals = [r["metrics"][metric] for r in source_rows]
            values[metric] = {"mean": statistics.mean(vals), "std": sample_std(vals), "values": vals}
        summary[source] = values
    return {"runs": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="pipeline/outputs/revision/attribution_v2")
    parser.add_argument("--results-dir", default="pipeline/results/outputs")
    parser.add_argument("--out", default="pipeline/outputs/revision/attribution_v2/p0_summary.json")
    args = parser.parse_args()
    report = {"controls": aggregate_controls(args.root), "egvd": aggregate_egvd(args.results_dir)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
