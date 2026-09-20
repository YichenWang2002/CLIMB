"""Paired comparison for two strict behavior-tree evaluation files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_mcnemar(flat_only: int, method_only: int) -> float:
    discordant = flat_only + method_only
    if discordant == 0:
        return 1.0
    tail = min(flat_only, method_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_bootstrap_delta_ci(pairs: list[tuple[bool, bool]], resamples: int = 10000,
                              seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the success-rate difference over resampled tasks."""
    import random
    rng = random.Random(seed)
    n = len(pairs)
    deltas = []
    for _ in range(resamples):
        method_hits = flat_hits = 0
        for _ in range(n):
            flat_ok, method_ok = pairs[rng.randrange(n)]
            flat_hits += int(flat_ok)
            method_hits += int(method_ok)
        deltas.append((method_hits - flat_hits) / n)
    deltas.sort()
    return (deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1])


def rate(group: dict) -> float:
    return group["success"] / group["n"] if group["n"] else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="paired bootstrap resamples for the delta CI; 0 disables")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    args = ap.parse_args()
    flat = json.loads(Path(args.flat).read_text(encoding="utf-8"))
    method = json.loads(Path(args.method).read_text(encoding="utf-8"))
    if flat["n"] != method["n"] or len(flat["details"]) != len(method["details"]):
        raise ValueError("evaluation sizes do not match")

    flat_ids = [row.get("record_id") for row in flat["details"]]
    method_ids = [row.get("record_id") for row in method["details"]]
    if any(value is None for value in flat_ids + method_ids):
        raise ValueError("evaluation details lack stable record_id; rerun strict_eval")
    if flat_ids != method_ids:
        raise ValueError("paired evaluation record order does not match")

    pairs = [(bool(a["success"]), bool(b["success"]))
             for a, b in zip(flat["details"], method["details"])]
    both_success = sum(a and b for a, b in pairs)
    flat_only = sum(a and not b for a, b in pairs)
    method_only = sum(not a and b for a, b in pairs)
    both_fail = sum(not a and not b for a, b in pairs)

    def grouped_delta(name: str) -> dict:
        keys = sorted(set(flat[name]) | set(method[name]))
        return {key: {
            "flat_rate": rate(flat[name][key]),
            "method_rate": rate(method[name][key]),
            "delta_pp": 100.0 * (rate(method[name][key]) - rate(flat[name][key])),
        } for key in keys}

    result = {
        "n": flat["n"],
        "flat_rate": flat["strict_success_rate"],
        "method_rate": method["strict_success_rate"],
        "delta_pp": 100.0 * (method["strict_success_rate"] - flat["strict_success_rate"]),
        "paired": {"both_success": both_success, "flat_only": flat_only,
                   "method_only": method_only, "both_fail": both_fail,
                   "discordant": flat_only + method_only,
                   "mcnemar_exact_p": exact_mcnemar(flat_only, method_only)},
        "per_domain": grouped_delta("per_domain"),
        "per_tier": grouped_delta("per_tier"),
        "per_bucket": grouped_delta("per_bucket"),
        "mean_generation_latency_s": {
            "flat": flat["mean_generation_latency_s"],
            "method": method["mean_generation_latency_s"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bootstrap > 0:
        ci_low, ci_high = paired_bootstrap_delta_ci(
            pairs, resamples=args.bootstrap, seed=args.bootstrap_seed)
        result["paired"]["delta_bootstrap_95ci"] = [ci_low, ci_high]
        result["paired"]["bootstrap_resamples"] = args.bootstrap
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

