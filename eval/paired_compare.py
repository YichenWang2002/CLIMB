"""Paired execution-success comparison with exact McNemar and bootstrap CI."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _exact_mcnemar(flat_only: int, method_only: int) -> float:
    discordant = flat_only + method_only
    if discordant == 0:
        return 1.0
    tail = min(flat_only, method_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def compare_results(flat: dict, method: dict, bootstrap_samples: int = 100_000,
                    seed: int = 42) -> dict:
    flat_details = flat.get("details", [])
    method_details = method.get("details", [])
    if len(flat_details) != len(method_details) or not flat_details:
        raise ValueError("results must contain aligned non-empty details")
    identity_fields = ("domain", "tier", "scenario")
    for index, (left, right) in enumerate(zip(flat_details, method_details)):
        if bool(left.get("record_id")) != bool(right.get("record_id")):
            raise ValueError("only one result contains stable record IDs")
        if left.get("record_id") and left["record_id"] != right["record_id"]:
            raise ValueError(f"record ID mismatch at row {index}")
        if any(left.get(field) != right.get(field) for field in identity_fields):
            raise ValueError(f"result alignment mismatch at row {index}")
    a = np.asarray([bool(row["success"]) for row in flat_details], dtype=np.int8)
    b = np.asarray([bool(row["success"]) for row in method_details], dtype=np.int8)
    flat_only = int(np.sum((a == 1) & (b == 0)))
    method_only = int(np.sum((a == 0) & (b == 1)))
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap_samples, dtype=np.float32)
    cursor = 0
    while cursor < bootstrap_samples:
        size = min(2_000, bootstrap_samples - cursor)
        indices = rng.integers(0, len(a), size=(size, len(a)))
        differences[cursor:cursor + size] = (b[indices] - a[indices]).mean(axis=1)
        cursor += size
    return {
        "n": len(a),
        "flat_success": int(a.sum()),
        "flat_success_rate": float(a.mean()),
        "method_success": int(b.sum()),
        "method_success_rate": float(b.mean()),
        "absolute_difference": float(b.mean() - a.mean()),
        "both_success": int(np.sum((a == 1) & (b == 1))),
        "flat_only": flat_only,
        "method_only": method_only,
        "both_fail": int(np.sum((a == 0) & (b == 0))),
        "mcnemar_exact_two_sided_p": _exact_mcnemar(flat_only, method_only),
        "paired_bootstrap_95_ci": [float(value) for value in
                                   np.quantile(differences, [0.025, 0.975])],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def compare_paths(flat_path: Path, method_path: Path,
                  bootstrap_samples: int = 100_000,
                  seed: int = 42) -> dict:
    flat = json.loads(Path(flat_path).read_text())
    method = json.loads(Path(method_path).read_text())
    return compare_results(flat, method, bootstrap_samples, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = compare_paths(Path(args.flat), Path(args.method),
                           args.bootstrap_samples, args.seed)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
