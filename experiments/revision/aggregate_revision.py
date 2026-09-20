"""Aggregate revision evaluation JSONs with paired, seed-aware statistics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _rate(payload: dict) -> float:
    for key in ("strict_success_rate", "exec_success_rate"):
        if key in payload:
            return float(payload[key])
    overall = payload.get("overall", {})
    for key in ("pass@1", "pass@8"):
        if key in overall:
            return float(overall[key])
    raise ValueError("evaluation JSON has no recognized success-rate field")


def _group(row: dict, subgroup: str) -> bool:
    if subgroup == "overall":
        return True
    if subgroup == "coordination":
        return row.get("tier") in {"T2", "T3"}
    if subgroup == "T1":
        return row.get("tier") == "T1"
    if subgroup == "T2":
        return row.get("tier") == "T2"
    if subgroup == "T3":
        return row.get("tier") == "T3"
    if subgroup == "new_primitive":
        return "+service" in str(row.get("scenario", ""))
    if subgroup == "shared_primitive":
        return "+service" not in str(row.get("scenario", ""))
    raise ValueError(f"unknown subgroup {subgroup}")


def _details(payload: dict) -> list[dict]:
    details = payload.get("details")
    if not isinstance(details, list) or not details:
        raise ValueError("evaluation JSON lacks non-empty details")
    return details


def _mcnemar(a: list[bool], b: list[bool]) -> float:
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum((not x) and y for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = min(a_only, b_only)
    cdf = sum(math.comb(n, k) for k in range(tail + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * cdf)


def _paired(a_payload: dict, b_payload: dict, bootstrap_samples: int,
            seed: int, subgroup: str = "overall") -> dict:
    left, right = _details(a_payload), _details(b_payload)
    by_id_left = {row.get("record_id"): row for row in left}
    by_id_right = {row.get("record_id"): row for row in right}
    if None in by_id_left or None in by_id_right:
        raise ValueError("paired aggregation requires stable record_id fields")
    if set(by_id_left) != set(by_id_right):
        raise ValueError("paired result task IDs do not match")
    ids = sorted(by_id_left)
    ids = [rid for rid in ids if _group(by_id_left[rid], subgroup)]
    if not ids:
        return {"n": 0, "subgroup": subgroup, "p_exact": 1.0}
    a = np.asarray([bool(by_id_left[rid].get("success")) for rid in ids], dtype=np.int8)
    b = np.asarray([bool(by_id_right[rid].get("success")) for rid in ids], dtype=np.int8)
    rng = np.random.default_rng(seed)
    samples = max(0, int(bootstrap_samples))
    if samples:
        indices = rng.integers(0, len(a), size=(samples, len(a)))
        deltas = (b[indices] - a[indices]).mean(axis=1)
        ci = [float(value) for value in np.quantile(deltas, [0.025, 0.975])]
    else:
        ci = None
    return {
        "n": len(a), "subgroup": subgroup,
        "a_success_rate": float(a.mean()), "b_success_rate": float(b.mean()),
        "delta_b_minus_a": float(b.mean() - a.mean()),
        "a_only": int(np.sum((a == 1) & (b == 0))),
        "b_only": int(np.sum((a == 0) & (b == 1))),
        "both_success": int(np.sum((a == 1) & (b == 1))),
        "both_fail": int(np.sum((a == 0) & (b == 0))),
        "p_exact": _mcnemar(a.tolist(), b.tolist()),
        "bootstrap_95_ci": ci,
        "bootstrap_samples": samples,
    }


def aggregate(paths: dict[str, list[tuple[int, Path]]], subgroups: tuple[str, ...]) -> dict:
    arms = {}
    for arm, values in sorted(paths.items()):
        per_seed = []
        for seed, path in sorted(values):
            payload = json.loads(path.read_text(encoding="utf-8"))
            details = _details(payload)
            rates = {}
            for subgroup in subgroups:
                selected = [row for row in details if _group(row, subgroup)]
                rates[subgroup] = (sum(bool(row.get("success")) for row in selected)
                                   / max(1, len(selected)))
            per_seed.append({"seed": seed, "path": str(path), "n": len(details),
                             "rates": rates})
        for subgroup in subgroups:
            values_for_group = [row["rates"][subgroup] for row in per_seed]
            if values_for_group:
                arms.setdefault(arm, {})[subgroup] = {
                    "mean": float(np.mean(values_for_group)),
                    "std": float(np.std(values_for_group, ddof=1)) if len(values_for_group) > 1 else 0.0,
                    "n_seeds": len(values_for_group),
                    "per_seed": values_for_group,
                }
        arms.setdefault(arm, {})["runs"] = per_seed
    return arms


def discover(root: str | Path) -> dict[str, list[tuple[int, Path]]]:
    root = Path(root)
    paths: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(root.glob("seed*/eval/*_test.json")):
        match = re.fullmatch(r"seed(\d+)", path.parent.parent.name)
        if not match:
            continue
        arm = path.stem[:-5] if path.stem.endswith("_test") else path.stem
        paths[arm].append((int(match.group(1)), path))
    return dict(paths)


def _holm(pairs: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pairs.items(), key=lambda item: item[1])
    adjusted = {}
    previous = 0.0
    m = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        value = min(1.0, max(previous, (m - index) * pvalue))
        adjusted[name] = value
        previous = value
    return adjusted


def _seed_gap_summary(entries: list[dict], subgroup: str) -> dict:
    """Summarize paired deltas across seeds without treating tasks as seeds."""
    deltas = [float(entry["delta_b_minus_a"]) for entry in entries
              if entry.get("subgroup") == subgroup and entry.get("n", 0)]
    if not deltas:
        return {"subgroup": subgroup, "n_seeds": 0, "mean": None,
                "std": None, "per_seed": []}
    return {
        "subgroup": subgroup,
        "n_seeds": len(deltas),
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "per_seed": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="revision root containing seed*/eval/*_test.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", default=None,
                        help="arm used as the left side of paired deltas; "
                             "default is spcl for attribution roots, otherwise flat")
    args = parser.parse_args()
    paths = discover(args.root)
    if not paths:
        raise FileNotFoundError(f"no seed*/eval/*_test.json files under {args.root}")
    subgroups = ("overall", "T1", "T2", "T3", "coordination",
                 "shared_primitive", "new_primitive")
    arms = aggregate(paths, subgroups)
    baseline = args.baseline or ("spcl" if "spcl" in paths and "flat" not in paths
                                 else "flat" if "flat" in paths else sorted(paths)[0])
    if baseline not in paths:
        raise ValueError(f"requested baseline {baseline!r} is absent; available={sorted(paths)}")
    paired = {}
    paired_summary = {}
    pvalues = {}
    for arm in sorted(paths):
        if arm == baseline:
            continue
        common = sorted(set(seed for seed, _ in paths[baseline]) &
                        set(seed for seed, _ in paths[arm]))
        entries = {}
        for seed in common:
            left = dict(paths[baseline])[seed]
            right = dict(paths[arm])[seed]
            left_payload = json.loads(left.read_text(encoding="utf-8"))
            right_payload = json.loads(right.read_text(encoding="utf-8"))
            entries[str(seed)] = {
                subgroup: _paired(left_payload, right_payload,
                                  args.bootstrap_samples, args.seed + seed + i,
                                  subgroup)
                for i, subgroup in enumerate(subgroups)
            }
            pvalues[f"{arm}/seed{seed}"] = entries[str(seed)]["overall"]["p_exact"]
        paired[arm] = entries
        paired_summary[arm] = {
            subgroup: _seed_gap_summary(
                [entry[subgroup] for entry in entries.values()], subgroup)
            for subgroup in subgroups
        }
    result = {
        "protocol_version": "iclr2027_revision_v1",
        "root": str(args.root), "baseline_for_pairing": baseline,
        "arms": arms, "paired_vs_baseline": paired,
        "paired_delta_summary": paired_summary,
        "holm_adjusted_p": _holm(pvalues),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "subgroup", "mean", "std", "n_seeds", "per_seed"))
        for arm, groups in sorted(arms.items()):
            for subgroup in subgroups:
                if subgroup not in groups:
                    continue
                row = groups[subgroup]
                writer.writerow((arm, subgroup, row["mean"], row["std"],
                                 row["n_seeds"], json.dumps(row["per_seed"])))
    print(json.dumps({"out": str(out), "csv": str(csv_path),
                      "arms": sorted(arms), "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()
