"""Paired comparison of two test-result JSONs on the same 480-task split.

Both files come from eval.evaluate --save-generations; each has a `details`
list aligned by index with the shared test.jsonl order, so per-task outcomes
are naturally paired. Reports discordant counts and an exact two-sided
McNemar p-value (binomial test on discordant pairs), plus per-scenario splits.

Usage:
  python -m eval.paired_test results/A_test.json results/B_test.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load_success(path: str) -> tuple[list[bool], list[str]]:
    d = json.load(open(path))
    details = d["details"]
    return ([bool(x["success"]) for x in details],
            [x.get("scenario", "?") for x in details])


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: Binomial(n=b+c, 0.5) on the smaller side."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * cdf)


def compare(a_path: str, b_path: str) -> dict:
    a_ok, scen_a = load_success(a_path)
    b_ok, scen_b = load_success(b_path)
    assert len(a_ok) == len(b_ok), "result files cover different task sets"
    assert scen_a == scen_b, "scenario order mismatch; files not index-aligned"

    out = {"n": len(a_ok), "A": a_path, "B": b_path,
           "A_success": sum(a_ok), "B_success": sum(b_ok)}
    by_scen: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0])  # both, a_only, b_only, neither
    tot = [0, 0, 0, 0]
    for ok_a, ok_b, s in zip(a_ok, b_ok, scen_a):
        idx = 0 if (ok_a and ok_b) else 1 if ok_a else 2 if ok_b else 3
        by_scen[s][idx] += 1
        tot[idx] += 1

    both, a_only, b_only, _ = tot
    out["overall"] = {
        "both": both, "A_only": a_only, "B_only": b_only,
        "mcnemar_p_exact": exact_mcnemar(a_only, b_only),
    }
    out["per_scenario"] = {
        s: {"both": v[0], "A_only": v[1], "B_only": v[2],
            "mcnemar_p_exact": exact_mcnemar(v[1], v[2])}
        for s, v in sorted(by_scen.items())
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    print(json.dumps(compare(args.a, args.b), indent=2))


if __name__ == "__main__":
    main()
