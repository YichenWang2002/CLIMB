"""Deterministic SPCL bucket-weight schedule derivation.

Implements the construction documented in
paper-kimi/iclr2027/spcl_appendix.tex ("SPCL Schedule Derivation: Arithmetic")
and used by all reported Llama / DeepSeek SPCL runs:

  1. R rounds are fixed by compute parity with the flat baseline; K equal-mass
     difficulty quartile buckets are the coarsest split keeping >= 1500 samples.
  2. Expanding windows: w_r = min(K, ceil(K*r/R))   (mass budget share r/R).
  3. Continuous exposure targets, from per-bucket development-side means
     Dbar_b (fused difficulty) and Ubar_b (validation-referenced utility):
        c_{r,b} = 1                                       , r = 1
                = (1/Ubar_b) / min_{j<w_r}(1/Ubar_j)      , 1 < r < R
                = R * (Dbar_b/Ubar_b) / max_j(Dbar_j/Ubar_j), r = R
  4. Protocol-induced projection onto the lattice forced by K and the
     remaining-round budget H = R - r + 1:
        Q_K(x)   = max(1/K, round(K x)/K)        consolidated buckets
        Q_H^+(x) = ceil(H x)/H                   newly opened frontier bucket
        m_{r,b}  = 1 if r == 1 else (Q_H^+ / Q_K per bucket class)

No evaluation outcome enters any step.  Re-running on the seed-42
development statistics reproduces the vectors used by the reported runs:
    m_1 = (1, 1), m_2 = (1, 1, 1.5), m_3 = (0.5, 0.75, 1.5, 3).

Usage:
    python -m curriculum.derive_spcl_schedule --R 3 --K 4 \
        --Dbar 0.190,0.374,0.603,0.833 --Ubar 0.576,0.592,0.479,0.354
"""
from __future__ import annotations

import argparse
import json
from math import ceil


def window_ladder(R: int, K: int) -> list[int]:
    return [min(K, ceil(K * r / R)) for r in range(1, R + 1)]


def q_k(x: float, K: int) -> float:
    return max(1.0 / K, round(K * x) / K)


def q_h_plus(x: float, H: int) -> float:
    return ceil(H * x) / H


def continuous_targets(R, K, Dbar, Ubar):
    """Return c[r][b] for b < w_r (0-indexed rounds/buckets)."""
    w = window_ladder(R, K)
    c = {}
    for r in range(1, R + 1):
        wr = w[r - 1]
        if r == 1:
            c[r] = [1.0] * wr
        elif r < R:
            inv = [1.0 / Ubar[b] for b in range(wr)]
            lo = min(inv)
            c[r] = [x / lo for x in inv]
        else:
            ratio = [Dbar[b] / Ubar[b] for b in range(wr)]
            hi = max(ratio)
            c[r] = [R * x / hi for x in ratio]
    return c


def realized_multipliers(R, K, Dbar, Ubar):
    """Project continuous targets onto the protocol lattice; frontier bucket
    (the one opened at round r) uses the one-sided H projection."""
    w = window_ladder(R, K)
    c = continuous_targets(R, K, Dbar, Ubar)
    m = {}
    for r in range(1, R + 1):
        wr = w[r - 1]
        frontier = w[r - 2] if r > 1 else 0          # buckets < frontier were open before
        row = []
        for b in range(wr):
            if r == 1:
                row.append(1.0)
            elif b < frontier:
                row.append(q_k(c[r][b], K))
            else:
                row.append(q_h_plus(c[r][b], R - r + 1))
        m[r] = row
    return c, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--Dbar", default="0.190,0.374,0.603,0.833",
                    help="per-bucket mean fused difficulty (development side)")
    ap.add_argument("--Ubar", default="0.576,0.592,0.479,0.354",
                    help="per-bucket mean utility (validation-referenced)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args()
    D = [float(x) for x in a.Dbar.split(",")]
    U = [float(x) for x in a.Ubar.split(",")]
    c, m = realized_multipliers(a.R, a.K, D, U)
    print("window ladder w   =", window_ladder(a.R, a.K))
    for r in sorted(c):
        print("c_%d (continuous) = %s" % (r, [round(x, 4) for x in c[r]]))
        print("m_%d (projected)  = %s" % (r, m[r]))
    expected = {1: [1.0, 1.0], 2: [1.0, 1.0, 1.5], 3: [0.5, 0.75, 1.5, 3.0]}
    ok = all(m[r] == expected[r] for r in expected if r in m)
    print("reproduces reported schedule:", ok)
    if a.out:
        json.dump({"window": window_ladder(a.R, a.K),
                   "continuous": {str(k): v for k, v in c.items()},
                   "multipliers": {str(k): v for k, v in m.items()},
                   "matches_reported_runs": ok},
                  open(a.out, "w"), indent=2)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
