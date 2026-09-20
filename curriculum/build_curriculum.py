"""Curriculum construction: stage split + within-stage Window Ordering + replay.

Stages (anti-forgetting design):
  stage1: T1 basics, no faults        -> XML syntax + action semantics
  stage2: T2 coordination, no faults  -> handover / co-transport / signals
  stage3: T3 + any faulted samples    -> Fallback recovery skills
Replay: stage2 mixes in 15% of stage1; stage3 mixes in 15% of stage1+2.

Within each stage: window_quantile ordering by DUE (DUCL paper's Window
Ordering), batch-aligned with training global batch.

Baselines produced for comparison:
  baseline_random.jsonl      - all data, shuffled (no curriculum)
  baseline_difficulty.jsonl  - all data, ascending difficulty only
  baseline_due_asc.jsonl     - all data, ascending DUE (flat, no stages)
Ablation:
  ours_stage{1,2,3}_noreplay.jsonl - same stages, replay disabled

Usage:
  python -m curriculum.build_curriculum --due outputs/curriculum/train_due.jsonl \
      --out outputs/curriculum --batch-size 16 --alpha 0.8
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} -> {path}")


def window_quantile(data: list, batch_size: int, alpha: float, seed: int,
                    key: str = "DUE") -> list:
    """DUCL Window Ordering (quantile variant): expanding easy->hard pool,
    uniform sampling inside the pool. Adapted from DUCL-main/scripts/data_order.py."""
    rng = random.Random(seed)
    sorted_data = sorted(data, key=lambda r: r[key])
    n = len(sorted_data)
    steps = math.ceil(n / batch_size)
    reach = max(1, math.ceil(alpha * steps))
    used = [False] * n
    out = []
    for s in range(steps):
        frac = min(1.0, 0.1 + 0.9 * (s + 1) / reach)
        pool_end = max(1, int(frac * n))
        pool = [i for i in range(pool_end) if not used[i]]
        if not pool:  # advance window
            pool = [i for i in range(n) if not used[i]]
        take = pool if len(pool) <= batch_size else rng.sample(pool, batch_size)
        for i in take:
            used[i] = True
            out.append(sorted_data[i])
    leftovers = [sorted_data[i] for i in range(n) if not used[i]]
    out += leftovers
    return out


def stage_of(rec: dict) -> int:
    meta = rec["meta"]
    has_fault = len(meta.get("faults", [])) > 0
    if has_fault or meta["tier"] == "T3":
        return 3
    return 1 if meta["tier"] == "T1" else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--due", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--replay-frac", type=float, default=0.15)
    ap.add_argument("--key", default="DUE",
                    help="score field used for within-stage Window Ordering "
                         "(e.g. DUE for embedding scores, E_DUE2 for "
                         "execution-grounded scores)")
    ap.add_argument("--shuffle-stages", action="store_true",
                    help="ablation: randomize stage membership, keeping stage sizes, "
                         "replay fraction and within-stage DUE window ordering identical; "
                         "isolates the effect of complexity-based staging")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = load_jsonl(args.due)
    out_dir = Path(args.out)

    stages = {1: [], 2: [], 3: []}
    for r in data:
        stages[stage_of(r)].append(r)
    print({k: len(v) for k, v in stages.items()})

    # replay mixing
    rep12 = rng.sample(stages[1], round(len(stages[1]) * args.replay_frac))
    rep3 = rng.sample(stages[1] + stages[2],
                     round((len(stages[1]) + len(stages[2])) * args.replay_frac))
    stage_sets = {1: stages[1], 2: stages[2] + rep12, 3: stages[3] + rep3}

    for s in (1, 2, 3):
        ordered = window_quantile(stage_sets[s], args.batch_size, args.alpha,
                                  args.seed + s, key=args.key)
        write_jsonl(out_dir / f"ours_stage{s}.jsonl", ordered)
        # no-replay ablation: same stage split, no replay mixing
        ordered_nr = window_quantile(stages[s], args.batch_size, args.alpha,
                                     args.seed + s, key=args.key)
        write_jsonl(out_dir / f"ours_stage{s}_noreplay.jsonl", ordered_nr)

    shuffled = list(data)
    rng.shuffle(shuffled)
    write_jsonl(out_dir / "baseline_random.jsonl", shuffled)
    # flat baselines need the legacy embedding-DUE fields; skip if absent
    if "difficulty" in data[0] and "DUE" in data[0]:
        write_jsonl(out_dir / "baseline_difficulty.jsonl",
                    sorted(data, key=lambda r: r["difficulty"]))
        write_jsonl(out_dir / "baseline_due_asc.jsonl", sorted(data, key=lambda r: r["DUE"]))

    if args.shuffle_stages:
        # same stage sizes / replay / window ordering, but stage membership is
        # a uniform random draw -- the only variable vs ours is complexity staging
        sizes = {s: len(stages[s]) for s in (1, 2, 3)}
        perm = list(data)
        rng.shuffle(perm)
        sh = {1: perm[:sizes[1]],
              2: perm[sizes[1]:sizes[1] + sizes[2]],
              3: perm[sizes[1] + sizes[2]:]}
        rep12s = rng.sample(sh[1], round(len(sh[1]) * args.replay_frac))
        rep3s = rng.sample(sh[1] + sh[2],
                           round((len(sh[1]) + len(sh[2])) * args.replay_frac))
        shuffled_sets = {1: sh[1], 2: sh[2] + rep12s, 3: sh[3] + rep3s}
        for s in (1, 2, 3):
            ordered = window_quantile(shuffled_sets[s], args.batch_size, args.alpha,
                                      args.seed + 100 + s, key=args.key)
            write_jsonl(out_dir / f"ours_stage{s}_shuffled.jsonl", ordered)


if __name__ == "__main__":
    main()
