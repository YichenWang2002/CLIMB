"""Build multi-seed SPCL attribution controls from one scoring pass.

Controls follow the paper's pre-specified battery:

``mixs``
    Shuffle the realized three-round SPCL multiset and repartition it. Every
    row, including duplicate draws, is retained exactly once in the control.
``hardtail``
    Sample the full pool in every round with the final-round hard-tail boost.
``anti``
    Use the hardest prefix first and mirror the SPCL boost profile.
``nou1``
    Optional utility ablation, retained for backwards-compatible replication.

The builder consumes ``spcl_scores.jsonl`` produced by build_spcl.py, so these
controls never silently rescore the data with a different model or seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .protocol import load_jsonl, row_id, sha256_file, write_json

V2_BOOSTS = (
    (1.0, 1.0),
    (1.0, 1.0, 1.5),
    (0.5, 0.75, 1.5, 3.0),
)
HARDTAIL_BOOSTS = (0.5, 0.75, 1.5, 3.0)


def _multiset_hash(rows: list[dict]) -> str:
    counts = Counter(row_id(row) for row in rows)
    payload = "\n".join(f"{key} {counts[key]}" for key in sorted(counts))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _load_scores(train_path: str | Path, score_path: str | Path) -> tuple[list[dict], list[dict]]:
    train = load_jsonl(train_path)
    scores = load_jsonl(score_path)
    if len(train) != len(scores):
        raise ValueError(f"score sidecar length {len(scores)} != train length {len(train)}")
    for i, (row, score) in enumerate(zip(train, scores)):
        expected = row_id(row)
        if score.get("spcl_record_id") != expected:
            raise ValueError(f"score sidecar identity mismatch at row {i}")
        for key in ("difficulty", "utility", "bucket"):
            if key not in score:
                raise ValueError(f"score sidecar row {i} lacks {key}")
    return train, scores


def _write_rounds(out_dir: Path, rounds: list[list[dict]], method: str,
                  source_dir: Path, seed: int, extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, rows in enumerate(rounds, 1):
        path = out_dir / f"round{index}.jsonl"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing curriculum file {path}")
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        files.append(str(path))
    report = {
        "protocol_version": "iclr2027_revision_v1",
        "method": method,
        "seed": seed,
        "source_spcl_dir": str(source_dir),
        "source_round_hashes": {
            path.name: sha256_file(path)
            for path in sorted(source_dir.glob("round*.jsonl"))
        },
        "round_files": files,
        "round_sizes": [len(rows) for rows in rounds],
        "round_tier_counts": [dict(Counter(r.get("meta", {}).get("tier", "?") for r in rows))
                              for rows in rounds],
    }
    if extra:
        report.update(extra)
    report["realized_multiset_sha256"] = _multiset_hash([row for rows in rounds for row in rows])
    write_json(out_dir / "report.json", report)
    return report


def build_mixs(train: list[dict], source_dir: Path, out_dir: Path,
               seed: int, rounds: int = 3) -> dict:
    source_rounds = []
    for index in range(1, rounds + 1):
        path = source_dir / f"round{index}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        source_rounds.append(load_jsonl(path))
    source_rows = [row for rows in source_rounds for row in rows]
    train_ids = {row_id(row) for row in train}
    if any(row_id(row) not in train_ids for row in source_rows):
        raise ValueError("SPCL round contains a row absent from the declared train split")
    source_multiset = _multiset_hash(source_rows)
    rng = random.Random(seed + 7001)
    shuffled = list(source_rows)
    rng.shuffle(shuffled)
    n = len(source_rounds[0])
    if any(len(rows) != n for rows in source_rounds):
        raise ValueError("SPCL rounds must have equal sizes for mixs repartitioning")
    result = _write_rounds(
        out_dir, [shuffled[index * n:(index + 1) * n] for index in range(rounds)],
        "mixs", source_dir, seed,
        {"source_realized_multiset_sha256": source_multiset,
         "exact_multiset_preserved": source_multiset == _multiset_hash(shuffled),
         "control_definition": "shuffle realized SPCL multiset, then repartition"},
    )
    if not result["exact_multiset_preserved"]:
        raise AssertionError("mixs changed the realized multiset")
    return result


def _sample_weighted(train: list[dict], scores: list[dict], bucket_count: int,
                     pool: list[int], utility_enabled: bool, multipliers: list[float],
                     seed: int, n: int) -> list[dict]:
    if not pool:
        raise ValueError("empty attribution pool")
    weights = []
    for index in pool:
        utility = float(scores[index]["utility"]) if utility_enabled else 1.0
        bucket = int(scores[index]["bucket"])
        if bucket < 0 or bucket >= bucket_count:
            raise ValueError(f"bucket {bucket} outside [0, {bucket_count})")
        weights.append(max(1e-12, utility * float(multipliers[bucket])))
    total = sum(weights)
    rng = random.Random(seed)
    cumulative = []
    running = 0.0
    for index, weight in zip(pool, weights):
        running += weight / total
        cumulative.append((running, index))
    out = []
    for _ in range(n):
        draw = rng.random()
        for cutoff, index in cumulative:
            if draw <= cutoff:
                out.append(train[index])
                break
        else:
            out.append(train[cumulative[-1][1]])
    rng.shuffle(out)
    return out


def build_weighted_control(train: list[dict], scores: list[dict], source_dir: Path,
                           out_dir: Path, method: str, seed: int, rounds: int,
                           bucket_count: int, windows: tuple[int, ...] = (2, 3, 4),
                           boost_profiles: tuple[tuple[float, ...], ...] = V2_BOOSTS) -> dict:
    n = len(train)
    if len(windows) < rounds or len(boost_profiles) < rounds:
        raise ValueError("attribution schedule must provide one window/profile per round")
    windows = tuple(int(x) for x in windows[:rounds])
    boost_profiles = tuple(tuple(float(x) for x in p) for p in boost_profiles[:rounds])
    for r, (window, profile) in enumerate(zip(windows, boost_profiles)):
        if not 0 < window <= bucket_count:
            raise ValueError(f"round {r + 1}: window {window} outside bucket count {bucket_count}")
        if len(profile) != window:
            raise ValueError(f"round {r + 1}: {len(profile)} boosts for window {window}")
    if rounds > len(windows):
        raise ValueError("the pre-specified attribution schedule has three rounds")
    if method == "hardtail":
        # Repeat the final-round profile over the full pool.  This preserves
        # the hard-tail dosage while removing the competence-pacing schedule.
        final_profile = list(boost_profiles[-1])
        round_multipliers = []
        for _ in range(rounds):
            full = [1.0] * bucket_count
            for bucket in range(len(final_profile)):
                full[bucket] = final_profile[bucket]
            round_multipliers.append(full)
        pools = [list(range(n))] * rounds
        definition = "full pool every round with static final-round boosts"
    elif method in {"anti", "nou1"}:
        round_multipliers = []
        pools = []
        for round_index, window in enumerate(windows[:rounds]):
            if method == "anti":
                pools.append([i for i, score in enumerate(scores)
                              if int(score["bucket"]) >= bucket_count - window])
            else:
                pools.append([i for i, score in enumerate(scores)
                              if int(score["bucket"]) < window])
            full = [1.0] * bucket_count
            for bucket in range(window):
                if method == "anti":
                    full[bucket_count - 1 - bucket] = boost_profiles[round_index][bucket]
                else:
                    full[bucket] = boost_profiles[round_index][bucket]
            round_multipliers.append(full)
        definition = "hardest prefix first with mirrored boosts" if method == "anti" \
            else "SPCL windows and boosts with utility set to one"
    else:
        raise ValueError(f"unknown weighted control {method}")
    if len(round_multipliers) != rounds:
        raise ValueError("rounds exceeds the pre-specified boost schedule")
    round_rows = []
    for index in range(rounds):
        multipliers = round_multipliers[index]
        round_rows.append(_sample_weighted(
            train, scores, bucket_count, pools[index], method != "nou1",
            multipliers, seed + index, n,
        ))
    return _write_rounds(
        out_dir, round_rows, method, source_dir, seed,
        {"control_definition": definition,
         "utility_enabled": method != "nou1",
         "bucket_count": bucket_count,
         "windows": list(windows),
         "boost_profiles": [list(p) for p in boost_profiles],
         "sampling_with_replacement": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mixs", "hardtail", "anti", "nou1"), required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--spcl-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--buckets", type=int, default=4)
    parser.add_argument("--windows", default="2,3,4",
                        help="comma-separated competence windows (default: 2,3,4)")
    parser.add_argument("--boosts", default="1,1|1,1,1.5|0.5,0.75,1.5,3.0",
                        help="per-round boost profiles separated by '|'")
    args = parser.parse_args()
    source_dir = Path(args.spcl_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory {out_dir}")
    score_path = source_dir / "spcl_scores.jsonl"
    train, scores = _load_scores(args.train, score_path)
    if args.method == "mixs":
        report = build_mixs(train, source_dir, out_dir, args.seed, args.rounds)
    else:
        windows = tuple(int(x) for x in args.windows.split(",") if x.strip())
        boosts = tuple(tuple(float(x) for x in part.split(",") if x.strip())
                       for part in args.boosts.split("|") if part.strip())
        report = build_weighted_control(train, scores, source_dir, out_dir,
                                         args.method, args.seed, args.rounds, args.buckets,
                                         windows=windows, boost_profiles=boosts)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
