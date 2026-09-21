"""SPCL curriculum builder: difficulty = model-distance + BT structural
complexity (rank-fused), discrete buckets, competence pacing, utility-weighted
within-window sampling.

Design (see ICASSP_PLAN.md successor discussion):
  * Difficulty D_i: midrank fusion of
      (a) current-model semantic NLL/top-1 error (score_mt_ducl.score_difficulty)
      (b) static BT structural complexity (n_robots, n_faults, sync/co/fb
          counts, depth, n_nodes -- all midranked, PCA-fused)
  * Utility U_i: target-coverage utility from score_mt_ducl.score_utility
    (text + executable-BT structure coverage, validation referenced).
  * Buckets: K quantile buckets over D (bucket 0 = easiest).
  * Competence pacing: round r samples from the expanding prefix of buckets
    (default bucket counts 1,2,4 over 3 rounds). Each round draws n_train
    samples WITH replacement from the window, probability proportional to U,
    so total optimizer steps match a flat N-epoch baseline exactly.
    Easy buckets stay in every later window (built-in rehearsal).

Usage:
  python -m curriculum.build_spcl --train outputs/dataset/train_aug10.jsonl \
      --val outputs/dataset/val.jsonl --out-dir outputs/spcl_seed42 \
      --rounds 3 --buckets 4 --window 1,2,4 --batch-size 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from curriculum.score_mt_ducl import (  # noqa: E402
    BASE, load_jsonl, load_model, score_difficulty, score_utility,
    midrank01, adaptive_fuse, record_id,
)
from curriculum.structural import structural_features  # noqa: E402

STRUCT_KEYS = ("n_sync", "n_co", "n_fb", "depth", "n_nodes")


def struct_difficulty(records: list[dict], fusion_mode: str = "pca") -> tuple[np.ndarray, dict]:
    """Static task-side difficulty: mechanical complexity of the gold BT."""
    cols = {"n_robots": [], "n_faults": [], **{k: [] for k in STRUCT_KEYS}}
    parse_fail = 0
    for r in records:
        meta = r.get("meta", {})
        try:
            feat = structural_features(r["output"])
        except Exception:
            feat = {k: 0 for k in STRUCT_KEYS}
            parse_fail += 1
        cols["n_robots"].append(float(len(meta.get("robots", []))))
        cols["n_faults"].append(float(len(meta.get("faults", []))))
        for k in STRUCT_KEYS:
            cols[k].append(float(feat[k]))
    ranks = [midrank01(np.asarray(v, dtype=np.float64)) for v in cols.values()]
    fused, weights = adaptive_fuse(np.column_stack(ranks), mode=fusion_mode)
    return np.asarray(fused, dtype=np.float64), {
        "struct_difficulty_components": list(cols.keys()),
        "struct_difficulty_fusion": "uniform_mean" if fusion_mode == "mean" else "PCA_abs_loading_on_midrank_components",
        "struct_difficulty_adaptive_weights": [float(w) for w in weights],
        "struct_parse_failures": parse_fail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--window", required=True,
                    help="comma list of bucket-prefix sizes per round (withheld; set your own)")
    ap.add_argument("--boosts", default="",
                    help="per-round bucket weight multipliers, rounds "
                         "separated by '|', e.g. '1,1|1,1,1.5|0.5,0.75,1.5,3'. "
                         "Count per round must equal that round's window size.")
    ap.add_argument("--utility-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--base", default=BASE,
                    help="backbone used for semantic difficulty scoring")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only score the first N training records")
    ap.add_argument("--round-size", type=int, default=0,
                    help="samples drawn per round (default: n_train)")
    ap.add_argument("--no-utility", action="store_true",
                    help="ablation: set utility U_i = 1 for all samples")
    ap.add_argument("--fusion", choices=["pca", "mean"], default="pca",
                    help="feature fusion rule; mean is the simple-averaging ablation")
    ap.add_argument("--no-structural-view", action="store_true",
                    help="ablation: use semantic difficulty only (drop structural view)")
    ap.add_argument("--shuffle-buckets", action="store_true",
                    help="ablation: randomly permute bucket labels")
    ap.add_argument("--mixture", default="",
                    help="ablation: comma list of per-bucket sampling proportions; "
                    "pool is always the full training set, boosts ignored")
    ap.add_argument("--reverse", action="store_true",
                    help="ablation: anti-curriculum, pool = hardest prefix first")
    ap.add_argument("--scores-cache", default=None,
                    help="reuse difficulty/utility/bucket assignments from a "
                         "spcl_scores.jsonl sidecar; skips all scoring")
    ap.add_argument("--only-round", type=int, default=-1,
                    help="build only this round's file (0-based); sampling "
                         "rng and weights stay identical to a full build")
    ap.add_argument("--gate-residual", default=None,
                    help="SPCL-lambda: JSON with \"R_b\" (per-bucket "
                         "residual learning distance in [0,1], measured on "
                         "training data only); gates every nominal boost "
                         "m>1 to 1+(m-1)*R_b")
    ap.add_argument("--adaptive-dose-residual", default=None,
                    help="derive dose from training-side difficulty and residual "
                         "NLL: m_b proportional to Dbar_b*R_b; mutually "
                         "exclusive with --boosts/--gate-residual")
    ap.add_argument("--adaptive-dose-du", action="store_true",
                    help="derive continuous dose from training-side bucket "
                         "difficulty/utility ratio; no grid, floor, or cap")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = [int(x) for x in args.window.split(",")]
    boosts = None
    if args.boosts:
        boosts = [[float(x) for x in part.split(",")]
                  for part in args.boosts.split("|")]
        assert len(boosts) == args.rounds, "boosts must have one entry per round"
        for r, bb in enumerate(boosts):
            assert len(bb) == window[r], (f"round {r}: {len(bb)} boosts but "
                                          f"window size {window[r]}")
    assert len(window) == args.rounds, "window must have one entry per round"
    assert all(0 < w <= args.buckets for w in window)
    if args.only_round >= 0:
        assert args.only_round < args.rounds, "only-round out of range"
    else:
        assert window[-1] == args.buckets, "last round must cover all buckets"
    if args.gate_residual and args.reverse:
        raise SystemExit("--gate-residual is defined for the normal "
                         "(easy-first) bucket indexing only")
    if args.gate_residual and args.mixture:
        raise SystemExit("--gate-residual cannot be combined with --mixture")
    if args.adaptive_dose_residual and (args.boosts or args.gate_residual or args.mixture or args.adaptive_dose_du):
        raise SystemExit("--adaptive-dose-residual cannot be combined with "
                         "--boosts, --gate-residual, or --mixture")
    if args.adaptive_dose_du and (args.boosts or args.gate_residual or args.mixture):
        raise SystemExit("--adaptive-dose-du cannot be combined with --boosts, "
                         "--gate-residual, or --mixture")

    train = load_jsonl(args.train)
    val = load_jsonl(args.val)
    if args.limit:
        train = train[: args.limit]
    n = len(train)
    print(f"train={n} val={len(val)} buckets={args.buckets} window={window}",
          flush=True)

    if args.scores_cache:
        # Reuse proven scores: identical buckets/utility without rescoring.
        cache = load_jsonl(args.scores_cache)
        if len(cache) != len(train):
            raise SystemExit(f"scores cache has {len(cache)} rows but train "
                             f"has {len(train)}")
        for i, row in enumerate(cache):
            if row.get("spcl_record_id") != record_id(train[i]):
                raise SystemExit(f"scores cache row {i} record id mismatch")
        difficulty = np.asarray(
            [float(r["difficulty"]) for r in cache], dtype=np.float64)
        utility = np.asarray(
            [float(r["utility"]) for r in cache], dtype=np.float64)
        bucket_cache = np.asarray(
            [int(r["bucket"]) for r in cache], dtype=np.int64)
        d_weights = None
        diff_report = {"difficulty_reused_from": args.scores_cache}
        struct_report = {"struct_difficulty_reused_from": args.scores_cache}
        util_report = {"utility_reused_from": args.scores_cache}
    else:
        # --- difficulty (a): current-model semantic distance ---------------
        model, tokenizer = load_model(args.base, None)
        model_ranked, nll, err, n_tok, diff_report = score_difficulty(
            train, model, tokenizer, args.batch_size, args.max_len,
            fusion_mode=args.fusion)
        del model
        import torch
        torch.cuda.empty_cache()

        # --- difficulty (b): static structural complexity ------------------
        struct_d, struct_report = struct_difficulty(train, fusion_mode=args.fusion)

        # --- fuse the two views ---------------------------------------------
        if args.no_structural_view:
            difficulty = np.asarray(midrank01(np.asarray(model_ranked)), dtype=np.float64)
            d_weights = [1.0, 0.0]
        else:
            difficulty, d_weights = adaptive_fuse(np.column_stack([
                midrank01(np.asarray(model_ranked)), midrank01(struct_d)]),
                mode=args.fusion)
        difficulty = np.asarray(difficulty, dtype=np.float64)

        # --- utility: target coverage (validation referenced) ---------------
        utility, util_report = score_utility(
            train, val, args.embed_device, args.utility_k, 64,
            fusion_mode=args.fusion)
        utility = np.asarray(utility, dtype=np.float64)
        bucket_cache = None
    if args.no_utility:
        utility = np.ones_like(utility)
    util_report = {k: v for k, v in util_report.items()
                   if not k.startswith("utility_text") and k != "utility_struct"
                   or k in ("utility_text_sigma", "utility_text_mean",
                            "utility_struct_sigma", "utility_struct_mean")}

    # --- buckets -------------------------------------------------------------
    if bucket_cache is not None:
        bucket = bucket_cache
    else:
        order = np.argsort(difficulty, kind="mergesort")
        bucket = np.empty(n, dtype=np.int64)
        bounds = np.array_split(order, args.buckets)
        for b, idxs in enumerate(bounds):
            bucket[idxs] = b
    if args.shuffle_buckets:
        bucket = np.random.default_rng(args.seed + 1000).permutation(bucket)

    # --- SPCL-lambda dose gating ----------------------------------------------
    gate_report = None
    if args.gate_residual:
        gate_rep = json.loads(Path(args.gate_residual).read_text())
        r_b = [min(1.0, max(0.0, float(x))) for x in gate_rep["R_b"]]
        if len(r_b) != args.buckets:
            raise SystemExit(f"gate R_b has {len(r_b)} entries but there are "
                             f"{args.buckets} buckets")
        nominal = [[m for m in row] for row in boosts] if boosts else None
        if boosts is not None:
            for r_idx, row in enumerate(boosts):
                for j, m in enumerate(row):
                    if m > 1.0:
                        row[j] = 1.0 + (m - 1.0) * r_b[j]
        gate_report = {
            "gate_residual_source": args.gate_residual,
            "gate_label": gate_rep.get("label"),
            "gate_mode": gate_rep.get("mode"),
            "gate_metric": gate_rep.get("metric"),
            "rule": "m_realized = 1 + (m_nominal - 1) * R_b for m_nominal > "
                    "1; m_nominal <= 1 unchanged; R_b measured on training "
                    "data only",
            "R_b": r_b,
            "nominal_boosts": nominal,
            "realized_boosts": [list(map(float, row)) for row in boosts]
            if boosts else None,
        }
        print(f"SPCL-lambda gate R_b={[round(x, 4) for x in r_b]} "
              f"realized={gate_report['realized_boosts']}", flush=True)

    adaptive_report = None
    adaptive_bucket_dose = None
    if args.adaptive_dose_du:
        if args.only_round < 1:
            raise SystemExit("--adaptive-dose-du is only valid for rounds 2/3")
        dbar = np.asarray([float(difficulty[bucket == b].mean())
                           for b in range(args.buckets)], dtype=np.float64)
        ubar = np.asarray([float(utility[bucket == b].mean())
                           for b in range(args.buckets)], dtype=np.float64)
        if args.only_round == 1:
            adaptive_bucket_dose = 1.0 / np.clip(ubar, 1e-12, None)
            dose_rule = "m_b=(1/Ubar_b)/mean_active(1/Ubar)"
        else:
            adaptive_bucket_dose = dbar / np.clip(ubar, 1e-12, None)
            dose_rule = "m_b=(Dbar_b/Ubar_b)/mean_active(Dbar/Ubar)"
        adaptive_report = {
            "mode": "continuous_difficulty_utility_ratio",
            "rule": dose_rule + "; all signals computed on training split",
            "difficulty_mean_b": [float(x) for x in dbar],
            "utility_mean_b": [float(x) for x in ubar],
            "raw_dose_b": [float(x) for x in adaptive_bucket_dose],
        }
        print("adaptive D/U dose raw="
              f"{[round(float(x), 6) for x in adaptive_bucket_dose]}", flush=True)
    if args.adaptive_dose_residual:
        if args.only_round < 1:
            raise SystemExit("--adaptive-dose-residual is only valid for rounds 2/3")
        rep = json.loads(Path(args.adaptive_dose_residual).read_text())
        rb = np.asarray(rep.get("R_b", []), dtype=np.float64)
        if rb.size != args.buckets:
            raise SystemExit(f"adaptive residual has {rb.size} entries but there are "
                             f"{args.buckets} buckets")
        rb = np.clip(rb, 0.0, 1.0)
        dbar = np.asarray([float(difficulty[bucket == b].mean())
                           for b in range(args.buckets)], dtype=np.float64)
        # Relative dose is generated solely from dimensionless, training-side
        # signals. A global scale is immaterial because p is normalized.
        adaptive_bucket_dose = dbar * rb
        adaptive_report = {
            "mode": "state_derived_residual_difficulty",
            "rule": "m_b=(Dbar_b*R_b)/mean_active(Dbar*R); "
                    "R_b=NLL_b(theta)/NLL_b(theta0), training data only",
            "residual_source": args.adaptive_dose_residual,
            "R_b": [float(x) for x in rb],
            "difficulty_mean_b": [float(x) for x in dbar],
            "raw_D_times_R_b": [float(x) for x in adaptive_bucket_dose],
        }
        print("adaptive dose raw D*R="
              f"{[round(float(x), 6) for x in adaptive_bucket_dose]}", flush=True)

    # --- paced, utility-weighted sampling -----------------------------------
    mix = None
    if args.mixture:
        mix = np.asarray([float(x) for x in args.mixture.split(",")],
                         dtype=np.float64)
        assert len(mix) == args.buckets, "mixture must have one entry per bucket"
    round_files = []
    rounds_to_build = ([args.only_round] if args.only_round >= 0
                       else range(args.rounds))
    for r in rounds_to_build:
        prefix = window[r]
        if mix is not None:
            pool = np.arange(n)
        elif args.reverse:
            pool = np.where(bucket >= args.buckets - prefix)[0]
        else:
            pool = np.where(bucket < prefix)[0]
        w = utility[pool].astype(np.float64)
        if mix is not None:
            mult = np.asarray([mix[bucket[i]] for i in pool], dtype=np.float64)
            w = w * mult
        elif boosts is not None:
            if args.reverse:
                mult = np.asarray([boosts[r][args.buckets - 1 - bucket[i]]
                                   for i in pool], dtype=np.float64)
            else:
                mult = np.asarray([boosts[r][bucket[i]] for i in pool],
                                  dtype=np.float64)
            w = w * mult
        elif adaptive_bucket_dose is not None:
            active = np.asarray([adaptive_bucket_dose[b] for b in bucket[pool]],
                                dtype=np.float64)
            denom = float(np.mean(active))
            if not np.isfinite(denom) or denom <= 0:
                raise SystemExit("adaptive dose has no positive residual-distance mass")
            w = w * (active / denom)
            adaptive_report["active_window"] = int(prefix)
            adaptive_report["realized_m"] = [
                float(adaptive_bucket_dose[b] / denom) for b in range(prefix)]
        w = np.clip(w, 1e-6, None)
        p = w / w.sum()
        rng = np.random.default_rng(args.seed + r)
        rs = args.round_size or n
        take = rng.choice(pool, size=rs, replace=True, p=p)
        take = take[rng.permutation(rs)]  # shuffle within the round
        path = out_dir / f"round{r + 1}.jsonl"
        with path.open("w") as f:
            for i in take:
                f.write(json.dumps(train[int(i)], ensure_ascii=False) + "\n")
        round_files.append(str(path))
        print(f"round{r + 1}: prefix={prefix} reverse={args.reverse} mix={mix is not None} "
              f"pool={len(pool)} sampled={rs} unique={len(set(take.tolist()))}",
              flush=True)

    report = {
        "method": "SPCL: model+structure difficulty, discrete buckets, "
                  "competence pacing, utility-weighted within-window sampling",
        "base_model": args.base,
        "n_train": n, "n_val": len(val), "seed": args.seed,
        "rounds": args.rounds, "buckets": args.buckets, "window": window,
        "boosts": boosts,
        "only_round": args.only_round,
        "scores_cache": args.scores_cache,
        "gate": gate_report,
        "adaptive_dose": adaptive_report,
        "fusion_mode": args.fusion,
        "no_structural_view": args.no_structural_view,
        "difficulty_fusion_weights": (
            [float(w) for w in d_weights] if d_weights is not None else None),
        "difficulty_model_report": diff_report,
        "difficulty_struct_report": struct_report,
        "utility_report": util_report,
        "bucket_sizes": [int((bucket == b).sum()) for b in range(args.buckets)],
        "bucket_difficulty_mean": [float(difficulty[bucket == b].mean())
                                   for b in range(args.buckets)],
        "bucket_scenarios": {
            str(b): {
                s: int(sum(1 for i in range(n)
                           if bucket[i] == b
                           and train[i].get("meta", {}).get("scenario", "?") == s))
                for s in sorted({train[i].get("meta", {}).get("scenario", "?")
                                 for i in range(n)})}
            for b in range(args.buckets)},
        "difficulty_utility_rank_corr": float(np.corrcoef(
            midrank01(difficulty), midrank01(utility))[0, 1]),
        "round_files": round_files,
        "samples_per_round": args.round_size or n,
        "budget_note": "each round samples n_train rows; 3 rounds == flat 3 "
                       "epochs in optimizer steps",
    }
    # Keep the original row order and the derived signals in a small sidecar.
    # Attribution controls can then reuse exactly the same scoring pass without
    # changing the model or validation data used to define the curriculum.
    score_path = out_dir / "spcl_scores.jsonl"
    with score_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(train):
            f.write(json.dumps({
                "spcl_record_id": record_id(row),
                "difficulty": float(difficulty[i]),
                "utility": float(utility[i]),
                "bucket": int(bucket[i]),
            }, ensure_ascii=False) + "\n")
    report["score_sidecar"] = str(score_path)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"bucket_sizes": report["bucket_sizes"],
                      "bucket_scenarios": report["bucket_scenarios"],
                      "d_weights": report["difficulty_fusion_weights"],
                      "du_rank_corr": report["difficulty_utility_rank_corr"]},
                     indent=2), flush=True)
    print("SPCL curriculum built.", flush=True)


if __name__ == "__main__":
    main()
