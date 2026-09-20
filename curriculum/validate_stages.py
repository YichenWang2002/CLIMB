"""Validate a DUE-driven stage partition WITHOUT any training.

Checks whether the data-driven stages capture real task difficulty by
correlating them with independent signals:

  1. structural complexity of the target BT (n_sync / n_co / n_fb / depth /
     n_nodes) -- should increase across stages if DUE captures difficulty;
  2. agreement (ARI / NMI) with the generation-time structural stages;
  3. tier composition (T1/T2/T3) per stage.

Optionally correlates DUE difficulty with an externally computed base-model
loss file (--base-loss jsonl with {"idx": ..., "loss": ...}) once GPU is back.

Usage:
  python -m curriculum.validate_stages \
      --staged outputs/curriculum_due/train_due_staged.jsonl \
      --out outputs/curriculum_due/validate_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from curriculum.build_curriculum import stage_of  # noqa: E402
from curriculum.structural import structural_features  # noqa: E402


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def ari_nmi(labels_a: list, labels_b: list) -> dict:
    try:
        from sklearn.metrics import (adjusted_rand_score,
                                     normalized_mutual_info_score)
        return {
            "ARI": adjusted_rand_score(labels_a, labels_b),
            "NMI": normalized_mutual_info_score(labels_a, labels_b),
        }
    except Exception as e:  # noqa: BLE001
        return {"ARI": None, "NMI": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True,
                    help="jsonl with a due_stage field (from curriculum.stage_due)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-loss", default=None,
                    help="optional jsonl with per-record base-model loss for "
                         "correlation with DUE difficulty")
    args = ap.parse_args()

    records = load_jsonl(args.staged)
    n = len(records)
    print(f"n={n}")

    feat_sum = defaultdict(lambda: defaultdict(float))
    feat_cnt = Counter()
    tier_by_stage = defaultdict(Counter)
    due_stages, struct_stages = [], []
    conf = Counter()  # (due_stage, structural_stage)

    difficulties, struct_complexity = [], []

    for r in records:
        ds = int(r["due_stage"])
        feat = structural_features(r["output"])
        ss = stage_of(r)  # generation-time structural stage from meta
        due_stages.append(ds)
        struct_stages.append(ss)
        conf[(ds, ss)] += 1
        feat_cnt[ds] += 1
        for k, v in feat.items():
            feat_sum[ds][k] += v
        tier_by_stage[ds][r["meta"]["tier"]] += 1
        difficulties.append(float(r.get("difficulty", 0.0)))
        struct_complexity.append(feat["n_sync"] + feat["n_co"] + feat["n_fb"])

    k_stages = sorted(feat_cnt)
    print("\nper-stage structural complexity (means):")
    header = ["stage", "n", "n_sync", "n_co", "n_fb", "depth", "n_nodes"]
    print("  " + "  ".join(f"{h:>8}" for h in header))
    report = {"n": n, "stages": {}}
    for st in k_stages:
        c = feat_cnt[st]
        means = {k: feat_sum[st][k] / c for k in ("n_sync", "n_co", "n_fb",
                                                  "depth", "n_nodes")}
        report["stages"][str(st)] = {"n": c, **means,
                                     "tiers": dict(tier_by_stage[st])}
        print("  " + "  ".join([f"{st:>8}", f"{c:>8}"] +
                                [f"{means[k]:>8.2f}" for k in header[2:]]))
        print(f"           tiers: {dict(tier_by_stage[st])}")

    ks = sorted({s for _, s in conf})
    print("\nconfusion (rows=due_stage, cols=structural stage):")
    print("          " + "  ".join(f"S{s:>4}" for s in ks))
    for ds in k_stages:
        row = [conf[(ds, s)] for s in ks]
        print(f"  due{ds}   " + "  ".join(f"{v:>6}" for v in row))

    agree = ari_nmi(due_stages, struct_stages)
    report["agreement_vs_structural"] = agree
    print(f"\nagreement with structural stages: ARI={agree.get('ARI')}, "
          f"NMI={agree.get('NMI')}")

    # Spearman between DUE difficulty and structural complexity
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(difficulties, struct_complexity)
        report["difficulty_vs_structural_complexity"] = {"rho": rho, "p": p}
        print(f"Spearman(difficulty, n_sync+n_co+n_fb): rho={rho:.4f} p={p:.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"spearman skipped: {e}")

    if args.base_loss:
        losses = {r["idx"]: r["loss"] for r in load_jsonl(args.base_loss)}
        xs, ys = [], []
        for i, r in enumerate(records):
            if i in losses:
                xs.append(float(r.get("difficulty", 0.0)))
                ys.append(losses[i])
        if xs:
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
            report["difficulty_vs_base_loss"] = {"rho": rho, "p": p, "n": len(xs)}
            print(f"Spearman(difficulty, base_loss): rho={rho:.4f} "
                  f"p={p:.2e} (n={len(xs)})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()
