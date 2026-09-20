"""Retune v4 mu_p temperature from cached semantic NLL without rerunning the LM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_jsonl
from .scoring_v4 import (_input_embeddings, _target_weights, diagnostics,
                         sinkhorn_utility, tie_rank)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--ot-device", default="cuda")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--blur", type=float, default=0.1)
    args = ap.parse_args()
    train, val = load_jsonl(args.scores), load_jsonl(args.val)
    semantic_nll = np.asarray([row["v4_semantic_nll"] for row in train], dtype=np.float64)
    difficulty = np.asarray([row["v4_difficulty"] for row in train], dtype=np.float64)
    semantic_counts = np.asarray([row["v4_semantic_tokens"] for row in train], dtype=np.float64)
    train_emb, train_report = _input_embeddings(
        train, args.embed_model, args.embed_device, args.embed_batch_size)
    val_emb, val_report = _input_embeddings(
        val, args.embed_model, args.embed_device, args.embed_batch_size)
    val_weights, target_report = _target_weights(val)
    utility, utility_report = sinkhorn_utility(
        train_emb, val_emb, semantic_nll, val_weights, args.temperature,
        args.epsilon, args.blur, args.ot_device)
    d_rank, u_rank = tie_rank(difficulty), tie_rank(utility)
    due = d_rank / np.maximum(u_rank, 1e-12)
    diagnostic = diagnostics(train, difficulty, utility, due, semantic_counts)
    ess_ok = 100.0 <= utility_report["mu_p_effective_sample_size"] <= 1000.0
    acceptance = dict(diagnostic["diagnostic_acceptance"])
    acceptance["mu_p_ess_between_100_and_1000"] = ess_ok
    accepted = all(acceptance.values())
    if not accepted:
        raise RuntimeError("retuned scoring checks failed: " + json.dumps(acceptance))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(train):
            item = dict(row)
            item.update({"v4_utility": float(utility[i]), "v4_due": float(due[i]),
                         "v4_d_rank": float(d_rank[i]), "v4_u_rank": float(u_rank[i])})
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    old_report_path = Path(args.scores).parent / "score_report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    report = {
        **old_report,
        "score_version": "bt_ducl_v4_semantic_nll_input_sinkhorn_rank_due_ess_calibrated",
        "temperature_selection": "training_semantic_nll_ESS_scan_no_validation_or_test_metric",
        "train_embedding_report": train_report,
        "validation_embedding_report": val_report,
        **target_report, **utility_report, **diagnostic,
        "diagnostic_acceptance": acceptance,
        "diagnostic_accepted": accepted,
    }
    (out_dir / "score_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
