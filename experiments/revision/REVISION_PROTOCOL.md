# ICLR 2027 revision protocol

This directory contains the executable protocol for the three highest-value
follow-up experiments. The protocol is deliberately additive: existing
`pipeline/outputs` results are not overwritten.

## Priority experiments

1. **Qwen2.5-1.5B replication**

   Run flat SFT and SPCL on the identical 6,000/600/600 splits for seeds
   `42,43,44`, using the local snapshot
   `/root/autodl-tmp/mBT/model/qwen25-15b`. Each arm has three equal
   optimizer epochs (effective batch 16); SPCL has three one-epoch rounds.
   Revision runs use TRL `chunked_nll`, which is the same NLL objective with
   chunked entropy accounting to avoid a full-vocabulary memory peak. The
   micro-batch and context limit are recorded in each manifest; on the 24GB
   GPU, `batch=16, accumulation=1, max_len=2048` is the validated setting.
   Strict test evaluation reports overall, T1/T2/T3, domain, and primitive
   groups. Topology SCD is evaluated separately. `EVAL_BATCH` is an explicit
   throughput-only setting recorded in the manifest; the validated Qwen run
   used 32 for generation on the 24GB GPU.

2. **Three-seed attribution**

   For Llama seeds `42,43,44`, train `spcl`, `mixs`, `hardtail`, and `anti`.
   `build_spcl.py` writes `spcl_scores.jsonl`; controls verify row identities
   against this sidecar. `mixs` has an exact realized-multiset hash gate.
   `aggregate_revision.py` reports mean/std across seeds, paired McNemar,
   bootstrap intervals, and Holm-adjusted p-values.

3. **Independent external domain**

   `build_external_domain.py` creates a 300-task `kitchen` split with tier mix
   20/40/40. The domain is absent from train/validation and introduces the
   unseen `SanitizeCounter` primitive. The split is executor-validated and
   identity-checked before inference. The default template prose is fully
   reproducible; `--nl-mode api` uses the benchmark's existing NL service.
   External evaluation defaults to `EVAL_BATCH=4`; this changes throughput
   only and keeps generation within the local 24GB GPU budget.

If training was launched with `RUN_SCD=0` to reserve time, run
`scripts/revision/run_scd_replication.sh` after both adapters exist. It only
performs the topology-constrained test pass and writes the two SCD result
files beside the unconstrained results.

## Dry run first

From `/root/autodl-tmp/mBT`:

```bash
PYTHONPATH=pipeline python3 -m experiments.revision.protocol \
  validate-external --external pipeline/outputs/dataset/test.jsonl \
  --train pipeline/outputs/dataset/train_aug10.jsonl \
  --val pipeline/outputs/dataset/val.jsonl
```

The command above should fail because the current test split is not the
external `kitchen` domain; that is an intentional leakage guard.

```bash
DRY_RUN=1 SEEDS="42" bash pipeline/scripts/revision/run_backbone_replication.sh
DRY_RUN=1 SEEDS="42" ARMS="spcl mixs hardtail anti" \
  bash pipeline/scripts/revision/run_attribution_multiseed.sh
DRY_RUN=1 bash pipeline/scripts/revision/run_external_domain.sh
```

Only after the command previews are inspected should `DRY_RUN=0` be used. A
Qwen run downloads nothing: the script requires the local model directory.
For final reporting, resolve and record the model snapshot hash in the
generated manifest. The default manifest hashes configuration/tokenizer files
and records weight metadata; use `--hash-weights` for a content-level audit.

## Frontier + EGVD fairness

`evaluate_candidates.py` is the common executor-scoring interface for any
external model. Its input is one JSON object per task:

```json
{"record_id": "<hash of instruction/input/output>",
 "candidates": ["<xml candidate 1>", "<xml candidate 2>"]}
```

Candidates now pass the same whitelist/schema validation used by
`strict_eval.py` before they reach the executor. The exact same `k`, executor,
early-stop rule, and task IDs must be used for a frontier model and COMPASS.
No frontier number is included in this repository until its candidate file and
generation budget are available; existing single-candidate 5-shot API files
are not silently relabeled as pass@k.

Use `prepare_candidates.py --data <test.jsonl> --source draw1.jsonl ...
--out frontier_candidates.jsonl` to align repeated `run_api.py` outputs. It
requires every source to contain exactly the same contiguous task indices and
keeps duplicate draws so every row has the requested fixed `k`.

When the semantic/utility sidecar has already been produced for the same base,
seed, and split, `run_backbone_replication.sh` accepts
`CURRICULUM_SOURCE=<path-to-spcl-dir>` to copy the immutable curriculum
artifacts rather than rescore the 6,000 training rows.

`report_revision.py` is a post-hoc reporting command. It reads only completed
evaluator JSONs, validates unique task IDs, and writes JSON/CSV/LaTeX outputs
with overall, tier, coordination, shared-primitive, and new-primitive rates.
For example:

```bash
PYTHONPATH=pipeline python3 -m experiments.revision.report_revision \
  --run flat=pipeline/outputs/revision/.../flat_test.json \
  --run spcl=pipeline/outputs/revision/.../spcl_test.json \
  --out pipeline/outputs/revision/qwen_report.json
```

With `flat` and `spcl` labels it automatically computes a paired report;
additional comparisons can be requested with `--pair left,right`. The
backbone script uses `flat` as its explicit baseline, while the attribution
script uses `spcl`, so the sign of every paired delta is unambiguous.

## Reporting rules

Do not call a result backbone-agnostic from one Qwen seed. Report preliminary
replication if fewer than three seeds finish. Do not call the kitchen result
“external validity” if the split validator fails. Keep the T1 single-agent
trade-off visible in the abstract/table; report coordination and overall
separately. Do not use the test set for checkpoint selection.
