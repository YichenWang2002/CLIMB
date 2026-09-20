# Exec-RFT protocol

This protocol replaces the static curriculum and the pure-selection
Exec-AC-SFT design. Its mechanism is **completion provenance under a fixed
training budget**: verified self-generated completions replace gold
completions in an otherwise identical online SFT loop. It does not use tier,
domain, fault, validation labels, or test information. The executor is a
sound verifier: reward 1 is returned iff parse, schema, and symbolic
execution all succeed, so a verified training sample is correct by
construction.

## Objective

The actor maximizes expected strict execution reward
`E_{x~D} E_{y~pi(.|x)} [R(x,y)]`. Training on executor-filtered samples is
the M-step of reward-weighted regression on the reward-tilted distribution
`q*(y|x) ∝ pi(y|x) R(x,y)` (Peters and Schaal 2007; ReST, Gulcehre et al.
2023; STaR, Zelikman et al. 2022; RFT, Yuan et al. 2023). Gold-only SFT
imitates one arbitrary point in the correct-completion set per mission;
verified self-training concentrates probability on correct completions the
current policy can already reach, which is the KL-closest correct
distribution to the actor. Because the verifier is sound, self-training adds
no incorrect behavior, unlike unverified self-distillation.

## Budget-matching contract

Both headline arms execute the identical loop from the identical
validation-selected warm start:

1. draw 16 prompts uniformly from the same prompt-only strata prior;
2. sample 4 rollouts per prompt (temperature 0.7, top-p 0.95) and verify
   each with the strict executor — **both arms pay this cost**;
3. compose 16 training slots:
   - `sft` (`--train-mix gold`): every slot keeps its gold completion;
     rollouts are discarded;
   - `exec_rft` (`--train-mix rft`): a draw with at least one verified
     rollout trains on one deduplicated verified self-generated completion;
     draws without one keep gold;
4. one identical full-completion SFT update (same optimizer, LR schedule,
   micro-batch, gradient clipping).

Optimizer steps, slot count, rollout FLOPs, LR schedule, warm start, strata
assignment hash, and decoding settings are asserted equal in
`train_metadata.json`; the only permitted difference is `train_mix`. Each
prompt contributes at most one verified sample per update, so the prompt
marginal distribution is identical across arms; gold slots anchor the format
and cover prompts with no verified rollout.

## Guarantees and limits

The per-step replacement is on-policy: verified samples come from the
current actor at the current update, so no stale-data correction is needed.
The protocol does not claim that a finite neural actor must beat SFT on
unseen tasks; it isolates one mechanism (completion provenance) so any gain
is attributable. Known limit: prompts the actor never solves in any rollout
(~21% in the seed-42 audit) contribute no verified samples and rely on gold
slots only.

## Stability conditions (learned from a failed run)

A first seed-42 attempt collapsed within ~20 updates (rollout success
0.6 -> 0.17 while the matched gold arm stayed flat).  Two causes were
identified and are now enforced by construction:

1. **Format-preserving replacement.**  The validator's `re_extract` strips
   the XML declaration, while every gold completion starts with
   `<?xml ...?>`.  Training on stripped bodies taught the actor to drop the
   declaration, then destabilized the whole root element.  Replacements now
   keep the model's own preamble plus the validated clean body plus the
   gold-style trailing newline (`_training_text`).
2. **Gold anchoring.**  `MAX_REPLACED=8` caps replacements at half of the
   16 slots, so at least 50% of every batch is gold and the feedback loop
   cannot run away.  `replace_rate` is logged per step; a sustained drop in
   rollout reward on the training prompts is the abort signal.

## Run

```bash
cd /root/autodl-tmp/mBT/pipeline
SEED=42 ROOT=outputs/bt_exec_rft_seed42 \
  WARMUP_FROM=outputs/bt_exec_ac_vs_sft_seed42/warmup \
  bash experiments/bt_ducl/run_exec_rft_vs_sft.sh
```

Useful overrides are `SEED`, `ROOT`, `ONLINE_UPDATES`, `ROLLOUTS_PER_ARM`,
`MAX_REPLACED`, `MICRO_BATCH`, `LOGPROB_BATCH`, `GENERATION_BATCH`, and
`SAVE_EVERY`. `WARMUP_FROM` may copy a same-seed warmup run to skip
retraining.

The runner selects one shared warm-start checkpoint on strict validation,
trains the two matched arms, selects one checkpoint per arm on strict
validation, and applies a pre-registered `MIN_VAL_GAIN=0.03` go/no-go gate.
Only when Exec-RFT exceeds matched SFT by at least three validation points
does it evaluate the held-out test once per arm and write the paired
comparison (exact McNemar plus paired bootstrap 95% CI) via
`experiments.bt_ducl.compare`. Per-step `replace_rate`, `n_replaced_slots`,
and `verified_draws` are logged so the replacement mechanism can be audited.
The headline claim requires the gate to pass on at least three seeds with
McNemar p < 0.05 on the paired test comparison.
