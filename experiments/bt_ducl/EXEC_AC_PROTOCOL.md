# Exec-AC-SFT protocol

This protocol is the replacement for the static DUE/ability curriculum. It
does not use tier, domain, fault, plan length, output XML, validation labels,
or manually defined phases. Its only automatically induced structure is a
prompt-only partition learned before online training.

## Objective

The actor is first trained with ordinary full-completion SFT. Every XML token,
including the declaration, root attributes, punctuation, closing tags, and EOS,
is supervised. The online stage keeps that actor loss unchanged and uses the
strict executor only as a curator feedback signal:

```text
reward(x, y) = 1 iff parse(y), schema(y), and execute(x, y) all succeed
             = 0 otherwise
```

The training rows are partitioned into `K = ceil(sqrt(T))` prompt strata, where
`T` is the fixed number of online updates. The deterministic partition is
TF-IDF -> SVD -> MiniBatchKMeans over `instruction + input` only; its assignment
hash and sizes are recorded. Initial stratum mass equals its row fraction, so
sampling a stratum and then a uniform row within it is initially exactly
uniform over rows. The actor receives a normal full-completion SFT update on
selected gold rows, and the curator receives importance-weighted policy
improvement feedback from pre/post-update rollout probabilities.

For each selected row, the default four stochastic rollouts use a leave-one-out
reward baseline. This avoids the biased single-rollout EMA approximation. The
uniform-online control uses the same stratum prior, row sampling, rollouts,
actor updates, optimizer steps, and decoding settings; it only disables the
OSMD update. Its row distribution therefore remains uniform despite the
automatic partition.

## Guarantees and limits

For a selected stratum `g`, the logged inclusion probability is its current
mass `p_g`; the row probability is `p_g / |g|`. With `B=select_size` draws,
the group utility estimator is the bounded policy improvement divided by
`B p_g`, which is unbiased for the clipped stratum utility. Repeated stratum
or row draws are allowed and counted separately. The exponentiated-gradient
rate and exploration mass are automatically computed from `(K, T, B)` using
the bounded multiple-play bandit scale; they are stored in metadata and are
not tuned from validation or test results.

`restart_block` and `restart_z` form a data-independent statistical detector.
When two reward-utility blocks differ beyond the configured confidence bound,
the curator is reset toward its pre-training row-fraction prior. The actor is
not reset. This is an automatic non-stationary response, not a hand-written
tier boundary; per-step fixed-share decay is disabled so ordinary updates are
not washed out before a stratum is revisited.

The regret statement applies to the finite-stratum curator under the usual
bounded, unbiased bandit-utility and small-update assumptions. It does not
mathematically imply
that a finite neural actor must beat SFT on an unseen test set. The headline
runner does not use a pilot gate or any third arm. It launches exactly two
arms, `sft` (uniform curator) and `exec_ac` (OSMD curator), from the same
validation-selected warm-start adapter. Both arms use identical partition
hash, row prior, rollout count, actor-update count, optimizer settings, and
decoding settings; only the curator update is different. Checkpoints are selected using
validation strict execution only, and the held-out test is evaluated once per
selected arm.

The full-sequence actor probability ratio is evaluated in log space in small
forward batches. `actor_log_ratio_clip` is an explicit numerical truncation for
large one-step SFT changes; its clipping fraction is logged, so runs with
material clipping cannot be presented as an exact-identity result.

## Run

```bash
cd /root/autodl-tmp/mBT/pipeline
SEED=42 ROOT=outputs/bt_exec_ac_vs_sft_seed42 \
  bash experiments/bt_ducl/run_exec_ac_vs_sft_single_seed.sh
```

Useful overrides are `SEED`, `ROOT`, `ONLINE_UPDATES`, `ROLLOUTS_PER_ARM`,
`MICRO_BATCH`, `LOGPROB_BATCH`, `GENERATION_BATCH`, and `SAVE_EVERY`.

The runner first selects one shared warm-start checkpoint on strict validation,
then trains and evaluates only the matched `sft` and `exec_ac` arms. It
then selects one checkpoint per arm on strict validation. A pre-registered
`MIN_VAL_GAIN=0.03` gate authorizes the held-out test only when Exec-AC exceeds
uniform SFT by at least three percentage points; otherwise it writes
`go_no_go.json` and stops without touching test. When authorized, it evaluates
the test set once per selected arm and writes the paired comparison under the
run directory. The headline runner uses `micro_batch=2`,
`generation_batch=16`, and `logprob_batch=8` on the 24GB GPU; these are memory
settings applied identically to both arms, not method differences. Checkpoints
are saved every 25 updates to limit validation multiple-comparisons while
retaining the final update.
