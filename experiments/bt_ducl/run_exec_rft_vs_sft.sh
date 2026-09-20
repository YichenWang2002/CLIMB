#!/usr/bin/env bash
# Headline two-arm protocol: budget-matched SFT vs Exec-RFT.
# Both arms run the SAME online loop: 16 prompts/update, 4 rollouts/prompt,
# strict executor verification, one full-completion SFT update.  The only
# difference is completion provenance in the training batch:
#   sft      (--train-mix gold): every slot keeps its gold completion;
#   exec_rft (--train-mix rft):  slots with a verified rollout train on one
#             deduplicated verified self-generated completion instead.
# Optimizer steps, token slots, rollout FLOPs, LR schedule, warm start, and
# decoding settings are identical, so total compute is matched by construction.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_exec_rft_seed${SEED}}"
TRAIN="${TRAIN:-outputs/dataset/train_aug10.jsonl}"
VAL="${VAL:-outputs/dataset/val.jsonl}"
TEST="${TEST:-outputs/dataset/test.jsonl}"
BASE="${BASE:-models/llama32-1b}"
LOAD_MODE="${LOAD_MODE:-4bit}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
# Optional: copy an existing warmup run (same seed) instead of retraining it.
WARMUP_FROM="${WARMUP_FROM:-}"
ONLINE_UPDATES="${ONLINE_UPDATES:-375}"
ONLINE_LR="${ONLINE_LR:-1e-5}"
SELECT_SIZE="${SELECT_SIZE:-16}"
N_STRATA="${N_STRATA:-0}"
# PARALLEL_ARMS=1 trains both arms concurrently on one GPU.  It defaults
# MICRO_BATCH to 1: two concurrent backward passes at micro_batch=2 exceed
# 24GB, while micro_batch=1 halves activation memory and leaves the
# optimizer-step semantics identical across arms.
PARALLEL_ARMS="${PARALLEL_ARMS:-0}"
if [ "$PARALLEL_ARMS" = "1" ]; then
  MICRO_BATCH="${MICRO_BATCH:-1}"
else
  MICRO_BATCH="${MICRO_BATCH:-2}"
fi
# K=2 rollouts per draw: enough to classify mastered/learnable/unsolved for
# the optimal-difficulty gate while halving the dominant generation cost.
ROLLOUTS_PER_ARM="${ROLLOUTS_PER_ARM:-2}"
KL_BETA="${KL_BETA:-0.1}"
LOGPROB_BATCH="${LOGPROB_BATCH:-8}"
GENERATION_BATCH="${GENERATION_BATCH:-32}"
SAVE_EVERY="${SAVE_EVERY:-25}"
RECOVERY_EVERY="${RECOVERY_EVERY:-1}"
ARM_RETRIES="${ARM_RETRIES:-10}"
MAX_NEW="${MAX_NEW:-1400}"
MAX_REPLACED="${MAX_REPLACED:-8}"
MIN_VAL_GAIN="${MIN_VAL_GAIN:-0.03}"

mkdir -p "$ROOT"

# One shared warm-start model is selected on held-out strict validation. Both
# headline arms branch from this exact adapter directory.
if [ -n "$WARMUP_FROM" ] && [ ! -f "$ROOT/warmup/train_metadata.json" ]; then
  cp -r "$WARMUP_FROM" "$ROOT/warmup"
fi
if [ ! -f "$ROOT/warmup/train_metadata.json" ]; then
  python3 -u -m experiments.bt_ducl.train_v4 --mode flat \
    --train "$TRAIN" --val "$VAL" --run-dir "$ROOT/warmup" \
    --epochs "$WARMUP_EPOCHS" --batch 4 --accum 4 --lr 1e-4 \
    --max-len 2560 --seed "$SEED" --loss-scope completion
fi

python3 - "$ROOT/warmup/train_metadata.json" "$ROOT/warmup" "$VAL" "$MAX_NEW" <<'PY'
import json, subprocess, sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
out_dir, val, max_new = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for step in metadata["checkpoint_steps"]:
    out = out_dir / "strict_val" / f"val_step_{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        continue
    subprocess.run([
        "python3", "-u", "-m", "experiments.bt_ducl.strict_eval",
        "--data", val, "--adapter", str(out_dir / f"checkpoint-{step}"),
        "--out", str(out), "--batch-size", "64", "--max-new", max_new,
    ], check=True)
PY
python3 -u -m experiments.bt_ducl.select_checkpoint \
  --eval-dir "$ROOT/warmup/strict_val" --out "$ROOT/warmup/selection.json"

WARM_START="$(python3 - "$ROOT/warmup/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["adapter"])
PY
)"

run_arm() {
  local name="$1" train_mix="$2"
  local out="$ROOT/$name"
  if [ -f "$out/train_metadata.json" ]; then return; fi
  local attempt=1 status=0
  while true; do
    if python3 -u -m experiments.bt_ducl.exec_ac_sft \
      --train "$TRAIN" --warm-start "$WARM_START" --base "$BASE" \
      --load-mode "$LOAD_MODE" \
      --run-dir "$out" --curator uniform --train-mix "$train_mix" \
      --max-replaced "$MAX_REPLACED" --kl-beta "$KL_BETA" \
      --updates "$ONLINE_UPDATES" \
      --lr "$ONLINE_LR" --select-size "$SELECT_SIZE" \
      --n-strata "$N_STRATA" --micro-batch "$MICRO_BATCH" \
      --rollouts-per-arm "$ROLLOUTS_PER_ARM" --max-new "$MAX_NEW" \
      --logprob-batch "$LOGPROB_BATCH" --generation-batch "$GENERATION_BATCH" \
      --save-every "$SAVE_EVERY" --recovery-every "$RECOVERY_EVERY" \
      --resume --seed "$SEED"; then
      break
    else
      status=$?
    fi
    if [ "$attempt" -ge "$ARM_RETRIES" ]; then
      echo "arm $name failed after $attempt attempts (status=$status)" >&2
      return "$status"
    fi
    echo "arm $name interrupted (status=$status); resuming in 5 seconds" >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

# Exactly two online arms are run: matched SFT and Exec-RFT.
if [ "$PARALLEL_ARMS" = "1" ]; then
  run_arm sft gold &
  sft_pid=$!
  run_arm exec_rft rft &
  rft_pid=$!
  status=0
  wait "$sft_pid" || status=$?
  wait "$rft_pid" || status=$?
  if [ "$status" -ne 0 ]; then
    echo "a headline arm failed (status=$status)" >&2
    exit "$status"
  fi
else
  run_arm sft gold
  run_arm exec_rft rft
fi

python3 - "$ROOT/sft/train_metadata.json" "$ROOT/exec_rft/train_metadata.json" <<'PY'
import json, sys
sft, rft = (json.load(open(path)) for path in sys.argv[1:])
matching = (
    "seed", "n_train", "updates", "select_size", "curator",
    "micro_batch", "rollouts_per_arm", "max_len", "max_new",
    "generation_batch", "logprob_batch", "temperature", "top_p",
    "learning_rate", "warm_start", "load_mode", "recovery_every",
    "max_replaced", "kl_beta",
)
mismatches = {key: (sft.get(key), rft.get(key)) for key in matching
              if sft.get(key) != rft.get(key)}
if sft.get("strata", {}).get("assignment_sha256") != rft.get("strata", {}).get("assignment_sha256"):
    mismatches["strata.assignment_sha256"] = (
        sft.get("strata", {}).get("assignment_sha256"),
        rft.get("strata", {}).get("assignment_sha256"),
    )
if mismatches:
    raise SystemExit("headline arms are not protocol matched: " + repr(mismatches))
if sft.get("train_mix") != "gold" or rft.get("train_mix") != "rft":
    raise SystemExit("unexpected headline train-mix configuration")
if sft.get("curator") != "uniform" or rft.get("curator") != "uniform":
    raise SystemExit("unexpected headline curator configuration")
print("protocol audit: SFT and Exec-RFT actor/rollout budgets match")
PY

eval_arm() {
  local name="$1"
  local out="$ROOT/$name"
  python3 - "$out" "$VAL" "$MAX_NEW" <<'PY'
import subprocess, sys
from pathlib import Path
run, val, max_new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
steps = sorted(int(path.name.split("-")[-1]) for path in run.glob("checkpoint-*")
               if path.is_dir())
if not steps:
    raise SystemExit(f"no checkpoints found under {run}")
for step in steps:
    out = run / "strict_val" / f"val_step_{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        continue
    subprocess.run([
        "python3", "-u", "-m", "experiments.bt_ducl.strict_eval",
        "--data", val, "--adapter", str(run / f"checkpoint-{step}"),
        "--out", str(out), "--batch-size", "64", "--max-new", max_new,
    ], check=True)
PY
  python3 -u -m experiments.bt_ducl.select_checkpoint \
    --eval-dir "$out/strict_val" --out "$out/selection.json"
}
eval_arm sft
eval_arm exec_rft

python3 - "$ROOT/sft/selection.json" "$ROOT/exec_rft/selection.json" "$ROOT/go_no_go.json" "$MIN_VAL_GAIN" <<'PY'
import json, sys
from pathlib import Path
sft = json.load(open(sys.argv[1]))["selected"]
rft = json.load(open(sys.argv[2]))["selected"]
threshold = float(sys.argv[4])
delta = float(rft["strict_success_rate"]) - float(sft["strict_success_rate"])
decision = {
    "metric": "held_out_validation_strict_success_rate",
    "sft": sft,
    "exec_rft": rft,
    "delta_exec_rft_minus_sft": delta,
    "minimum_gain": threshold,
    "test_authorized": delta >= threshold,
}
Path(sys.argv[3]).write_text(json.dumps(decision, indent=2), encoding="utf-8")
print(json.dumps(decision, indent=2))
if delta < threshold:
    print(f"go/no-go gate: validation gain {delta:.4f} < {threshold:.4f}; test not run")
PY
AUTHORIZED="$(python3 - "$ROOT/go_no_go.json" <<'PY'
import json, sys
print("true" if json.load(open(sys.argv[1]))["test_authorized"] else "false")
PY
)"
if [ "$AUTHORIZED" != "true" ]; then
  exit 0
fi

for name in sft exec_rft; do
  out="$ROOT/$name"
  adapter="$(python3 - "$out/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["adapter"])
PY
)"
  python3 -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$adapter" --out "$out/strict_test.json" \
    --batch-size 64 --max-new "$MAX_NEW"
done

python3 -u -m experiments.bt_ducl.compare \
  --flat "$ROOT/sft/strict_test.json" \
  --method "$ROOT/exec_rft/strict_test.json" \
  --out "$ROOT/compare_exec_rft_vs_sft.json"

echo "Completed exactly two headline arms under $ROOT"
