#!/usr/bin/env bash
# Headline two-arm protocol: matched full-completion SFT vs Exec-AC-SFT.
# The SFT arm uses the same online actor-update and rollout budget as Exec-AC;
# its curator is fixed to uniform, so the only method difference is OSMD
# reward-driven stratum selection. No separate pilot/control experiment is
# launched by this script.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_exec_ac_vs_sft_seed${SEED}}"
TRAIN="${TRAIN:-outputs/dataset/train_aug10.jsonl}"
VAL="${VAL:-outputs/dataset/val.jsonl}"
TEST="${TEST:-outputs/dataset/test.jsonl}"
BASE="${BASE:-models/llama32-1b}"
LOAD_MODE="${LOAD_MODE:-4bit}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
ONLINE_UPDATES="${ONLINE_UPDATES:-225}"
ONLINE_LR="${ONLINE_LR:-1e-5}"
SELECT_SIZE="${SELECT_SIZE:-16}"
N_STRATA="${N_STRATA:-0}"
MICRO_BATCH="${MICRO_BATCH:-2}"
ROLLOUTS_PER_ARM="${ROLLOUTS_PER_ARM:-4}"
LOGPROB_BATCH="${LOGPROB_BATCH:-8}"
GENERATION_BATCH="${GENERATION_BATCH:-64}"
SAVE_EVERY="${SAVE_EVERY:-25}"
RECOVERY_EVERY="${RECOVERY_EVERY:-1}"
ARM_RETRIES="${ARM_RETRIES:-10}"
MAX_NEW="${MAX_NEW:-1400}"
MIN_VAL_GAIN="${MIN_VAL_GAIN:-0.03}"

mkdir -p "$ROOT"

# One shared warm-start model is selected on held-out strict validation. Both
# headline arms branch from this exact adapter directory.
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
  local name="$1" curator="$2"
  local out="$ROOT/$name"
  if [ -f "$out/train_metadata.json" ]; then return; fi
  local attempt=1 status=0
  while true; do
    if python3 -u -m experiments.bt_ducl.exec_ac_sft \
      --train "$TRAIN" --warm-start "$WARM_START" --base "$BASE" \
      --load-mode "$LOAD_MODE" \
      --run-dir "$out" --curator "$curator" --updates "$ONLINE_UPDATES" \
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

# Exactly two online arms are run: matched SFT and adaptive Exec-AC.
run_arm sft uniform
run_arm exec_ac osmd

python3 - "$ROOT/sft/train_metadata.json" "$ROOT/exec_ac/train_metadata.json" <<'PY'
import json, sys
sft, ac = (json.load(open(path)) for path in sys.argv[1:])
matching = (
    "seed", "n_train", "updates", "select_size",
    "micro_batch", "rollouts_per_arm", "max_len", "max_new",
    "generation_batch", "logprob_batch", "temperature", "top_p",
    "learning_rate", "warm_start", "load_mode", "recovery_every",
)
mismatches = {key: (sft.get(key), ac.get(key)) for key in matching
              if sft.get(key) != ac.get(key)}
mismatches.update({key: (sft.get(key), ac.get(key)) for key in (
    "osmd_eta", "exploration_floor", "fixed_share", "utility_clip")
    if sft.get(key) != ac.get(key)})
if sft.get("strata", {}).get("assignment_sha256") != ac.get("strata", {}).get("assignment_sha256"):
    mismatches["strata.assignment_sha256"] = (
        sft.get("strata", {}).get("assignment_sha256"),
        ac.get("strata", {}).get("assignment_sha256"),
    )
if mismatches:
    raise SystemExit("headline arms are not protocol matched: " + repr(mismatches))
if sft.get("curator") != "uniform" or ac.get("curator") != "osmd":
    raise SystemExit("unexpected headline curator configuration")
print("protocol audit: SFT and Exec-AC actor/rollout budgets match")
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
eval_arm exec_ac

python3 - "$ROOT/sft/selection.json" "$ROOT/exec_ac/selection.json" "$ROOT/go_no_go.json" "$MIN_VAL_GAIN" <<'PY'
import json, sys
from pathlib import Path
sft = json.load(open(sys.argv[1]))["selected"]
ac = json.load(open(sys.argv[2]))["selected"]
threshold = float(sys.argv[4])
delta = float(ac["strict_success_rate"]) - float(sft["strict_success_rate"])
decision = {
    "metric": "held_out_validation_strict_success_rate",
    "sft": sft,
    "exec_ac": ac,
    "delta_exec_ac_minus_sft": delta,
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

for name in sft exec_ac; do
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
  --method "$ROOT/exec_ac/strict_test.json" \
  --out "$ROOT/compare_exec_ac_vs_sft.json"

echo "Completed exactly two headline arms under $ROOT"
