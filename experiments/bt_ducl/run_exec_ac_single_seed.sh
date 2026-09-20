#!/usr/bin/env bash
# Executor-guided online curriculum protocol.
#
# The warm start is ordinary full-completion SFT.  OSMD and uniform-online
# then receive identical rollout and actor-update budgets; only the curator
# update differs.  Validation is used for checkpoint selection and the test
# set is touched exactly once, after the pilot gate and final selection.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_exec_ac_seed${SEED}}"
TRAIN="${TRAIN:-outputs/dataset/train_aug10.jsonl}"
VAL="${VAL:-outputs/dataset/val.jsonl}"
TEST="${TEST:-outputs/dataset/test.jsonl}"
BASE="${BASE:-models/llama32-1b}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
PILOT_UPDATES="${PILOT_UPDATES:-20}"
ONLINE_UPDATES="${ONLINE_UPDATES:-225}"
ONLINE_LR="${ONLINE_LR:-1e-5}"
CANDIDATE_SIZE="${CANDIDATE_SIZE:-64}"
SELECT_SIZE="${SELECT_SIZE:-16}"
MICRO_BATCH="${MICRO_BATCH:-4}"
ROLLOUTS_PER_ARM="${ROLLOUTS_PER_ARM:-4}"
LOGPROB_BATCH="${LOGPROB_BATCH:-2}"
GENERATION_BATCH="${GENERATION_BATCH:-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
MAX_NEW="${MAX_NEW:-1400}"
FORCE_FULL="${FORCE_FULL:-0}"

mkdir -p "$ROOT"

if [ ! -f "$ROOT/warmup/flat/train_metadata.json" ]; then
  python3 -u -m experiments.bt_ducl.train_v4 --mode flat \
    --train "$TRAIN" --val "$VAL" --run-dir "$ROOT/warmup/flat" \
    --epochs "$WARMUP_EPOCHS" --batch 4 --accum 4 --lr 1e-4 \
    --max-len 2560 --seed "$SEED" --loss-scope completion
fi

python3 - "$ROOT/warmup/flat/train_metadata.json" "$ROOT/warmup/flat" "$VAL" "$MAX_NEW" <<'PY'
import json, subprocess, sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())
out_dir, val, max_new = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
for step in metadata["checkpoint_steps"]:
    out = out_dir / "strict_val" / f"val_step_{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        continue
    adapter = out_dir / f"checkpoint-{step}"
    subprocess.run(["python3", "-u", "-m", "experiments.bt_ducl.strict_eval",
                    "--data", val, "--adapter", str(adapter), "--out", str(out),
                    "--batch-size", "64", "--max-new", max_new],
                   check=True)
PY
python3 -u -m experiments.bt_ducl.select_checkpoint \
  --eval-dir "$ROOT/warmup/flat/strict_val" \
  --out "$ROOT/warmup/flat/selection.json"

WARM_START="$(python3 - "$ROOT/warmup/flat/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["adapter"])
PY
)"

run_online() {
  local curator="$1" updates="$2" out="$3"
  if [ -f "$out/train_metadata.json" ]; then return; fi
  python3 -u -m experiments.bt_ducl.exec_ac_sft \
    --train "$TRAIN" --warm-start "$WARM_START" --base "$BASE" \
    --run-dir "$out" --curator "$curator" --updates "$updates" \
    --lr "$ONLINE_LR" \
    --candidate-size "$CANDIDATE_SIZE" --select-size "$SELECT_SIZE" \
    --micro-batch "$MICRO_BATCH" --rollouts-per-arm "$ROLLOUTS_PER_ARM" \
    --max-new "$MAX_NEW" --logprob-batch "$LOGPROB_BATCH" \
    --generation-batch "$GENERATION_BATCH" \
    --save-every "$SAVE_EVERY" --seed "$SEED"
}

run_online uniform "$PILOT_UPDATES" "$ROOT/pilot/uniform"
run_online osmd "$PILOT_UPDATES" "$ROOT/pilot/osmd"

eval_online_val() {
  local run="$1"
  python3 - "$run" "$VAL" "$MAX_NEW" <<'PY'
import json, subprocess, sys
from pathlib import Path
run, val, max_new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
metadata = json.loads((run / "train_metadata.json").read_text())
steps = sorted(int(path.name.split("-")[-1]) for path in run.glob("checkpoint-*")
               if path.is_dir())
if not steps:
    raise SystemExit(f"no online checkpoints found under {run}")
for step in steps:
    out = run / "strict_val" / f"val_step_{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        continue
    subprocess.run(["python3", "-u", "-m", "experiments.bt_ducl.strict_eval",
                    "--data", val, "--adapter", str(run / f"checkpoint-{step}"),
                    "--out", str(out), "--batch-size", "64", "--max-new", max_new],
                   check=True)
PY
  python3 -u -m experiments.bt_ducl.select_checkpoint \
    --eval-dir "$run/strict_val" --out "$run/selection.json"
}
eval_online_val "$ROOT/pilot/uniform"
eval_online_val "$ROOT/pilot/osmd"

python3 - "$ROOT/pilot/uniform/selection.json" "$ROOT/pilot/osmd/selection.json" "$FORCE_FULL" <<'PY'
import json, sys
u = json.load(open(sys.argv[1]))["selected"]["strict_success_rate"]
o = json.load(open(sys.argv[2]))["selected"]["strict_success_rate"]
force = bool(int(sys.argv[3]))
print(f"pilot gate: uniform={u:.4f} osmd={o:.4f} delta_pp={(o-u)*100:.2f}")
if not force and o < u + 0.03:
    raise SystemExit("ABORT: OSMD pilot did not exceed matched uniform-online control by 3pp; set FORCE_FULL=1 only for diagnostic runs.")
PY

run_online uniform "$ONLINE_UPDATES" "$ROOT/final/uniform"
run_online osmd "$ONLINE_UPDATES" "$ROOT/final/osmd"
eval_online_val "$ROOT/final/uniform"
eval_online_val "$ROOT/final/osmd"

for pair in "warmup/flat" "final/uniform" "final/osmd"; do
  run="$ROOT/$pair"
  adapter="$(python3 - "$run/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["adapter"])
PY
)"
  python3 -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$adapter" --out "$run/strict_test.json" \
    --batch-size 64 --max-new "$MAX_NEW"
done

python3 -u -m experiments.bt_ducl.compare \
  --flat "$ROOT/warmup/flat/strict_test.json" \
  --method "$ROOT/final/uniform/strict_test.json" \
  --out "$ROOT/compare_uniform.json"
python3 -u -m experiments.bt_ducl.compare \
  --flat "$ROOT/warmup/flat/strict_test.json" \
  --method "$ROOT/final/osmd/strict_test.json" \
  --out "$ROOT/compare_osmd.json"
python3 -u -m experiments.bt_ducl.compare \
  --flat "$ROOT/final/uniform/strict_test.json" \
  --method "$ROOT/final/osmd/strict_test.json" \
  --out "$ROOT/compare_osmd_vs_uniform.json"

echo "Completed executor-guided SFT/online-curriculum protocol under $ROOT"
