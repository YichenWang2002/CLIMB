#!/usr/bin/env bash
# v5 protocol: full-completion loss (both arms), permutation curriculum,
# constant LR, val-based checkpoint selection, test untouched.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_ducl_v5_seed${SEED}}"
TRAIN="outputs/dataset/train_aug10.jsonl"
VAL="outputs/dataset/val.jsonl"
TEST="outputs/dataset/test.jsonl"
# Reuse existing scores (expensive LM NLL pass) unless a fresh run is wanted:
#   SCORES=outputs/bt_ducl_v4_seed42/scoring/scores.jsonl bash run_v4_single_seed.sh
SCORES="${SCORES:-}"
mkdir -p "$ROOT"

if [ -z "$SCORES" ]; then
  python3 -u -m experiments.bt_ducl.scoring_v4 \
    --train "$TRAIN" --val "$VAL" --out-dir "$ROOT/scoring" \
    --score-batch-size 2 --embed-batch-size 64 --max-len 2560 \
    --embed-device cpu --ot-device cuda --batch-size 16 \
    --temperature 0.08 --epsilon 0.2 --blur 0.1 --seed "$SEED"
  SCORES="$ROOT/scoring/scores.jsonl"
fi

python3 -u -m experiments.bt_ducl.train_v4 --mode flat \
  --train "$TRAIN" --val "$VAL" --run-dir "$ROOT/flat" \
  --epochs 3 --batch 4 --accum 4 --lr 1e-4 --max-len 2560 --seed "$SEED" \
  --loss-scope completion

# ---- Catastrophe gate -------------------------------------------------------
# Evaluate the FIRST saved checkpoint of the baseline arm on val before
# spending any DUCL compute. The failed semantic-mask run was already 0/600
# here; this gate would have stopped the pipeline at that point.
# (188 = mid-epoch-1 for 6000 rows / effective batch 16 / 3 epochs.)
mkdir -p "$ROOT/flat/strict_val"
GATE_OUT="$ROOT/flat/strict_val/val_step_188.json"
python3 -u -m experiments.bt_ducl.strict_eval --data "$VAL" \
  --adapter "$ROOT/flat/checkpoint-188" \
  --out "$GATE_OUT" --batch-size 64 --max-new 1400
python3 - "$GATE_OUT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
rate = float(value["strict_success_rate"])
parse_fail = sum(v for k, v in value["fail_reasons"].items() if k.startswith("parse:"))
parse_ok = 1.0 - parse_fail / value["n"]
print(f"gate: flat checkpoint-188 val strict={rate:.4f} parse_ok={parse_ok:.3f}")
if rate < 0.01 or parse_ok < 0.75:
    print("ABORT: baseline is structurally broken (loss/pipeline regression).")
    print("Do NOT start DUCL; fix the shared setup first.")
    sys.exit(1)
PY
# ------------------------------------------------------------------------------

python3 -u -m experiments.bt_ducl.train_v4 --mode ducl \
  --train "$TRAIN" --scores "$SCORES" --val "$VAL" \
  --run-dir "$ROOT/ducl" --epochs 3 --batch 4 --accum 4 --lr 1e-4 \
  --max-len 2560 --seed "$SEED" --loss-scope completion

for mode in flat ducl; do
  mkdir -p "$ROOT/$mode/strict_val"
  for step in 188 375 563 750 938 1125; do
    out="$ROOT/$mode/strict_val/val_step_${step}.json"
    [ -f "$out" ] && continue   # idempotent resume; gate already wrote 188
    python3 -u -m experiments.bt_ducl.strict_eval --data "$VAL" \
      --adapter "$ROOT/$mode/checkpoint-$step" \
      --out "$out" --batch-size 64 --max-new 1400
  done
  python3 -m experiments.bt_ducl.select_checkpoint \
    --eval-dir "$ROOT/$mode/strict_val" --out "$ROOT/$mode/selection.json"
done

# The selected adapters are read from selection.json. Test is intentionally
# not run here so checkpoint selection cannot accidentally inspect test output.
