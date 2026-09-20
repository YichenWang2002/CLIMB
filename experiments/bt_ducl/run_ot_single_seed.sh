#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_ducl_ot_seed${SEED}}"
TRAIN="outputs/dataset/train_aug10.jsonl"
VAL="outputs/dataset/val.jsonl"
TEST="outputs/dataset/test.jsonl"
mkdir -p "$ROOT"

python3 -u -m experiments.bt_ducl.audit --train "$TRAIN" --val "$VAL" --test "$TEST" \
  > "$ROOT/audit.json" 2>&1
python3 -u -m experiments.bt_ducl.scoring_ot \
  --train "$TRAIN" --val "$VAL" --out-dir "$ROOT/scoring" \
  --score-batch-size 2 --embed-batch-size 64 --batch-size 16 \
  --max-len 2560 --embed-device cpu --ot-device cuda \
  --alpha 0.8 --seed "$SEED"

python3 -u -m experiments.bt_ducl.train --mode flat \
  --train "$TRAIN" --val "$VAL" --run-dir "$ROOT/flat" \
  --epochs 3 --batch 4 --accum 4 --lr 1e-4 --max-len 2560 \
  --sampler-alpha 0.8 --curriculum-epochs 0 --seed "$SEED"
python3 -u -m experiments.bt_ducl.train --mode ducl \
  --train "$TRAIN" --ordered "$ROOT/scoring/scores.jsonl" --val "$VAL" \
  --run-dir "$ROOT/ducl_ce1" --epochs 3 --batch 4 --accum 4 --lr 1e-4 \
  --max-len 2560 --sampler-alpha 0.8 --curriculum-epochs 1 --seed "$SEED"

python3 -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$ROOT/flat" --out "$ROOT/flat_test.json" --batch-size 64 --max-new 1400
python3 -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$ROOT/ducl_ce1" --out "$ROOT/ducl_test.json" --batch-size 64 --max-new 1400
python3 -u -m experiments.bt_ducl.compare \
  --flat "$ROOT/flat_test.json" --method "$ROOT/ducl_test.json" \
  --out "$ROOT/comparison.json"
