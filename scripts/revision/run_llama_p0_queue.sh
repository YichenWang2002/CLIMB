#!/usr/bin/env bash
# P0 queue: attribution controls and deployed EGVD, isolated from historical outputs.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-/root/miniconda3/bin/python}
BASE=models/llama32-1b
VAL=data/val.jsonl
TEST=data/test.jsonl
ROOT=outputs/revision/attribution_v2

train_eval_control() {
  local seed=$1 arm=$2
  local runroot="$ROOT/seed${seed}"
  local ckpt="$runroot/checkpoints/$arm/stage3"
  local out="$runroot/eval/${arm}_test.json"
  if [[ -f "$out" && -d "$ckpt" ]]; then
    echo "[skip] seed=${seed} arm=${arm} already complete"
    return
  fi
  mkdir -p "$runroot/eval"
  "$PY" -u -m training.sft_lora --mode staged \
    --stages "$runroot/curriculum/$arm/round1.jsonl" \
             "$runroot/curriculum/$arm/round2.jsonl" \
             "$runroot/curriculum/$arm/round3.jsonl" \
    --val "$VAL" --run-name "$arm" --checkpoint-root "$runroot/checkpoints" \
    --base "$BASE" --epochs 1 --lr 1e-4 --batch 8 --accum 2 \
    --max-len 2560 --loss-type nll --seed "$seed"
  "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$ckpt" --base "$BASE" --out "$out" \
    --batch-size 64 --max-new 1400
}

 # seed42 anti is launched separately in the interactive run. The queue
 # skips it after completion and still trains a clean hardtail seed42.
for seed in 42 43 44; do
  for arm in anti hardtail; do
    train_eval_control "$seed" "$arm"
  done
done

# Keep historical EGVD files intact and write a clean P0 rerun for the
# reviewer-requested SPCL seeds 43, 44, and 45.
for seed in 43 44 45; do
  out="results/outputs/egvd_v2_s${seed}_rerun_constrained_k8_t07.json"
  adapter="outputs/checkpoints/spcl_v2_s${seed}/stage3"
  if [[ -f "$out" ]]; then
    echo "[skip] EGVD seed=${seed} rerun already complete"
    continue
  fi
  if ! "$PY" -u -m eval.egvd --data "$TEST" --adapter "$adapter" --base "$BASE" \
      --out "$out" --k 8 --temperature 0.7 --top-p 0.95 \
      --batch-size "${EGVD_BATCH:-32}" --max-new 1400 --seed "$seed" \
      --constrained --save-generations; then
    echo "[retry] EGVD seed=${seed} with batch-size=16"
    "$PY" -u -m eval.egvd --data "$TEST" --adapter "$adapter" --base "$BASE" \
      --out "$out" --k 8 --temperature 0.7 --top-p 0.95 --batch-size 16 \
      --max-new 1400 --seed "$seed" --constrained --save-generations
  fi
done
echo "P0 queue complete: $(date -Is)"
