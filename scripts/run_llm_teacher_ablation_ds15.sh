#!/bin/bash
# Supervision-source ablation, backbone-generalization leg: same two arms on
# DeepSeek-R1-Distill-Qwen-1.5B instead of Llama-3.2-1B.
#
#   arm P: planner data (train_aug10)    x SPCL x {,-SCD}
#   arm L: LLM-teacher data (same inputs) x SPCL x {,-SCD}
#
# Reuses train_llmteacher.jsonl from the llama leg (no API cost). SPCL
# difficulty is re-scored with the 1.5B base (--base), so each arm's schedule
# is model-target on this backbone; all other hyperparameters identical.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3

BASE15=../model/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN_MATCH=data/train_planner_matched.jsonl
TRAIN_LLMT=data/train_llmteacher.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

if [ ! -s "$TRAIN_MATCH" ] || [ ! -s "$TRAIN_LLMT" ]; then
  echo "REFUSING: matched files missing (run run_llm_teacher_matched.sh first)."
  exit 1
fi
echo "matched arms: planner=$(wc -l < "$TRAIN_MATCH") rows, llmteacher=$(wc -l < "$TRAIN_LLMT") rows"

run_arm () {  # $1 = train file, $2 = tag (planner|llmteacher)
  local TRAIN=$1 TAG=$2
  local CUR=outputs/spcl_${TAG}_ds15_seed42
  local NAME=${TAG}_spcl_ds15
  echo "=== [$TAG] build SPCL curriculum on the 1.5B base ==="
  $PY -u -m curriculum.build_spcl \
    --train "$TRAIN" --val "$VAL" --out-dir "$CUR" --base "$BASE15" \
    --rounds 3 --buckets 4 --window 2,3,4 \
    --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
    --batch-size 8 --max-len 2560 --embed-device cpu --seed 42
  echo "=== [$TAG] staged SPCL training ==="
  $PY -u -m training.sft_lora --mode staged --base "$BASE15" \
    --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
    --val "$VAL" --run-name "$NAME" --epochs 1 --lr 1e-4 --seed 42
  echo "=== [$TAG] test eval: unconstrained and +SCD(topology) ==="
  $PY -u -m eval.evaluate --data "$TEST" --base "$BASE15" \
    --adapter "outputs/checkpoints/$NAME/stage3" \
    --out "outputs/${NAME}_test.json" --batch-size 64 --save-generations
  $PY -u -m eval.eval_constrained --data "$TEST" --base "$BASE15" \
    --adapter "outputs/checkpoints/$NAME/stage3" \
    --out "outputs/${NAME}_scd_test.json" --batch-size 48 --level topology
}

run_arm "$TRAIN_MATCH" planner
run_arm "$TRAIN_LLMT" llmteacher

echo "=== paired McNemar, full system (SPCL+SCD) on the 1.5B backbone ==="
$PY -m eval.paired_test outputs/llmteacher_spcl_ds15_scd_test.json \
    outputs/planner_spcl_ds15_scd_test.json

echo "=== DS15 DONE ==="
