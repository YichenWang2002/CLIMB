#!/bin/bash
# Supervision-source ablation for the FULL system (SPCL training + SCD decoding).
#
# Leave-one-out logic, matching how the paper's SPCL x SCD 2x2 works:
#   arm P (already trained, zero cost): planner data  x SPCL -> 62.0, +SCD -> 66.8
#   arm L (this script): LLM-teacher data x SPCL -> eval +-SCD
# The two arms share the same 6,000 aug10 inputs byte-for-byte; only the gold
# XML differs (planner-compiled vs DeepSeek-5-shot, both executor-verified).
# SPCL re-derives its schedule from each arm's OWN data with identical
# hyperparameters (rounds/buckets/window/boosts/seed) -- the method is held
# fixed, the data is replaced. Delta on the +SCD column = the supervision-data
# contribution inside the deployed system.
#
# Step 1 needs OPENAI_API_KEY; everything else is local/GPU. LLM responses are
# disk-cached, so reruns never re-pay for generation.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3

TRAIN_AUG=data/train_aug10.jsonl        # canonical planner inputs
TRAIN_LLMT=data/train_llmteacher.jsonl  # same inputs, LLM trees
VAL=data/val.jsonl
TEST=data/test.jsonl
PLANER_SCD=results/outputs/spcl_v2_constrained_test.json  # 66.8 row
NAME=llmteacher_spcl
CUR=outputs/spcl_llmteacher_seed42

echo "=== [0/5] offline wiring check: planner golds must pass ~100% ==="
$PY -m datagen.build_llm_teacher --selftest --limit 200 \
    --train "$TRAIN_AUG" --out /tmp/unused.jsonl

echo "=== [1/5] build LLM-teacher file on the same 6,000 inputs ==="
$PY -u -m datagen.build_llm_teacher \
    --train "$TRAIN_AUG" --out "$TRAIN_LLMT" --max-rounds 4

echo "=== [2/5] SPCL: identical hyperparameters, schedule re-derived ==="
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN_LLMT" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42

echo "=== [3/5] staged SPCL training (3 paced rounds = matched budget) ==="
$PY -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name "$NAME" --epochs 1 --lr 1e-4 --seed 42

echo "=== [4/5] test eval: unconstrained and +SCD(topology) ==="
$PY -u -m eval.evaluate --data "$TEST" --adapter "outputs/checkpoints/$NAME/stage3" \
  --out "outputs/${NAME}_test.json" --batch-size 64 --save-generations
$PY -u -m eval.eval_constrained --data "$TEST" --adapter "outputs/checkpoints/$NAME/stage3" \
  --out "outputs/${NAME}_scd_test.json" --batch-size 48 --level topology

echo "=== [5/5] paired McNemar vs planner full system (66.8) ==="
$PY -m eval.paired_test "outputs/${NAME}_scd_test.json" "$PLANER_SCD"

# ---- SECONDARY (optional): data x training interaction row ----------------
# Completes the 2x2 {planner, llm} x {flat, spcl}; the planner-flat cell is
# the existing 55.8 baseline. A wider gap under SPCL than under flat means
# planner data and SPCL synergize (structural staging needs clean trees);
# a narrower gap means curriculum partly compensates for weaker supervision.
# $PY -u -m training.sft_lora --mode flat --stages "$TRAIN_LLMT" \
#   --val "$VAL" --run-name llmteacher_flat --epochs 3 --lr 1e-4 --seed 42
# $PY -u -m eval.evaluate --data "$TEST" --adapter outputs/checkpoints/llmteacher_flat/stage1 \
#   --out outputs/llmteacher_flat_test.json --batch-size 64 --save-generations
# $PY -u -m eval.eval_constrained --data "$TEST" --adapter outputs/checkpoints/llmteacher_flat/stage1 \
#   --out outputs/llmteacher_flat_scd_test.json --batch-size 48 --level topology

echo "=== DONE ==="
