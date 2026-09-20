#!/bin/bash
# Fair MT-DUCL v2: three independent one-epoch optimizer/LR resets.
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
ROOT=outputs/mt_ducl_v2_sanity
NAME=mt_ducl_v2_sanity
mkdir -p "$ROOT"

PREV=""
for round in 0 1 2; do
  SCORE="$ROOT/scores_round${round}.jsonl"
  ORDERED="$ROOT/ordered_round${round}.jsonl"
  if [ -n "$PREV" ]; then
    ADAPTER_ARG=(--adapter "$PREV")
    UTILITY_ARG=(--utility-source "$ROOT/scores_round0.jsonl")
  else
    ADAPTER_ARG=()
    UTILITY_ARG=()
  fi

  if [ -s "$SCORE" ] && [ -s "$ORDERED" ] && [ -s "${SCORE%.jsonl}.report.json" ]; then
    echo "=== MT-DUCL v2 round $round: reuse existing scores ==="
  else
    echo "=== MT-DUCL v2 round $round: score current model ==="
    python3 -u -m curriculum.score_mt_ducl \
      --train "$TRAIN" --val "$VAL" \
      --out "$SCORE" --ordered-out "$ORDERED" \
      "${ADAPTER_ARG[@]}" "${UTILITY_ARG[@]}" \
      --batch-size 4 --max-len 2560 --embed-device cpu \
      --alpha 0.8 --seed 42
  fi

  RUN_DIR="outputs/checkpoints/${NAME}_round$((round + 1))"
  echo "=== MT-DUCL v2 round $((round + 1)): ordered SFT ==="
  INIT_ARG=()
  if [ -n "$PREV" ]; then
    INIT_ARG=(--init-adapter "$PREV")
  fi
  python3 -u -m training.sft_lora --mode flat \
    --stages "$ORDERED" --val "$VAL" \
    --run-name "${NAME}_round$((round + 1))" \
    --epochs 1 --lr 1e-4 --preserve-order --seed 42 "${INIT_ARG[@]}"
  PREV="$RUN_DIR/stage1"
done

echo "=== MT-DUCL v2 validation ==="
python3 -u -m eval.evaluate --data "$VAL" \
  --adapter "$PREV" --out mt_ducl_v2_sanity_val.json \
  --batch-size 64 --save-generations

echo "=== MT-DUCL v2 held-out test ==="
python3 -u -m eval.evaluate --data "$TEST" \
  --adapter "$PREV" --out mt_ducl_v2_sanity_test.json \
  --batch-size 64 --save-generations

echo "=== MT-DUCL v2 sanity complete ==="
