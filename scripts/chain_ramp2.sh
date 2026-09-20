#!/bin/bash
# Sensitivity battery: window/rounds/boost-ratio variants (same budget, seed 42)
# V1 win1234: 4 rounds, window 1,2,3,4, round-size 4500 (total = 3x6000)
# V2 win124 : 3 rounds, window 1,2,4 (start from easiest bucket)
# V3 geo2   : window 2,3,4, geometric boost ratio 8
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

run_variant () {
  NAME=$1; ROUNDS=$2; WINDOW=$3; BOOSTS=$4; RSIZE=$5
  CUR=outputs/spcl_${NAME}_seed42
  LAST=stage${ROUNDS}
  echo "=== [${NAME}] build curriculum (rounds=${ROUNDS} window=${WINDOW}) ==="
  $PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
    --rounds "$ROUNDS" --buckets 4 --window "$WINDOW" --boosts "$BOOSTS" \
    --round-size "$RSIZE" --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 || return 1
  STAGES=""; for r in $(seq 1 "$ROUNDS"); do STAGES="$STAGES $CUR/round${r}.jsonl"; done
  echo "=== [${NAME}] staged training ==="
  $PY -u -m training.sft_lora --mode staged --stages $STAGES \
    --val "$VAL" --run-name "spcl_${NAME}" --epochs 1 --lr 1e-4 --seed 42 || return 1
  echo "=== [${NAME}] held-out test eval ==="
  $PY -u -m eval.evaluate --data "$TEST" \
    --adapter "outputs/checkpoints/spcl_${NAME}/${LAST}" \
    --out "results/outputs/spcl_${NAME}_test.json" --batch-size 64 --save-generations || return 1
  echo "=== ${NAME} DONE ==="
}

run_variant win1234 4 "1,2,3,4" "1|1,1.5|0.75,1.5,2.25|0.5,0.75,1.5,3" 4500
run_variant win124  3 "1,2,4"   "1|1,1.5|0.5,0.75,1.5,3" 0
run_variant geo2    3 "2,3,4"   "1,1|1,2,4|1,2,4,8" 0
echo "=== ALL SENSITIVITY VARIANTS DONE ==="
