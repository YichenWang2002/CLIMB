#!/bin/bash
# Attribution battery part 2: mix (shuffled realized draws) + hardtail + anti
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl

run_train_eval () {
  NAME=$1; ROUNDS=$2; CUR=$3
  LAST=stage${ROUNDS}
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

echo "=== [mixs] shuffled realized v2 draws (prebuilt rounds, no build step) ==="
run_train_eval mixs 3 outputs/spcl_mixs_seed42

echo "=== [hardtail] build: window 4,4,4 + static v2 round-3 boosts ==="
CUR=outputs/spcl_hardtail_seed42
$PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window "4,4,4" \
  --boosts "0.5,0.75,1.5,3|0.5,0.75,1.5,3|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval hardtail 3 "$CUR"

echo "=== [anti] build: window 2,3,4 + v2 boosts + reverse ==="
CUR=outputs/spcl_anti_seed42
$PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window "2,3,4" --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" --reverse \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval anti 3 "$CUR"

echo "=== ALL ATTR2 DONE ==="
