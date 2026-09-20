#!/bin/bash
# Attribution-control battery for SPCL (same budget as v2: 3 rounds x n_train, seed 42)
# nou1    : utility ablated (U=1), curriculum + boosts intact
# perm    : bucket labels permuted (destroy difficulty ordering), utility + boosts intact
# mix     : no curriculum (full pool every round), fixed mixture = v2 marginal bucket dist
# hardtail: no curriculum, full pool, static v2 round-3 boost profile every round
# anti    : anti-curriculum (hardest prefix first), mirrored boost profile
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
V2BOOSTS="1,1|1,1,1.5|0.5,0.75,1.5,3"

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

echo "=== [nou1] build: window 2,3,4 + v2 boosts + no-utility ==="
CUR=outputs/spcl_nou1_seed42
$PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window "2,3,4" --boosts "$V2BOOSTS" --no-utility \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval nou1 3 "$CUR"

echo "=== [perm] build: window 2,3,4 + v2 boosts + shuffled buckets ==="
CUR=outputs/spcl_perm_seed42
$PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window "2,3,4" --boosts "$V2BOOSTS" --shuffle-buckets \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval perm 3 "$CUR"

echo "=== [mix] build: window 4,4,4 + fixed mixture (v2 marginal) ==="
CUR=outputs/spcl_mix_seed42
$PY -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window "4,4,4" --mixture "0.3049,0.3332,0.2216,0.1404" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval mix 3 "$CUR"

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
  --rounds 3 --buckets 4 --window "2,3,4" --boosts "$V2BOOSTS" --reverse \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42 && \
run_train_eval anti 3 "$CUR"

echo "=== ALL ATTRIBUTION VARIANTS DONE ==="
