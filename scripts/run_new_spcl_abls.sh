#!/bin/bash
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
BOOSTS='1,1|1,1,1.5|0.5,0.75,1.5,3'

run_one () {
  local name="$1"; local cur="$2"; local ckpt="spcl_${name}"
  echo "===== ${name}: build curriculum ====="
  shift 2
  "$PY" -u -m curriculum.build_spcl \
    --train "$TRAIN" --val "$VAL" --out-dir "$cur" \
    --rounds 3 --buckets 4 --window 2,3,4 \
    --boosts "$BOOSTS" --batch-size 8 --max-len 2560 \
    --embed-device cpu --seed 42 "$@"
  echo "===== ${name}: staged SFT ====="
  "$PY" -u -m training.sft_lora --mode staged \
    --stages "$cur/round1.jsonl" "$cur/round2.jsonl" "$cur/round3.jsonl" \
    --val "$VAL" --run-name "$ckpt" --epochs 1 --lr 1e-4 --seed 42
  echo "===== ${name}: 600-row evaluation (filter T2/T3 to 480 in analysis) ====="
  "$PY" -u -m eval.evaluate --data "$TEST" \
    --adapter "outputs/checkpoints/$ckpt/stage3" \
    --out "results/outputs/${ckpt}_test.json" \
    --batch-size 64 --save-generations
  echo "===== ${name}: done ====="
}

# 1) Replace every PCA fusion by uniform rank averaging.
run_one simpleavg outputs/spcl_simpleavg_seed42 --fusion mean

# 2) Drop the structural difficulty view; retain semantic difficulty,
# utility, expanding windows, and the reference boosts.
run_one nostruct outputs/spcl_nostruct_seed42 --no-structural-view

# 3) Conventional full-pool curriculum baseline: uniform utility and
# explicit 3x oversampling of the hardest bucket in every round.
run_one uniform_hard3 outputs/spcl_uniform_hard3_seed42 \
  --no-utility --window 4,4,4 \
  --boosts '1,1,1,3|1,1,1,3|1,1,1,3'

echo 'ALL NEW SPCL ABLATIONS COMPLETE'
