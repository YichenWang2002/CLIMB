#!/bin/bash
cd .
PY=/root/miniconda3/bin/python
OUT=results/outputs
echo "[orch2] start $(date)"
for S in 43 44; do
  echo "[orch2] seed$S constrained greedy eval"
  $PY -u -m eval.eval_constrained --data data/test.jsonl \
    --adapter outputs/checkpoints/spcl_v2_s$S/stage3 \
    --out $OUT/spcl_v2_s${S}_constrained_test.json --batch-size 48 > /tmp/s${S}_mcd.log 2>&1
  echo "[orch2] seed$S constrained eval done (exit $?)"
  echo "[orch2] seed$S constrained EGVD k=8"
  $PY -u -m eval.egvd --data data/test.jsonl \
    --adapter outputs/checkpoints/spcl_v2_s$S/stage3 \
    --out $OUT/egvd_v2_s${S}_constrained_k8_t07.json \
    --k 8 --temperature 0.7 --batch-size 16 --constrained --save-generations > /tmp/s${S}_egvd.log 2>&1
  echo "[orch2] seed$S EGVD done (exit $?)"
done
echo "[orch2] ALL DONE $(date)"
