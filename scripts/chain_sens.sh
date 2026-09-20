#!/bin/bash
cd .
PY=/root/miniconda3/bin/python
echo "[sens] waiting for P0 chain (flat EGVD) ..."
while pgrep -f "chain_p0.sh" > /dev/null || pgrep -f "spcl_cmp_flat/stage1" > /dev/null; do sleep 60; done
echo "[sens] P0 chain finished, start T=0.3 sensitivity $(date)"
$PY -u -m eval.egvd --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out results/outputs/egvd_v2_constrained_k8_t03.json \
  --k 8 --temperature 0.3 --batch-size 16 --constrained --save-generations > /tmp/sens_t03.log 2>&1
echo "[sens] t03 exit $?"
echo "[sens] ALL DONE $(date)"