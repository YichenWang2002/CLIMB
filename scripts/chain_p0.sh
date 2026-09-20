#!/bin/bash
cd .
PY=/root/miniconda3/bin/python
echo "[p0] regression smoke: SPCL+SCD topology 16题 (期望14/16)"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out /tmp/reg_scd16.json --limit 16 --batch-size 16 --level topology > /tmp/reg_scd16.log 2>&1
echo "[p0] smoke exit $?"
echo "[p0] Flat+SCD full 600"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_flat/stage1 \
  --out results/outputs/flat_scd_topology_test.json --batch-size 48 --level topology > /tmp/flat_scd.log 2>&1
echo "[p0] flat+scd exit $?"
echo "[p0] Flat+SCD+EGVD k=8 (ckpt 续跑支持)"
$PY -u -m eval.egvd --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_flat/stage1 \
  --out results/outputs/egvd_flat_constrained_k8_t07.json \
  --k 8 --temperature 0.7 --batch-size 16 --constrained --save-generations > /tmp/flat_egvd.log 2>&1
echo "[p0] flat+egvd exit $?"
echo "[p0] ALL DONE $(date)"