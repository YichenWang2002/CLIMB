#!/bin/bash
cd .
PY=/root/miniconda3/bin/python
echo "[abl] grammar smoke16 $(date)"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out /tmp/smoke_grammar5.json --limit 16 --batch-size 16 --level grammar > /tmp/smoke_g5.log 2>&1
echo "[abl] smoke done (exit $?)"
echo "[abl] type full 600"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out results/outputs/spcl_v2_scd_type_test.json --batch-size 48 --level type > /tmp/abl_type.log 2>&1
echo "[abl] type done (exit $?)"
echo "[abl] grammar full 600"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out results/outputs/spcl_v2_scd_grammar_test.json --batch-size 48 --level grammar > /tmp/abl_grammar.log 2>&1
echo "[abl] grammar done (exit $?)"
echo "[abl] ALL DONE $(date)"
