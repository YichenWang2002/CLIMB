#!/bin/bash
cd .
PY=/root/miniconda3/bin/python
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out /tmp/smoke_grammar6.json --limit 16 --batch-size 16 --level grammar > /tmp/smoke_g6.log 2>&1
echo "smoke exit $?"
$PY -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_cmp_method_v2/stage3 \
  --out results/outputs/spcl_v2_scd_grammar_test.json --batch-size 48 --level grammar > /tmp/abl_grammar.log 2>&1
echo "grammar exit $?"
