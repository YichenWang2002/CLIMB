#!/bin/bash
# Symbolic-execution evaluation of an adapter on the held-out suite.
# Usage: bash scripts/run_eval.sh ADAPTER OUT_STEM [SPLIT]
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8

ADAPTER="${1:?usage: run_eval.sh ADAPTER OUT_STEM [SPLIT]}"
STEM="${2:?usage: run_eval.sh ADAPTER OUT_STEM [SPLIT]}"
SPLIT="${3:-data/test.jsonl}"

python -u -m eval.evaluate --data "${SPLIT}" \
  --adapter "${ADAPTER}" \
  --out "results/${STEM}.json" --batch-size 64 --save-generations
