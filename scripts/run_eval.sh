#!/bin/bash
# Symbolic-execution evaluation of an adapter on the held-out benchmark.
# Usage: bash scripts/run_eval.sh ADAPTER OUT_STEM [SPLIT]
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:?usage: run_eval.sh ADAPTER OUT_STEM [SPLIT]}"
STEM="${2:?usage: run_eval.sh ADAPTER OUT_STEM [SPLIT]}"
SPLIT="${3:-data/test.jsonl}"

python -u -m eval.evaluate --data "${SPLIT}" \
  --adapter "${ADAPTER}" \
  --out "outputs/results/${STEM}.json" --batch-size 64 --save-generations
