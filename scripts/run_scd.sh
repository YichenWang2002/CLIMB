#!/bin/bash
# SCD (Symbolic Constrained Decoding): greedy decoding with platform-map
# constraints on identifier/relation attribute values. Zero extra token cost.
# Usage: bash scripts/run_scd.sh ADAPTER OUT_STEM [LEVEL]
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8

ADAPTER="${1:?usage: run_scd.sh ADAPTER OUT_STEM [LEVEL]}"
STEM="${2:?usage: run_scd.sh ADAPTER OUT_STEM [LEVEL]}"
LEVEL="${3:-topology}"   # topology (full SCD) | type | grammar

python -u -m eval.eval_constrained --data data/test.jsonl \
  --adapter "${ADAPTER}" --level "${LEVEL}" \
  --out "results/${STEM}.json" --batch-size 48
