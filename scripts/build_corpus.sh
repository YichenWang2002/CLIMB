#!/bin/bash
# Corpus construction: STRIPS sample -> forward-search plan -> BT compile ->
# executor validation -> natural-language rewriting -> splits.
# The NL step calls an OpenAI-compatible API (set OPENAI_API_KEY / OPENAI_BASE_URL).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m datagen.build_dataset --full --workers 16
python -m datagen.rename_skills \
  --train data/train.jsonl --out data/train_aug.jsonl \
  --frac 0.10 --seed 123 --include-faulted --double-frac 0.2
