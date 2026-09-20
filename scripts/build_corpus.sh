#!/bin/bash
# Full corpus construction from scratch: STRIPS sample -> plan -> BT compile
# -> executor validation -> NL rewriting (needs LLM API key) -> splits,
# then the primitive-renaming augmentation used for all paper training runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

echo "=== [1/2] sample + validate + NL-rewrite the three splits ==="
python -m datagen.build_dataset --full --workers 16

echo "=== [2/2] primitive-renaming augmentation (train_aug10) ==="
python -m datagen.rename_skills \
  --train data/train.jsonl --out data/train_aug10.jsonl \
  --frac 0.10 --seed 123 --workers 16 --include-faulted --double-frac 0.2
