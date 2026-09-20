#!/usr/bin/env bash
# Protocol-matched DeepSeek-R1-Distill-Qwen-1.5B transfer run.
# This intentionally contains no residual gate or adaptive-dose option.
set -euo pipefail

cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PY=/root/miniconda3/bin/python
BASE=../model/DeepSeek-R1-Distill-Qwen-1.5B
TRAIN=data/train_aug10.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
ROOT_OUT=outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42
CUR=$ROOT_OUT/curriculum/spcl
CKPT=$ROOT_OUT/checkpoints
EVAL=$ROOT_OUT/eval
mkdir -p "$ROOT_OUT" "$EVAL"

test -s "$BASE/config.json"
test "$(wc -l < "$TRAIN")" -eq 6000
test "$(wc -l < "$VAL")" -eq 600
test "$(wc -l < "$TEST")" -eq 600

echo "[1/6] Build protocol-matched SPCL curriculum (no residual gate)"
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --base "$BASE" --seed 42

echo "[2/6] Flat SFT: 3 epochs, effective batch 16 (micro-batch 16), seed 42"
$PY -u -m training.sft_lora --mode flat \
  --stages "$TRAIN" --val "$VAL" \
  --run-name deepseek_matched_flat --checkpoint-root "$CKPT" \
  --base "$BASE" --epochs 3 --lr 1e-4 --batch 16 --accum 1 \
  --max-len 2560 --loss-type chunked_nll --seed 42

echo "[3/6] SPCL SFT: 3 paced rounds, same optimizer budget"
$PY -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name deepseek_matched_spcl --checkpoint-root "$CKPT" \
  --base "$BASE" --epochs 1 --lr 1e-4 --batch 16 --accum 1 \
  --max-len 2560 --loss-type chunked_nll --seed 42

echo "[4/6] Greedy strict evaluation (flat and SPCL)"
$PY -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_flat/stage1" --base "$BASE" \
  --out "$EVAL/flat_test.json" --batch-size 16 --max-new 1400
$PY -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_spcl/stage3" --base "$BASE" \
  --out "$EVAL/spcl_test.json" --batch-size 16 --max-new 1400

echo "[5/6] Topology-SCD evaluation (flat+SCD and SPCL+SCD)"
$PY -u -m eval.eval_constrained --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_flat/stage1" --base "$BASE" \
  --out "$EVAL/flat_scd_test.json" --batch-size 16 --max-new 1400 --level topology
$PY -u -m eval.eval_constrained --data "$TEST" \
  --adapter "$CKPT/deepseek_matched_spcl/stage3" --base "$BASE" \
  --out "$EVAL/spcl_scd_test.json" --batch-size 16 --max-new 1400 --level topology

echo "[6/6] Run-level provenance"
$PY - <<'PY'
import json, hashlib
from pathlib import Path
root = Path("outputs/revision/deepseek_r1_qwen15b_matched_v2_b16/seed42")
test = Path("data/test.jsonl")
meta = [json.loads(x)["meta"] for x in test.open()]
coord = sum(m.get("scenario") != "independent" for m in meta)
manifest = {
    "base_model": "../model/DeepSeek-R1-Distill-Qwen-1.5B",
    "seed": 42, "train": "data/train_aug10.jsonl",
    "train_sha256": hashlib.sha256(Path("data/train_aug10.jsonl").read_bytes()).hexdigest(),
    "val": "data/val.jsonl", "test": "data/test.jsonl",
    "test_n": len(meta), "coordination_n": coord,
    "flat_epochs": 3, "spcl_rounds": 3, "effective_batch": 16,
    "micro_batch": 16, "gradient_accumulation": 1, "max_len": 2560,
    "lr": 1e-4, "loss_type": "chunked_nll",
    "window": [2,3,4], "boosts": "1,1|1,1,1.5|0.5,0.75,1.5,3",
    "residual_gate": False, "adaptive_dose": False,
    "scd_level": "topology", "decode": {"greedy": True, "max_new_tokens": 1400}
}
(root / "matched_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
echo "DONE: $ROOT_OUT"
