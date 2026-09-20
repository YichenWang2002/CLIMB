#!/bin/bash
# Budget-matched supervision-source ablation (fixes the yield confound):
# the LLM teacher only passed 4160/6000 tasks, so its arm trained on less
# data. This script restricts the PLANNER arm to exactly the same 4160
# inputs the teacher solved -- same tasks, same NL, same optimizer budget;
# only the tree-writer differs. Then runs the 1.5B backbone leg (both arms,
# also matched).
#
#   arm A-matched : planner trees,   teacher-solved 4160 tasks
#   arm B         : LLM-teacher trees, same 4160 tasks (already trained on
#                   llama; results exist -> only the paired test is redone)
set -euo pipefail
cd .
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3

TRAIN_AUG=data/train_aug10.jsonl
TRAIN_LLMT=data/train_llmteacher.jsonl
TRAIN_MATCH=data/train_planner_matched.jsonl
VAL=data/val.jsonl
TEST=data/test.jsonl
NAME=planner_matched_spcl
CUR=outputs/spcl_planner_matched_seed42

echo "=== [1/4] build matched planner file (teacher-solved inputs only) ==="
$PY - "$TRAIN_AUG" "$TRAIN_LLMT" "$TRAIN_MATCH" <<'EOF'
import json, sys
aug, llmt, out = sys.argv[1:4]
solved = {json.loads(l)["input"] for l in open(llmt)}
kept, total = [], 0
for line in open(aug):
    total += 1
    r = json.loads(line)
    if r["input"] in solved:
        kept.append(line)
with open(out, "w") as fh:
    fh.writelines(kept)
print(f"kept {len(kept)}/{total} rows -> {out}")
EOF

echo "=== [2/4] SPCL + training on llama-1B, planner-matched arm ==="
$PY -u -m curriculum.build_spcl \
  --train "$TRAIN_MATCH" --val "$VAL" --out-dir "$CUR" \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cpu --seed 42
$PY -u -m training.sft_lora --mode staged \
  --stages "$CUR/round1.jsonl" "$CUR/round2.jsonl" "$CUR/round3.jsonl" \
  --val "$VAL" --run-name "$NAME" --epochs 1 --lr 1e-4 --seed 42

echo "=== [3/4] test eval: unconstrained and +SCD(topology) ==="
$PY -u -m eval.evaluate --data "$TEST" \
  --adapter "outputs/checkpoints/$NAME/stage3" \
  --out "outputs/${NAME}_test.json" --batch-size 64 --save-generations
$PY -u -m eval.eval_constrained --data "$TEST" \
  --adapter "outputs/checkpoints/$NAME/stage3" \
  --out "outputs/${NAME}_scd_test.json" --batch-size 48 --level topology

echo "=== [4/4] paired McNemar: matched planner vs LLM teacher (llama leg) ==="
$PY -m eval.paired_test outputs/${NAME}_scd_test.json outputs/llmteacher_spcl_scd_test.json
$PY -m eval.paired_test outputs/${NAME}_test.json outputs/llmteacher_spcl_test.json

echo "=== MATCHED LLAMA LEG DONE; launching 1.5B leg ==="
bash scripts/run_llm_teacher_ablation_ds15.sh
