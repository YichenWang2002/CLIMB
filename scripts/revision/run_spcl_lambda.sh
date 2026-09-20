#!/usr/bin/env bash
# SPCL-lambda: dose-gated SPCL for cross-backbone replication.
#
# Identical protocol to run_backbone_replication.sh (same batch/accum/
# max-len/loss/decode/eval), with one principled change: every nominal
# hard-bucket boost m>1 at a round boundary is gated by the fraction of
# learning distance that remains in that bucket,
#
#     m_realized = 1 + (m_nominal - 1) * clip01(NLL_b(ckpt)/NLL_b(base)),
#
# measured on TRAINING data only by curriculum.residual_probe. The rule has
# no hyperparameters and is applied identically to every backbone and seed.
#
# Round-1 boosts are (1,1) so round 1 is byte-identical to the frozen v2
# curriculum; gated rounds are rebuilt from the same scores sidecar, so
# buckets/utility/sampling-rng stay identical to v2 except for the gate.
#
# Dry run:  DRY_RUN=1 bash pipeline/scripts/revision/run_spcl_lambda.sh
# Real run: SEEDS=42 DRY_RUN=0 bash pipeline/scripts/revision/run_spcl_lambda.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/root/miniconda3/bin/python}"

# --- defaults mirror the qwen25_1p5b_chunked_b16_rerun protocol ------------
BASE="${BASE:-../model/qwen25-15b}"
BACKBONE_NAME="${BACKBONE_NAME:-qwen25_1p5b_lambda}"
SIDECAR="${SIDECAR:-outputs/revision/qwen25_1p5b_chunked_b16_rerun/seed42/curriculum/spcl/spcl_scores.jsonl}"
SEEDS="${SEEDS:-42}"
ROOT="${ROOT:-outputs/revision/${BACKBONE_NAME}}"
TRAIN="${TRAIN:-data/train_aug10.jsonl}"
VAL="${VAL:-data/val.jsonl}"
TEST="${TEST:-data/test.jsonl}"
DRY_RUN="${DRY_RUN:-1}"
RUN_SCD="${RUN_SCD:-1}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-1}"
MAX_LEN="${MAX_LEN:-2048}"
EVAL_BATCH="${EVAL_BATCH:-4}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
BOOSTS="${BOOSTS:-1,1|1,1,1.5|0.5,0.75,1.5,3}"
WINDOW="${WINDOW:-2,3,4}"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then
    "$@"
  fi
}

[[ -f "$SIDECAR" ]] || { echo "missing scores sidecar: $SIDECAR" >&2; exit 2; }
[[ -d "$BASE" ]] || { echo "BASE is not a model dir: $BASE" >&2; exit 2; }

# --- shared base-model exec probe (round-1 boundary), built once ------------
BASE_PROBE="$ROOT/probes/base_exec.json"
if [[ ! -s "$BASE_PROBE" || "$DRY_RUN" == 1 ]]; then
  run "$PY" -u -m curriculum.residual_probe --mode exec \
    --train "$TRAIN" --scores "$SIDECAR" --base "$BASE" \
    --label "${BACKBONE_NAME}_base" --device cuda --batch-size 64 \
    --per-bucket 100 --sample-seed 2026 --out "$BASE_PROBE"
fi

for SEED in $SEEDS; do
  SEED_ROOT="$ROOT/seed${SEED}"
  CUR="$SEED_ROOT/curriculum/spcl_lambda"
  CKPT="$SEED_ROOT/checkpoints"
  EVAL="$SEED_ROOT/eval"
  PROBES="$SEED_ROOT/probes"
  MANIFEST="$SEED_ROOT/manifest.json"
  if [[ -e "$MANIFEST" && "$DRY_RUN" != 1 ]]; then
    echo "refusing to reuse existing run manifest: $MANIFEST" >&2; exit 3
  fi
  mkdir -p "$SEED_ROOT" "$PROBES"

  run "$PY" -u -m experiments.revision.protocol manifest \
    --phase "backbone_lambda/${BACKBONE_NAME}" --status planned \
    --base "$BASE" --seed "$SEED" --train "$TRAIN" --val "$VAL" --test "$TEST" \
    --optimizer-steps "{\"flat_epochs\":0,\"spcl_rounds\":3,\"effective_batch\":$((BATCH * ACCUM)),\"micro_batch\":$BATCH,\"gradient_accumulation\":$ACCUM,\"loss_type\":\"$LOSS_TYPE\",\"max_len\":$MAX_LEN,\"staged_sampler\":\"random\",\"dose_gate\":\"1+(m-1)*R_b; R_b=fraction of bucket still failing greedy+symbolic execution on stratified training subsample (100/bucket, seed 2026); training data only\"}" \
    --decode-config "{\"greedy\":true,\"max_new_tokens\":1400,\"eval_batch\":$EVAL_BATCH}" --out "$MANIFEST" \
    --overwrite

  PREV=""
  for ROUND in 1 2 3; do
    IDX=$((ROUND - 1))
    GATE_ARGS=()
    if [[ "$ROUND" -gt 1 ]]; then
      GATE_ARGS=(--gate-residual "$PROBES/exec_s$((ROUND - 1)).json")
    fi
    echo "=== [${BACKBONE_NAME} s${SEED}] build round ${ROUND} ==="
    run "$PY" -u -m curriculum.build_spcl --train "$TRAIN" --val "$VAL" \
      --out-dir "$CUR" --rounds 3 --buckets 4 --window "$WINDOW" \
      --boosts "$BOOSTS" --scores-cache "$SIDECAR" --only-round "$IDX" \
      --base "$BASE" --seed "$SEED" "${GATE_ARGS[@]}"

    echo "=== [${BACKBONE_NAME} s${SEED}] train round ${ROUND} ==="
    INIT_ARG=()
    if [[ -n "$PREV" ]]; then
      INIT_ARG=(--init-adapter "$PREV")
    fi
    run "$PY" -u -m training.sft_lora --mode flat \
      --stages "$CUR/round${ROUND}.jsonl" --val "$VAL" \
      --run-name "spcl_r${ROUND}" --checkpoint-root "$CKPT" --base "$BASE" \
      --epochs 1 --lr 1e-4 --batch "$BATCH" --accum "$ACCUM" \
      --max-len "$MAX_LEN" --loss-type "$LOSS_TYPE" --seed "$((SEED + ROUND - 1))" \
      "${INIT_ARG[@]}"
    PREV="$CKPT/spcl_r${ROUND}/stage1"

    if [[ "$ROUND" -lt 3 ]]; then
      echo "=== [${BACKBONE_NAME} s${SEED}] exec probe boundary theta_${ROUND} ==="
      run "$PY" -u -m curriculum.residual_probe --mode exec \
        --train "$TRAIN" --scores "$SIDECAR" --base "$BASE" \
        --adapter "$PREV" --label "${BACKBONE_NAME}_s${SEED}_theta${ROUND}" \
        --device cuda --batch-size 64 --per-bucket 100 --sample-seed 2026 \
        --out "$PROBES/exec_s${ROUND}.json"
    fi
  done

  echo "=== [${BACKBONE_NAME} s${SEED}] exposure accounting ==="
  run "$PY" - "$SIDECAR" "$CUR" << 'PYEOF'
import hashlib, json, sys
import numpy as np
from pathlib import Path
sidecar_path, cur = sys.argv[1], Path(sys.argv[2])
side = [json.loads(l) for l in Path(sidecar_path).read_text().splitlines() if l.strip()]
bucket = np.array([int(r["bucket"]) for r in side])
idx_of = {r["spcl_record_id"]: i for i, r in enumerate(side)}
draws = np.zeros(len(side), dtype=np.int64)
for rnd in (1, 2, 3):
    for line in (cur / f"round{rnd}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = hashlib.sha256(json.dumps(
            {k: rec.get(k) for k in ("instruction", "input", "output")},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        draws[idx_of[rid]] += 1
for b in range(4):
    m = bucket == b
    print(f"bucket {b}: lambda={draws[m].mean():.3f} "
          f"never={(draws[m] == 0).mean() * 100:.1f}%", flush=True)
PYEOF

  echo "=== [${BACKBONE_NAME} s${SEED}] held-out test (strict) ==="
  run "$PY" -u -m experiments.bt_ducl.strict_eval --data "$TEST" \
    --adapter "$CKPT/spcl_r3/stage1" --base "$BASE" \
    --out "$EVAL/spcl_lambda_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
  if [[ "$RUN_SCD" == 1 ]]; then
    echo "=== [${BACKBONE_NAME} s${SEED}] held-out test (SCD topology) ==="
    run "$PY" -u -m eval.eval_constrained --data "$TEST" \
      --adapter "$CKPT/spcl_r3/stage1" --base "$BASE" --level topology \
      --out "$EVAL/spcl_lambda_scd_test.json" --batch-size "$EVAL_BATCH" --max-new 1400
  fi
  echo "=== ${BACKBONE_NAME} seed ${SEED} DONE ==="
done
echo "SPCL-lambda protocol emitted. DRY_RUN=$DRY_RUN ROOT=$ROOT"
