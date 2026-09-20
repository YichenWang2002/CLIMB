#!/usr/bin/env bash
# Frozen cross-backbone replication for the locally available Gemma 2 2B base.
# The only backbone-specific change is the tokenizer chat-template compatibility
# layer shared by training and evaluation; all data, curriculum, LoRA, optimizer,
# and decoding settings are inherited from run_backbone_replication.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PIPELINE"

export BASE="${BASE:-../model/gemma2-2b}"
export BACKBONE_NAME="${BACKBONE_NAME:-gemma2_2b}"
export ROOT="${ROOT:-outputs/revision/${BACKBONE_NAME}}"
export SEEDS="${SEEDS:-42 43 44}"
export MAX_LEN="${MAX_LEN:-2560}"
export BATCH="${BATCH:-4}"
export ACCUM="${ACCUM:-4}"
export CURRICULUM_BATCH="${CURRICULUM_BATCH:-2}"
export LOSS_TYPE="${LOSS_TYPE:-nll}"
export EVAL_BATCH="${EVAL_BATCH:-4}"
export DRY_RUN="${DRY_RUN:-1}"
export RUN_SCD="${RUN_SCD:-1}"
export OVERWRITE="${OVERWRITE:-0}"

exec bash "$SCRIPT_DIR/run_backbone_replication.sh"
