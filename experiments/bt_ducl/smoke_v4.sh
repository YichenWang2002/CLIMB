#!/usr/bin/env bash
# ~10-minute catastrophe preflight: train a tiny flat model on a 2400-row
# subset for 1 epoch, then strict-eval 192 val rows. Run this AFTER ANY
# change to the loss, tokenizer handling, collator or eval path, BEFORE
# launching run_v4_single_seed.sh. It exists because the semantic-mask loss
# shipped in v4 produced 0/600 strict success and was only discovered after
# two full 3-epoch trainings.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED="${SEED:-42}"
ROOT="${ROOT:-outputs/bt_ducl_smoke_seed${SEED}}"
TRAIN="outputs/dataset/train_aug10.jsonl"
VAL="outputs/dataset/val.jsonl"
mkdir -p "$ROOT"

# The train file is grouped by tier, so a plain prefix slice would be T2-only.
# Build a seeded tier-stratified 2400-row subset (30/50/20, as in train).
python3 - "$TRAIN" "$ROOT/train_smoke.jsonl" <<'PY'
import json, random, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
by_tier = {}
for i, row in enumerate(rows):
    by_tier.setdefault(row["meta"]["tier"], []).append(i)
rng = random.Random(42)
quota = {"T1": 720, "T2": 1200, "T3": 480}
pick = []
for tier, count in quota.items():
    indices = by_tier[tier][:]
    rng.shuffle(indices)
    pick += indices[:count]
rng.shuffle(pick)
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    for i in pick:
        handle.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
print(f"smoke subset written: {len(pick)} rows, quota={quota}")
PY

python3 -u -m experiments.bt_ducl.train_v4 --mode flat \
  --train "$ROOT/train_smoke.jsonl" --val "$VAL" --run-dir "$ROOT/flat" \
  --epochs 1 --batch 4 --accum 4 --lr 1e-4 --max-len 2560 --seed "$SEED" \
  --loss-scope completion

python3 -u -m experiments.bt_ducl.strict_eval --data "$VAL" \
  --adapter "$ROOT/flat/final" --out "$ROOT/smoke_val.json" \
  --batch-size 64 --max-new 1400 --limit 192

python3 - "$ROOT/smoke_val.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
n = value["n"]
rate = float(value["strict_success_rate"])
parse_fail = sum(v for k, v in value["fail_reasons"].items() if k.startswith("parse:"))
parse_ok = 1.0 - parse_fail / n
print(f"smoke: strict={rate:.3f} ({int(round(rate * n))}/{n})  parse_ok={parse_ok:.3f}")
print("smoke fail reasons:", json.dumps(value["fail_reasons"], ensure_ascii=False))
# The smoke model is weak (20% data, 1 epoch), so the gate checks structure,
# not task mastery: XML must mostly parse. The broken v4 loss gave parse_ok
# ~0.30 at this scale; a healthy loss gives > 0.9.
if parse_ok < 0.75 or rate < 0.01:
    print("SMOKE FAILED: do NOT launch full training; the shared setup is broken.")
    sys.exit(1)
print("smoke passed: safe to launch run_v4_single_seed.sh")
PY
