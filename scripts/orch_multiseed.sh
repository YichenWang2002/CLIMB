#!/bin/bash
set -uo pipefail
cd .
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python

echo "[orch] waiting for seed43 stage3 training (log /tmp/s43_r3.log)..."
while pgrep -f "run-name spcl_v2_s43_r3" > /dev/null; do sleep 60; done
echo "[orch] seed43 stage3 done. installing checkpoint -> spcl_v2_s43/stage3"
rm -rf outputs/checkpoints/spcl_v2_s43/stage3
cp -r outputs/checkpoints/spcl_v2_s43_r3/stage1 outputs/checkpoints/spcl_v2_s43/stage3

echo "[orch] launch seed44 curriculum (GPU) + seed43 test eval (GPU, concurrent)"
$PY -u -m curriculum.build_spcl --train data/train_aug10.jsonl \
  --val data/val.jsonl --out-dir outputs/spcl_v2_seed44 \
  --rounds 3 --buckets 4 --window 2,3,4 \
  --boosts "1,1|1,1,1.5|0.5,0.75,1.5,3" \
  --batch-size 8 --max-len 2560 --embed-device cuda --seed 44 > /tmp/s44_cur.log 2>&1 &
CURPID=$!
$PY -u -m eval.evaluate --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_v2_s43/stage3 \
  --out outputs/spcl_v2_s43_test.json --batch-size 48 --save-generations > /tmp/s43_eval.log 2>&1
echo "[orch] seed43 eval done (exit $?)"
wait $CURPID
echo "[orch] seed44 curriculum done (exit $?)"

echo "[orch] launch seed44 3-stage training"
$PY -u -m training.sft_lora --mode staged \
  --stages outputs/spcl_v2_seed44/round1.jsonl outputs/spcl_v2_seed44/round2.jsonl outputs/spcl_v2_seed44/round3.jsonl \
  --val data/val.jsonl --run-name spcl_v2_s44 --epochs 1 --lr 1e-4 --seed 44 > /tmp/s44_train.log 2>&1
echo "[orch] seed44 training done (exit $?)"

echo "[orch] launch seed44 test eval"
$PY -u -m eval.evaluate --data data/test.jsonl \
  --adapter outputs/checkpoints/spcl_v2_s44/stage3 \
  --out outputs/spcl_v2_s44_test.json --batch-size 48 --save-generations > /tmp/s44_eval.log 2>&1
echo "[orch] seed44 eval done (exit $?)"

echo "[orch] ALL DONE"
ls -la outputs/spcl_v2_s43_test.json outputs/spcl_v2_s44_test.json
