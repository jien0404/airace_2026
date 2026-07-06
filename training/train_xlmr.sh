#!/usr/bin/env bash
# Train XLM-R base (multi-task NER + assertion) — CHẠY ĐỘC LẬP trên 1 GPU.
# Mặc định GPU 1. Chạy từ gốc repo:  bash training/train_xlmr.sh
# Tùy chỉnh:  GPU=1 EPOCHS=5 BS=32 bash training/train_xlmr.sh
set -euo pipefail

GPU="${GPU:-1}"
DATA="${DATA:-training/dataset}"
OUT="${OUT:-runs/xlmr}"
EPOCHS="${EPOCHS:-4}"
BS="${BS:-16}"
LR="${LR:-3e-5}"
MAXLEN="${MAXLEN:-256}"
MODEL="xlm-roberta-base"

[ -f "$DATA/labels.txt" ] || { echo "!! thiếu $DATA — giải nén dataset.zip trước"; exit 1; }
mkdir -p "$OUT"
echo "[xlmr] GPU=$GPU data=$DATA epochs=$EPOCHS bs=$BS -> $OUT (log $OUT/train.log)"

CUDA_VISIBLE_DEVICES="$GPU" python -m training.train_mtl \
  --data_dir "$DATA" --model "$MODEL" --out "$OUT" \
  --epochs "$EPOCHS" --bs "$BS" --lr "$LR" --max_len "$MAXLEN" --fp16 \
  2>&1 | tee "$OUT/train.log"
