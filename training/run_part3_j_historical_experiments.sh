#!/usr/bin/env bash
set -euo pipefail

# Baseline so sánh: part3_i_wer_only_30e_lr2e5.
# Script chỉ chạy hai thí nghiệm J; không chạy lại H hoặc I.

unzip -q -o ner_J_part3_historical_scope_v2_windows_20260803.zip \
  -d datasets/ner_J_part3_historical_scope_v2_windows_20260803/

python -m training.train_v2 \
  --data datasets/ner_J_part3_historical_scope_v2_windows_20260803/track_a \
  --model xlm-roberta-base \
  --out runs/part3_j_historical_full_30e_lr2e5 \
  --epochs 30 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --lr 2e-5 \
  --max-len 256 \
  --fp16

python -m training.predict_v2 \
  --model runs/part3_j_historical_full_30e_lr2e5/best \
  --input input_turn2 \
  --out result/part3_j_historical_full_30e_lr2e5

python -m training.train_assertion_head \
  --data datasets/ner_J_part3_historical_scope_v2_windows_20260803/track_a \
  --init-checkpoint runs/part3_i_wer_only_30e_lr2e5/best \
  --historical-dev datasets/ner_J_part3_historical_scope_v2_windows_20260803/track_a/historical_dev.jsonl \
  --out runs/part3_j_historical_head_only_8e_lr5e5 \
  --focus isHistorical \
  --targeted-id-prefix part3_historical_scope_v2_train_reviewed_160_20260803: \
  --targeted-mass 0.10 \
  --epochs 8 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr 5e-5 \
  --threshold 0.60 \
  --fp16

python -m training.predict_v2 \
  --model runs/part3_j_historical_head_only_8e_lr5e5/best \
  --input input_turn2 \
  --out result/part3_j_historical_head_only_8e_lr5e5
