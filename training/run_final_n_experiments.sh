#!/usr/bin/env bash
set -euo pipefail

# Final N: chỉ tối ưu WER/NER + assertion. Không sinh, calibrate hay chấm candidates.
# Chạy trên GPU 1: CUDA_VISIBLE_DEVICES=1 bash training/run_final_n_experiments.sh

N1_DATA="${N1_DATA:-datasets/ner_N1_final_generalization_v3_v67_part3_heldout_windows_20260804/track_a}"
N2_DATA="${N2_DATA:-datasets/ner_N2_final_generalization_v3_v67_part3_split_windows_20260804/track_a}"
N3_DATA="${N3_DATA:-datasets/ner_N3_final_generalization_v3_v67_private_refit_windows_20260804/track_a}"
INPUT_DIR="${INPUT_DIR:-input_turn2}"

for required in "$N1_DATA/train.jsonl" "$N2_DATA/train.jsonl" "$N3_DATA/train.jsonl" "$INPUT_DIR"
do
  if [[ ! -e "$required" ]]; then
    echo "Thiếu đầu vào: $required" >&2
    exit 2
  fi
done

train_full() {
  local data="$1"
  local out="$2"
  python -m training.train_v2 \
    --data "$data" \
    --model xlm-roberta-base \
    --out "$out" \
    --epochs 30 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 2e-5 \
    --max-len 256 \
    --fp16
}

predict_best() {
  local model="$1"
  local out="$2"
  python -m training.predict_v2 \
    --model "$model" \
    --input "$INPUT_DIR" \
    --out "$out" \
    --max-len 256 \
    --max-words 180 \
    --overlap-words 45 \
    --type-threshold 0.60 \
    --disagreement-threshold 0.85 \
    --assertion-threshold 0.60 \
    --assertion-policy part3 \
    --assertion-aggregation selected
}

train_full "$N1_DATA" runs/part3_n1_v3_v67_heldout_30e_lr2e5
predict_best runs/part3_n1_v3_v67_heldout_30e_lr2e5/best result/part3_n1_v3_v67_heldout_30e_lr2e5

train_full "$N2_DATA" runs/part3_n2_v3_v67_split_30e_lr2e5
predict_best runs/part3_n2_v3_v67_split_30e_lr2e5/best result/part3_n2_v3_v67_split_30e_lr2e5

FINAL_EPOCH="$(python -m training.final_epoch --report runs/part3_n2_v3_v67_split_30e_lr2e5/training_report.json)"
echo "Refit N3 đúng ${FINAL_EPOCH} epoch theo N2."
python -m training.train_v2 \
  --data "$N3_DATA" \
  --model xlm-roberta-base \
  --out runs/part3_n3_v3_v67_private_refit \
  --epochs "$FINAL_EPOCH" \
  --batch-size 16 \
  --eval-batch-size 32 \
  --lr 2e-5 \
  --max-len 256 \
  --save-last \
  --fp16
predict_best runs/part3_n3_v3_v67_private_refit/last result/part3_n3_v3_v67_private_refit

echo "Hoàn tất N1/N2/N3. Baseline public bắt buộc: J-full 35.9792; WER 56.5873; J_assertion 49.5329."
