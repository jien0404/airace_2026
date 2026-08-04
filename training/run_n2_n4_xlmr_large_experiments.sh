#!/usr/bin/env bash
set -euo pipefail

# So sánh hai cách chia dữ liệu bằng cùng một backbone/recipe:
#   N2: Part 3 split 80/20, source bank N2-v3-v67 hiện tại.
#   N4: Part 2 groundtruth + Part 3 mỗi nguồn split 80/20, Part 1 và synthetic trong train.
#
# Chạy trên GPU 1:
#   CUDA_VISIBLE_DEVICES=1 bash training/run_n2_n4_xlmr_large_experiments.sh
#
# Có thể override đường dẫn/recipe bằng biến môi trường, ví dụ:
#   EPOCHS=20 BATCH_SIZE=4 GRAD_ACCUMULATION=4 CUDA_VISIBLE_DEVICES=1 \
#     bash training/run_n2_n4_xlmr_large_experiments.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

N2_ZIP="${N2_ZIP:-$REPO_ROOT/ner_N2_final_generalization_v3_v67_part3_split_windows_20260804.zip}"
N2_ROOT="${N2_ROOT:-$REPO_ROOT/datasets/ner_N2_final_generalization_v3_v67_part3_split_windows_20260804}"
N4_ZIP="${N4_ZIP:-$REPO_ROOT/ner_N4_gold23_mix_v3_v67_windows_20260804.zip}"
N4_ROOT="${N4_ROOT:-$REPO_ROOT/datasets/ner_N4_gold23_mix_v3_v67_windows_20260804}"
INPUT_DIR="${INPUT_DIR:-$REPO_ROOT/input_turn2}"

MODEL="${MODEL:-FacebookAI/xlm-roberta-large}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUMULATION="${GRAD_ACCUMULATION:-2}"
LR="${LR:-2e-5}"
MAX_LEN="${MAX_LEN:-256}"

N2_OUT="${N2_OUT:-runs/part3_n2_v3_v67_xlmr_large_30e_lr2e5}"
N2_PRED="${N2_PRED:-result/part3_n2_v3_v67_xlmr_large_30e_lr2e5}"
N4_OUT="${N4_OUT:-runs/part3_n4_gold23_mix_xlmr_large_30e_lr2e5}"
N4_PRED="${N4_PRED:-result/part3_n4_gold23_mix_xlmr_large_30e_lr2e5}"

ensure_dataset() {
  local archive="$1"
  local destination="$2"
  if [[ ! -f "$archive" ]]; then
    echo "Thiếu dataset ZIP: $archive" >&2
    exit 2
  fi
  if [[ ! -f "$destination/track_a/train.jsonl" ]]; then
    mkdir -p "$destination"
    unzip -q -o "$archive" -d "$destination"
  fi
  for required in \
    "$destination/track_a/train.jsonl" \
    "$destination/track_a/validation.jsonl" \
    "$destination/track_a/test.jsonl"
  do
    if [[ ! -f "$required" ]]; then
      echo "Dataset giải nén thiếu file: $required" >&2
      exit 2
    fi
  done
}

for required in "$INPUT_DIR" "$N2_ZIP" "$N4_ZIP"
do
  if [[ ! -e "$required" ]]; then
    echo "Thiếu đầu vào: $required" >&2
    exit 2
  fi
done

ensure_dataset "$N2_ZIP" "$N2_ROOT"
ensure_dataset "$N4_ZIP" "$N4_ROOT"

train_one() {
  local data="$1"
  local out="$2"
  echo "===== TRAIN: $data -> $out ====="
  python -m training.train_v2 \
    --data "$data" \
    --model "$MODEL" \
    --out "$out" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --grad-accumulation "$GRAD_ACCUMULATION" \
    --lr "$LR" \
    --max-len "$MAX_LEN" \
    --fp16
}

predict_one() {
  local model="$1"
  local out="$2"
  echo "===== PREDICT: $model -> $out ====="
  python -m training.predict_v2 \
    --model "$model" \
    --input "$INPUT_DIR" \
    --out "$out" \
    --max-len "$MAX_LEN" \
    --max-words 180 \
    --overlap-words 45 \
    --type-threshold 0.60 \
    --disagreement-threshold 0.85 \
    --assertion-threshold 0.60 \
    --assertion-policy part3 \
    --assertion-aggregation selected
}

train_one "$N2_ROOT/track_a" "$N2_OUT"
predict_one "$N2_OUT/best" "$N2_PRED"

train_one "$N4_ROOT/track_a" "$N4_OUT"
predict_one "$N4_OUT/best" "$N4_PRED"

echo "Hoàn tất N2 và N4 với $MODEL."
echo "N2 prediction: $N2_PRED"
echo "N4 prediction: $N4_PRED"
