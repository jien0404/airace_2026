#!/usr/bin/env bash
set -euo pipefail

# Exactly three new training runs: two fair N1 comparisons, then one N3 refit using the N1
# winner's recipe and optimizer-update budget. Candidates are never generated or optimized.
# Run: CUDA_VISIBLE_DEVICES=1 bash training/run_final_tokenbudget_large_experiments.sh

SHA="e803003cee5a43d740bf4024e65b82d7a6a7214389e3338c0d83352333959b9b"
MODEL="${MODEL:-FacebookAI/xlm-roberta-large}"
INPUT_DIR="${INPUT_DIR:-input_turn2}"
N1_CONTROL="datasets/ner_N1_v3_v67_tb256_ok35_windows_20260805/track_a"
N1_REGULARIZED="datasets/ner_N1_v3_v67_tb256_ok50_windows_20260805/track_a"
N3_CONTROL="datasets/ner_N3_v3_v67_tb256_ok35_windows_20260805/track_a"
N3_REGULARIZED="datasets/ner_N3_v3_v67_tb256_ok50_windows_20260805/track_a"
RUN_CONTROL="runs/final_n1_tb256_large_control_20260805"
RUN_REGULARIZED="runs/final_n1_tb256_large_regularized_20260805"

for required in \
  "$N1_CONTROL/train.jsonl" "$N1_CONTROL/raw_validation.jsonl" \
  "$N1_REGULARIZED/train.jsonl" "$N1_REGULARIZED/raw_validation.jsonl" \
  "$N3_CONTROL/train.jsonl" "$N3_REGULARIZED/train.jsonl" "$INPUT_DIR"
do
  if [[ ! -e "$required" ]]; then
    echo "Thiếu đầu vào: $required" >&2
    echo "Build trước bằng: python -m training.build_final_tokenbudget_datasets --model $MODEL" >&2
    exit 2
  fi
done

train_n1() {
  local data="$1"
  local out="$2"
  local encoder_lr="$3"
  local negative_ratio="$4"
  python -m training.train_v2 \
    --data "$data" \
    --model "$MODEL" \
    --out "$out" \
    --epochs 30 \
    --batch-size 8 \
    --eval-batch-size 16 \
    --grad-accumulation 2 \
    --lr 2e-5 \
    --encoder-lr "$encoder_lr" \
    --head-lr 2e-5 \
    --max-len 256 \
    --span-weight 0.5 \
    --negative-span-ratio "$negative_ratio" \
    --early-stopping-patience 5 \
    --prefix-subword-cap 64 \
    --eval-overlap-words 45 \
    --require-provenance \
    --require-dataset-variant n1 \
    --require-part3-sha256 "$SHA" \
    --fp16
}

predict_checkpoint() {
  local checkpoint="$1"
  local out="$2"
  python -m training.predict_v2 \
    --model "$checkpoint" \
    --input "$INPUT_DIR" \
    --out "$out" \
    --max-len 256 \
    --max-words 180 \
    --overlap-words 45 \
    --prefix-subword-cap 64 \
    --type-threshold 0.60 \
    --disagreement-threshold 0.85 \
    --assertion-threshold 0.60 \
    --assertion-policy part3 \
    --assertion-aggregation selected \
    --confidence-out "${out}_confidence"
}

train_n1 "$N1_CONTROL" "$RUN_CONTROL" 2e-5 2
predict_checkpoint "$RUN_CONTROL/best" result/final_n1_tb256_large_control_20260805

train_n1 "$N1_REGULARIZED" "$RUN_REGULARIZED" 1e-5 3
predict_checkpoint "$RUN_REGULARIZED/best" result/final_n1_tb256_large_regularized_20260805

CONTROL_REPORT="$RUN_CONTROL/training_report.json"
REGULARIZED_REPORT="$RUN_REGULARIZED/training_report.json"
WINNER="$(python -m training.select_final_n_recipe --control "$CONTROL_REPORT" --regularized "$REGULARIZED_REPORT" --field recipe)"
MAX_UPDATES="$(python -m training.select_final_n_recipe --control "$CONTROL_REPORT" --regularized "$REGULARIZED_REPORT" --field updates)"

if [[ "$WINNER" == "regularized" ]]; then
  N3_DATA="$N3_REGULARIZED"
  ENCODER_LR="1e-5"
  NEGATIVE_RATIO="3"
else
  N3_DATA="$N3_CONTROL"
  ENCODER_LR="2e-5"
  NEGATIVE_RATIO="2"
fi

RUN_N3="runs/final_n3_tb256_large_refit_${WINNER}_20260805"
echo "N1 winner=$WINNER; refit N3 for exactly $MAX_UPDATES optimizer updates."
python -m training.train_v2 \
  --data "$N3_DATA" \
  --model "$MODEL" \
  --out "$RUN_N3" \
  --epochs 30 \
  --max-updates "$MAX_UPDATES" \
  --batch-size 8 \
  --eval-batch-size 16 \
  --grad-accumulation 2 \
  --lr 2e-5 \
  --encoder-lr "$ENCODER_LR" \
  --head-lr 2e-5 \
  --max-len 256 \
  --span-weight 0.5 \
  --negative-span-ratio "$NEGATIVE_RATIO" \
  --prefix-subword-cap 64 \
  --eval-overlap-words 45 \
  --require-provenance \
  --require-dataset-variant n3 \
  --require-part3-sha256 "$SHA" \
  --save-last \
  --fp16
predict_checkpoint "$RUN_N3/last" "result/final_n3_tb256_large_refit_${WINNER}_20260805"

echo "Done. N2-large cũ vẫn là fallback; không tối ưu candidates."
