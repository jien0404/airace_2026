#!/usr/bin/env bash
set -euo pipefail

# One new N2-large training run. External 108 is inference-only and never used for training,
# checkpoint selection, threshold calibration, or automatic labels.
# Run: CUDA_VISIBLE_DEVICES=1 bash training/run_n2_external_generalization_experiment.sh

SHA="e803003cee5a43d740bf4024e65b82d7a6a7214389e3338c0d83352333959b9b"
MODEL="${MODEL:-FacebookAI/xlm-roberta-large}"
DATA="datasets/ner_N2_v3_v67_tb256_ok45_windows_20260805/track_a"
PART3_INPUT="${PART3_INPUT:-input_turn2}"
EXTERNAL_NOTES="${EXTERNAL_NOTES:-annotation/data/external_benhvien108_qa_v1/notes}"
RUN="runs/part3_n2_tb256_large_balanced_20260805"

for required in "$DATA/train.jsonl" "$DATA/raw_validation.jsonl" "$PART3_INPUT" "$EXTERNAL_NOTES"
do
  if [[ ! -e "$required" ]]; then
    echo "Thiếu đầu vào: $required" >&2
    exit 2
  fi
done

python -m training.train_v2 \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$RUN" \
  --epochs 30 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --grad-accumulation 2 \
  --lr 2e-5 \
  --encoder-lr 1e-5 \
  --head-lr 2e-5 \
  --max-len 256 \
  --span-weight 0.5 \
  --assertion-weight 1.0 \
  --negative-span-ratio 2 \
  --early-stopping-patience 5 \
  --prefix-subword-cap 64 \
  --eval-overlap-words 45 \
  --require-provenance \
  --require-dataset-variant n2 \
  --require-part3-sha256 "$SHA" \
  --fp16

predict() {
  local input="$1"
  local out="$2"
  shift 2
  python -m training.predict_v2 \
    --model "$RUN/best" \
    --input "$input" \
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
    "$@"
}

predict "$PART3_INPUT" result/part3_n2_tb256_large_balanced_20260805

DEFAULT="result/external108_n2_tb256_large_balanced_default_20260805"
H055="result/external108_n2_tb256_large_balanced_h055_20260805"
RECALL="result/external108_n2_tb256_large_balanced_recall_h055_20260805"

predict "$EXTERNAL_NOTES" "$DEFAULT" \
  --confidence-out "${DEFAULT}_confidence"
predict "$EXTERNAL_NOTES" "$H055" \
  --historical-threshold 0.55
predict "$EXTERNAL_NOTES" "$RECALL" \
  --historical-threshold 0.55 \
  --ner-threshold-map configs/n2_external_recall_probe_v1.json

python -m annotation.external_challenge.stage_prediction_view \
  --notes "$EXTERNAL_NOTES" --predictions "$DEFAULT" \
  --out annotation/data/external108_n2_tb_default_20260805 \
  --name n2_tb_default --overwrite
python -m annotation.external_challenge.stage_prediction_view \
  --notes "$EXTERNAL_NOTES" --predictions "$H055" \
  --out annotation/data/external108_n2_tb_h055_20260805 \
  --name n2_tb_h055 --overwrite
python -m annotation.external_challenge.stage_prediction_view \
  --notes "$EXTERNAL_NOTES" --predictions "$RECALL" \
  --out annotation/data/external108_n2_tb_recall_h055_20260805 \
  --name n2_tb_recall_h055 --overwrite

echo "Hoàn tất N2 balanced và ba probe external 108. Chưa train N3."

