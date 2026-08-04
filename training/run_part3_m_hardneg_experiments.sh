#!/usr/bin/env bash
set -euo pipefail

# Part 3-M only optimizes WER/NER and assertion. Candidate generation/scoring is absent.
# Run with, for example:
#   CUDA_VISIBLE_DEVICES=1 bash training/run_part3_m_hardneg_experiments.sh

J_CHECKPOINT="${J_CHECKPOINT:-runs/part3_j_historical_full_30e_lr2e5/best}"
L_HEAD_CHECKPOINT="${L_HEAD_CHECKPOINT:-runs/part3_l_head_mass005_4e_lr5e5/best}"
INPUT_DIR="${INPUT_DIR:-input_turn2}"
L_DATA="${L_DATA:-datasets/ner_L_assertion_balanced_windows_20260803/track_a}"
M_DATA="${M_DATA:-datasets/ner_M_assertion_hardneg_windows_20260804/track_a}"
M_DEV="$M_DATA/m_assertion_hardneg_dev.jsonl"
L_DEV="$M_DATA/l_assertion_balanced_dev.jsonl"
M_PREFIX="part3_m_assertion_hardneg_train_reviewed_166_20260804:"
RUN_WINDOW_ABLATION="${RUN_WINDOW_ABLATION:-1}"

for required_path in \
  "$J_CHECKPOINT/hybrid_meta.json" \
  "$L_HEAD_CHECKPOINT/hybrid_meta.json" \
  "$L_DATA/train.jsonl" \
  "$M_DATA/train.jsonl" \
  "$M_DATA/test.jsonl" \
  "$M_DEV" \
  "$L_DEV" \
  "$INPUT_DIR"
do
  if [[ ! -e "$required_path" ]]; then
    echo "Thiếu đầu vào: $required_path" >&2
    exit 2
  fi
done

mkdir -p configs result runs

calibrate_checkpoint() {
  local checkpoint="$1"
  local tag="$2"
  local map="configs/${tag}_assertion_thresholds.json"

  python -m training.calibrate_assertion_thresholds \
    --checkpoint "$checkpoint" \
    --dev "$M_DEV" \
    --dev "$L_DEV" \
    --out-map "$map" \
    --out-report "result/${tag}_assertion_thresholds_report.json" \
    --global-threshold 0.60 \
    --low 0.40 \
    --high 0.85 \
    --step 0.05 \
    --min-positives 3 \
    --focus-assertion isHistorical
}

predict_checkpoint() {
  local checkpoint="$1"
  local out="$2"
  shift 2

  python -m training.predict_v2 \
    --model "$checkpoint" \
    --input "$INPUT_DIR" \
    --out "$out" \
    --assertion-policy part3 \
    --assertion-aggregation selected \
    "$@"
}

# 1) No-training threshold controls on the best L head-only checkpoint. These isolate
# assertion decision thresholds while keeping every NER span/type prediction unchanged.
for threshold in 0.60 0.65 0.70 0.75
do
  tag="${threshold/./}"
  predict_checkpoint \
    "$L_HEAD_CHECKPOINT" \
    "result/part3_m_lhead5_historical_thr${tag}" \
    --historical-threshold "$threshold"
done

calibrate_checkpoint "$L_HEAD_CHECKPOINT" "part3_m_lhead5_combined_dev"
predict_checkpoint \
  "$L_HEAD_CHECKPOINT" \
  result/part3_m_lhead5_calibrated \
  --assertion-threshold-map configs/part3_m_lhead5_combined_dev_assertion_thresholds.json

# 2) Train only the isHistorical row. Encoder, BIO, type/span heads and other assertion
# rows remain immutable and are hash-checked by train_assertion_head.
run_m_head() {
  local mass="$1"
  local tag="$2"
  local run_dir="runs/part3_m_head_${tag}"
  local map_tag="part3_m_head_${tag}_combined_dev"

  python -m training.train_assertion_head \
    --data "$M_DATA" \
    --init-checkpoint "$L_HEAD_CHECKPOINT" \
    --historical-dev "$M_DEV" \
    --safety-dev "$L_DEV" \
    --public-dev "$M_DATA/test.jsonl" \
    --selection-mode independent_first \
    --out "$run_dir" \
    --focus isHistorical \
    --targeted-id-prefix "$M_PREFIX" \
    --targeted-mass "$mass" \
    --epochs 4 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --lr 5e-5 \
    --threshold 0.60 \
    --public-dev-tolerance 0.01 \
    --safety-dev-tolerance 0.01 \
    --save-every-epoch \
    --fp16

  predict_checkpoint \
    "$run_dir/best" \
    "result/part3_m_head_${tag}_thr060"

  calibrate_checkpoint "$run_dir/best" "$map_tag"
  predict_checkpoint \
    "$run_dir/best" \
    "result/part3_m_head_${tag}_calibrated" \
    --assertion-threshold-map "configs/${map_tag}_assertion_thresholds.json"
}

run_m_head 0.02 mass002_4e_lr5e5
run_m_head 0.05 mass005_4e_lr5e5

# 3) Matched train+infer length ablation. Both arms start from J, use exactly the same
# L corpus/hyperparameters, and differ only in max_len. This is deliberately low-dose
# because every previous full-head K/L continuation, including the mass=0 control, regressed.
run_window_control() {
  local max_len="$1"
  local run_dir="runs/part3_m_window${max_len}_control_1e_lr1e6"
  local pred_dir="result/part3_m_window${max_len}_control_1e_lr1e6"

  python -m training.train_v2 \
    --data "$L_DATA" \
    --init-checkpoint "$J_CHECKPOINT" \
    --out "$run_dir" \
    --epochs 1 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-6 \
    --max-len "$max_len" \
    --save-every-epoch \
    --fp16

  predict_checkpoint \
    "$run_dir/best" \
    "$pred_dir" \
    --max-len "$max_len"
}

if [[ "$RUN_WINDOW_ABLATION" == "1" ]]; then
  run_window_control 256
  run_window_control 384
fi

echo "Hoàn tất Part 3-M. Baseline leaderboard bắt buộc: J-full 30e lr2e-5 (35.9792)."
echo "Không dùng J_candidates để chọn threshold, checkpoint hoặc thí nghiệm."
