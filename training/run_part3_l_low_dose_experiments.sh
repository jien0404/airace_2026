#!/usr/bin/env bash
set -euo pipefail

# K full retrains regressed from J, so this suite always warm-starts J and keeps the
# targeted sampling mass explicit.  It optimizes only WER/NER and assertions.
J_CHECKPOINT="${J_CHECKPOINT:-runs/part3_j_historical_full_30e_lr2e5/best}"
INPUT_DIR="${INPUT_DIR:-input_turn2}"
L_DATA="datasets/ner_L_assertion_balanced_windows_20260803/track_a"
K_WER_DATA="datasets/ner_K_wer_windows_20260803/track_a"
L_PREFIX="part3_l_assertion_balanced_train_reviewed_232_20260803:"
K_PREFIX="part3_k_wer_patch_reviewed_317_20260803:"

for required_path in \
  "$J_CHECKPOINT/hybrid_meta.json" \
  "$L_DATA/train.jsonl" \
  "$L_DATA/targeted_assertion_dev.jsonl" \
  "$K_WER_DATA/train.jsonl" \
  "$INPUT_DIR"
do
  if [[ ! -e "$required_path" ]]; then
    echo "Thiếu đầu vào: $required_path" >&2
    exit 2
  fi
done

python -m training.audit_token_budget \
  --data "$L_DATA" \
  --model "$J_CHECKPOINT" \
  --max-len 256 \
  --candidate-max-len 384 \
  --out result/part3_l_token_budget_audit.json

run_assertion_head() {
  local mass="$1"
  local tag="$2"
  local run_dir="runs/part3_l_head_${tag}"
  local pred_dir="result/part3_l_head_${tag}"

  python -m training.train_assertion_head \
    --data "$L_DATA" \
    --init-checkpoint "$J_CHECKPOINT" \
    --historical-dev "$L_DATA/targeted_assertion_dev.jsonl" \
    --public-dev "$L_DATA/test.jsonl" \
    --out "$run_dir" \
    --focus isHistorical \
    --targeted-id-prefix "$L_PREFIX" \
    --targeted-mass "$mass" \
    --epochs 4 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --lr 5e-5 \
    --threshold 0.60 \
    --independent-dev-tolerance 0.01 \
    --save-every-epoch \
    --fp16

  python -m training.predict_v2 \
    --model "$run_dir/best" \
    --input "$INPUT_DIR" \
    --out "$pred_dir" \
    --assertion-policy part3 \
    --assertion-aggregation selected
}

# Assertion-head-only: encoder, BIO, span/type heads and the two non-focused assertion
# rows are hash-checked as immutable by train_assertion_head.
run_assertion_head 0.02 mass002_4e_lr5e5
run_assertion_head 0.05 mass005_4e_lr5e5
run_assertion_head 0.10 mass010_4e_lr5e5

run_wer_warm_prefix() {
  local mass="$1"
  local tag="$2"
  local run_dir="runs/part3_l_wer_${tag}"
  local pred_dir="result/part3_l_wer_${tag}"

  python -m training.train_v2 \
    --data "$K_WER_DATA" \
    --init-checkpoint "$J_CHECKPOINT" \
    --out "$run_dir" \
    --epochs 2 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 5e-6 \
    --max-len 256 \
    --targeted-id-prefix "$K_PREFIX" \
    --targeted-mass "$mass" \
    --save-every-epoch \
    --fp16

  python -m training.predict_v2 \
    --model "$run_dir/best" \
    --input "$INPUT_DIR" \
    --out "$pred_dir" \
    --assertion-policy part3 \
    --assertion-aggregation selected
}

run_wer_warm_regex() {
  local regex="$1"
  local mass="$2"
  local tag="$3"
  local run_dir="runs/part3_l_wer_${tag}"
  local pred_dir="result/part3_l_wer_${tag}"

  python -m training.train_v2 \
    --data "$K_WER_DATA" \
    --init-checkpoint "$J_CHECKPOINT" \
    --out "$run_dir" \
    --epochs 2 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 5e-6 \
    --max-len 256 \
    --targeted-id-regex "$regex" \
    --targeted-mass "$mass" \
    --save-every-epoch \
    --fp16

  python -m training.predict_v2 \
    --model "$run_dir/best" \
    --input "$INPUT_DIR" \
    --out "$pred_dir" \
    --assertion-policy part3 \
    --assertion-aggregation selected
}

# Rehearsal control measures checkpoint drift without drawing any K-WER patch record.
run_wer_warm_prefix 0 control_mass000_2e_lr5e6
run_wer_warm_prefix 0.005 all_mass0005_2e_lr5e6
run_wer_warm_prefix 0.01 all_mass001_2e_lr5e6

# Family ablations test whether K's regression came from mixing incompatible error families.
run_wer_warm_regex '^part3_k_wer_patch_reviewed_317_20260803:C:targeted\.(type_confusion|boundary)\.' \
  0.005 type_boundary_mass0005_2e_lr5e6
run_wer_warm_regex '^part3_k_wer_patch_reviewed_317_20260803:C:targeted\.fp_context_contrast\.' \
  0.005 fp_context_mass0005_2e_lr5e6

echo "Hoàn tất Part 3 L low-dose suite. Baseline bắt buộc: J-full 30e lr2e-5."
