#!/usr/bin/env bash
set -euo pipefail

CONTROL="runs/final_n1_tb256_large_control_20260805"
REGULARIZED="runs/final_n1_tb256_large_regularized_20260805"
for report in "$CONTROL/training_report.json" "$REGULARIZED/training_report.json"
do
  if [[ ! -f "$report" ]]; then
    echo "Thiếu $report" >&2
    exit 2
  fi
done

WINNER="$(python -m training.select_final_n_recipe \
  --control "$CONTROL/training_report.json" \
  --regularized "$REGULARIZED/training_report.json" \
  --field recipe)"
N3="runs/final_n3_tb256_large_refit_${WINNER}_20260805"
OUT="result/final_tokenbudget_large_runs_and_results_20260805.zip"

zip -qr "$OUT" \
  result/final_tokenbudget_build.json \
  "$CONTROL/training_report.json" \
  "$CONTROL/test_metrics.json" \
  "$CONTROL/test_metrics_best.json" \
  "$CONTROL/best/hybrid_meta.json" \
  "$REGULARIZED/training_report.json" \
  "$REGULARIZED/test_metrics.json" \
  "$REGULARIZED/test_metrics_best.json" \
  "$REGULARIZED/best/hybrid_meta.json" \
  "$N3/training_report.json" \
  "$N3/test_metrics.json" \
  "$N3/test_metrics_best.json" \
  "$N3/test_metrics_last.json" \
  "$N3/best/hybrid_meta.json" \
  "$N3/last/hybrid_meta.json" \
  result/final_n1_tb256_large_control_20260805 \
  result/final_n1_tb256_large_control_20260805_confidence \
  result/final_n1_tb256_large_regularized_20260805 \
  result/final_n1_tb256_large_regularized_20260805_confidence \
  "result/final_n3_tb256_large_refit_${WINNER}_20260805" \
  "result/final_n3_tb256_large_refit_${WINNER}_20260805_confidence"

echo "$OUT"

