#!/usr/bin/env bash
set -euo pipefail

RUN="runs/part3_n2_tb256_large_balanced_20260805"
OUT="result/n2_external_generalization_results_20260805.zip"
for required in \
  "$RUN/training_report.json" \
  "$RUN/test_metrics.json" \
  "$RUN/test_metrics_best.json" \
  "$RUN/best/hybrid_meta.json"
do
  if [[ ! -f "$required" ]]; then
    echo "Thiếu $required" >&2
    exit 2
  fi
done

zip -qr "$OUT" \
  result/final_tokenbudget_build_n2_balanced.json \
  "$RUN/training_report.json" \
  "$RUN/test_metrics.json" \
  "$RUN/test_metrics_best.json" \
  "$RUN/best/hybrid_meta.json" \
  result/part3_n2_tb256_large_balanced_20260805 \
  result/external108_n2_tb256_large_balanced_default_20260805 \
  result/external108_n2_tb256_large_balanced_default_20260805_confidence \
  result/external108_n2_tb256_large_balanced_h055_20260805 \
  result/external108_n2_tb256_large_balanced_recall_h055_20260805

echo "$OUT"

