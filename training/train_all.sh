#!/usr/bin/env bash
# Tiện ích: chạy CẢ HAI script train nền cùng lúc (ViHealthBERT@GPU0, XLM-R@GPU1).
# Muốn theo dõi log riêng / kiểm soát từng tiến trình -> nên chạy 2 script tách:
#   bash training/train_vihealthbert.sh     # terminal 1 (GPU 0)
#   bash training/train_xlmr.sh             # terminal 2 (GPU 1)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

GPU=0 bash "$here/train_vihealthbert.sh" &
P1=$!
GPU=1 bash "$here/train_xlmr.sh" &
P2=$!
echo "[train_all] vihealthbert PID=$P1 (GPU0), xlmr PID=$P2 (GPU1)"
echo "[train_all] log: runs/vihealthbert/train.log , runs/xlmr/train.log"
wait $P1; wait $P2
echo "[train_all] XONG. best: runs/vihealthbert/best , runs/xlmr/best"
