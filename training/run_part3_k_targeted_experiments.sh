#!/usr/bin/env bash
set -euo pipefail

python -m training.train_v2 --data datasets/ner_K_wer_windows_20260803/track_a \
  --model xlm-roberta-base --out runs/part3_k_wer_30e_lr2e5 \
  --epochs 30 --batch-size 16 --eval-batch-size 32 --lr 2e-5 --max-len 256 --fp16
python -m training.predict_v2 --model runs/part3_k_wer_30e_lr2e5/best \
  --input input_turn2 --out result/part3_k_wer_30e_lr2e5 \
  --assertion-policy part3 --assertion-aggregation selected

python -m training.train_v2 --data datasets/ner_K_assertion_windows_20260803/track_a \
  --model xlm-roberta-base --out runs/part3_k_assertion_30e_lr2e5 \
  --epochs 30 --batch-size 16 --eval-batch-size 32 --lr 2e-5 --max-len 256 --fp16
python -m training.predict_v2 --model runs/part3_k_assertion_30e_lr2e5/best \
  --input input_turn2 --out result/part3_k_assertion_30e_lr2e5 \
  --assertion-policy part3 --assertion-aggregation selected

python -m training.train_v2 --data datasets/ner_K_joint_windows_20260803/track_a \
  --model xlm-roberta-base --out runs/part3_k_joint_30e_lr2e5 \
  --epochs 30 --batch-size 16 --eval-batch-size 32 --lr 2e-5 --max-len 256 --fp16
python -m training.predict_v2 --model runs/part3_k_joint_30e_lr2e5/best \
  --input input_turn2 --out result/part3_k_joint_30e_lr2e5 \
  --assertion-policy part3 --assertion-aggregation selected

unzip -q -o ner_K_wer_windows_20260803.zip -d datasets/ner_K_wer_windows_20260803/
unzip -q -o ner_K_assertion_windows_20260803.zip -d datasets/ner_K_assertion_windows_20260803/
unzip -q -o ner_K_joint_windows_20260803.zip -d datasets/ner_K_joint_windows_20260803/