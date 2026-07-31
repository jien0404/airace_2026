from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

from .schema_v2 import ASSERTION_TYPES

from .model_v2 import HybridNER
from .windowing_v2 import sliding_windows


def encoder_capacity(model) -> int:
    """Số token TỐI ĐA encoder nhận được.

    PhoBERT là RoBERTa: `position_ids = cumsum(mask) * mask + padding_idx`, nên chỉ số vị trí lớn
    nhất là `padding_idx + seq_len`. Với `max_position_embeddings=258` và `pad_token_id=1`, chuỗi
    dài quá 256 làm tra bảng position embedding ra ngoài biên — CUDA báo `device-side assert
    triggered` ở tận lớp attention, không hề nhắc tới độ dài. Mặc định `--max-len 512` chạm đúng
    bẫy này; train dùng 256 nên không lộ.
    """
    config = model.encoder.config
    limit = int(getattr(config, "max_position_embeddings", 512))
    padding = int(getattr(config, "pad_token_id", 0) or 0)
    return limit - padding - 1


def _encode_words(words, tokenizer, max_len, bos, eos, unk):
    input_ids = [bos]
    first, last = [], []
    limit = 0
    for word in words:
        subwords = tokenizer.encode(word, add_special_tokens=False) or [unk]
        if len(input_ids) + len(subwords) + 1 > max_len:
            break
        first.append(len(input_ids))
        input_ids.extend(subwords)
        last.append(len(input_ids) - 1)
        limit += 1
    input_ids.append(eos)
    return input_ids, first, last, limit


def _decode_bio(labels, probs, tokens):
    candidates = []
    index = 0
    while index < len(labels):
        label = labels[index]
        if label == "O":
            index += 1
            continue
        typ = label.split("-", 1)[-1]
        end = index + 1
        while end < len(labels) and labels[end] == f"I-{typ}":
            end += 1
        candidates.append({
            "word_start": index,
            "word_end": end,
            "type": typ,
            "position": [tokens[index][1], tokens[end - 1][2]],
            "bio_confidence": sum(probs[index:end]) / (end - index),
        })
        index = end
    return candidates


@torch.no_grad()
def predict_text(
    text,
    model,
    tokenizer,
    meta,
    device,
    max_len=256,
    max_words=180,
    overlap_words=45,
    type_threshold=0.60,
    disagreement_threshold=0.85,
    assertion_threshold=0.60,
):
    labels = meta["labels"]
    types = meta["types"]
    assertions = meta["assertions"]
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.cls_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.sep_token_id
    unk = tokenizer.unk_token_id or tokenizer.pad_token_id or 0
    predictions = []
    for window in sliding_windows(text, max_words, overlap_words, include_header=True):
        prefix, content = window["prefix"], window["content"]
        words = [token for token, _, _ in prefix + content]
        input_ids, first, last, word_limit = _encode_words(
            words, tokenizer, max_len, bos, eos, unk
        )
        prefix_len = min(len(prefix), word_limit)
        content_limit = max(0, word_limit - prefix_len)
        if not content_limit:
            continue
        tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(tensor)
        result = model(tensor, mask)
        bio_probability = torch.softmax(result["bio_logits"][0], dim=-1)
        word_label_ids = [
            int(bio_probability[position].argmax()) for position in first[prefix_len:word_limit]
        ]
        word_labels = [labels[label_id] for label_id in word_label_ids]
        word_confidence = [
            float(bio_probability[position, label_id])
            for position, label_id in zip(first[prefix_len:word_limit], word_label_ids)
        ]
        visible_content = content[:content_limit]
        candidates = _decode_bio(word_labels, word_confidence, visible_content)
        if not candidates:
            continue
        span_starts = torch.tensor([[
            first[prefix_len + candidate["word_start"]] for candidate in candidates
        ]], device=device)
        span_ends = torch.tensor([[
            last[prefix_len + candidate["word_end"] - 1] for candidate in candidates
        ]], device=device)
        scored = model(tensor, mask, span_starts=span_starts, span_ends=span_ends)
        span_probabilities = torch.softmax(scored["span_logits"][0], dim=-1)
        assertion_probabilities = torch.sigmoid(scored["assertion_logits"][0])
        for index, candidate in enumerate(candidates):
            span_type_id = int(span_probabilities[index].argmax())
            span_confidence = float(span_probabilities[index, span_type_id])
            bio_type = candidate["type"]
            if span_type_id == 0:
                if span_confidence >= type_threshold:
                    continue
                final_type = bio_type
                confidence = candidate["bio_confidence"]
            else:
                span_type = types[span_type_id - 1]
                if span_type == bio_type:
                    final_type = bio_type
                    confidence = (candidate["bio_confidence"] + span_confidence) / 2
                else:
                    if max(candidate["bio_confidence"], span_confidence) < disagreement_threshold:
                        continue
                    final_type = (
                        span_type if span_confidence > candidate["bio_confidence"] else bio_type
                    )
                    confidence = max(candidate["bio_confidence"], span_confidence)
            start, end = candidate["position"]
            entity_assertions = []
            if final_type in ASSERTION_TYPES:
                entity_assertions = [
                    name for assertion_index, name in enumerate(assertions)
                    if float(assertion_probabilities[index, assertion_index]) >= assertion_threshold
                ]
            predictions.append({
                "text": text[start:end],
                "position": [start, end],
                "type": final_type,
                "assertions": entity_assertions,
                "candidates": [],
                "_confidence": confidence,
            })
    # Hợp nhất overlap từ sliding windows. Same span/type giữ confidence cao nhất;
    # khác span cạnh tranh thì ưu tiên confidence cao, sau đó span dài hơn.
    exact = {}
    for entity in predictions:
        key = (tuple(entity["position"]), entity["type"])
        if key not in exact or entity["_confidence"] > exact[key]["_confidence"]:
            exact[key] = entity
    chosen = []
    for entity in sorted(
        exact.values(),
        key=lambda row: (-row["_confidence"], -(row["position"][1] - row["position"][0]), row["position"][0]),
    ):
        start, end = entity["position"]
        if any(not (end <= other["position"][0] or start >= other["position"][1]) for other in chosen):
            continue
        chosen.append(entity)
    chosen.sort(key=lambda row: row["position"])
    for entity in chosen:
        entity.pop("_confidence", None)
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Inference hybrid NER v2")
    parser.add_argument("--model-dir", "--model", dest="model_dir", required=True)
    parser.add_argument("--input-dir", "--input", dest="input_dir", required=True)
    parser.add_argument("--out-dir", "--out", dest="out_dir", required=True)
    parser.add_argument(
        "--max-len", type=int, default=0,
        help="0 = lấy theo lúc train, và luôn bị kẹp theo sức chứa của encoder",
    )
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument("--type-threshold", type=float, default=0.60)
    parser.add_argument("--disagreement-threshold", type=float, default=0.85)
    parser.add_argument("--assertion-threshold", type=float, default=0.60)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model, meta = HybridNER.load(args.model_dir, device)
    capacity = encoder_capacity(model)
    max_len = args.max_len or int(meta.get("max_len") or capacity)
    if max_len > capacity:
        print(f"[cảnh báo] max_len={max_len} vượt sức chứa encoder; hạ về {capacity}")
        max_len = capacity
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.txt")),
        key=lambda path: int(Path(path).stem) if Path(path).stem.isdigit() else Path(path).stem,
    )
    for path in files:
        text = Path(path).read_text(encoding="utf-8")
        entities = predict_text(
            text, model, tokenizer, meta, device,
            max_len, args.max_words, args.overlap_words,
            args.type_threshold, args.disagreement_threshold, args.assertion_threshold,
        )
        (Path(args.out_dir) / f"{Path(path).stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{Path(path).name}: {len(entities)}")


if __name__ == "__main__":
    main()
