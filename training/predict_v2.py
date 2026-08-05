from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import torch

from .schema_v2 import ASSERTION_TYPES
from .assertion_policy import (
    POLICIES,
    apply_optional_historical_education_firewall,
    assertions_from_probabilities,
    load_type_thresholds,
    resolve_thresholds,
)

from .model_v2 import HybridNER
from .windowing_v2 import sliding_windows

ASSERTION_AGGREGATIONS = ("selected", "max")


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


def _merge_predictions(
    predictions,
    assertion_names,
    assertion_thresholds,
    assertion_policy,
    assertion_aggregation="selected",
    assertion_type_thresholds=None,
    keep_diagnostics=False,
):
    """Gộp output các cửa sổ và chốt assertion sau khi gộp.

    ``selected`` giữ hành vi cũ: assertion lấy từ bản sao có NER confidence cao nhất.
    ``max`` giữ span/type theo bản sao tốt nhất nhưng lấy max probability assertion qua
    mọi bản sao có cùng span/type. Đây là ablation inference-only cho cue nằm ở một cửa sổ
    overlap khác; không thay đổi candidate, span hay type.
    """
    if assertion_aggregation not in ASSERTION_AGGREGATIONS:
        raise ValueError(f"Assertion aggregation không hợp lệ: {assertion_aggregation}")
    exact = {}
    for entity in predictions:
        key = (tuple(entity["position"]), entity["type"])
        current = exact.get(key)
        if current is None:
            exact[key] = entity
            continue
        aggregate = None
        if assertion_aggregation == "max":
            left = current.get("_assertion_probabilities") or []
            right = entity.get("_assertion_probabilities") or []
            if len(left) != len(right):
                raise ValueError("Số assertion probability giữa các cửa sổ không khớp")
            aggregate = [max(a, b) for a, b in zip(left, right)]
        if entity["_confidence"] > current["_confidence"]:
            exact[key] = entity
            current = entity
        if aggregate is not None:
            current["_assertion_probabilities"] = aggregate

    chosen = []
    for entity in sorted(
        exact.values(),
        key=lambda row: (
            -row["_confidence"],
            -(row["position"][1] - row["position"][0]),
            row["position"][0],
        ),
    ):
        start, end = entity["position"]
        if any(
            not (end <= other["position"][0] or start >= other["position"][1])
            for other in chosen
        ):
            continue
        chosen.append(entity)
    chosen.sort(key=lambda row: row["position"])
    for entity in chosen:
        probabilities = entity.pop("_assertion_probabilities", [])
        if entity["type"] in ASSERTION_TYPES:
            entity["assertions"] = assertions_from_probabilities(
                entity["type"],
                assertion_names,
                probabilities,
                assertion_thresholds,
                assertion_policy,
                assertion_type_thresholds,
            )
        else:
            entity["assertions"] = []
        if keep_diagnostics:
            entity["_assertion_probabilities"] = probabilities
        else:
            for key in tuple(entity):
                if key.startswith("_"):
                    entity.pop(key, None)
    return chosen


def _threshold_for_type(default: float, mapping: dict | None, entity_type: str) -> float:
    value = (mapping or {}).get(entity_type, default)
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"Threshold {entity_type} phải nằm trong [0,1], nhận {value}")
    return value


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
    assertion_thresholds=None,
    assertion_type_thresholds=None,
    assertion_policy="part3",
    assertion_aggregation="selected",
    ner_type_thresholds=None,
    prefix_subword_cap=64,
    tokenizer_aware_windows=True,
    keep_diagnostics=False,
    historical_education_firewall=False,
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
    per_assertion_thresholds = resolve_thresholds(
        assertion_threshold, assertion_thresholds
    )
    predictions = []
    for window in sliding_windows(
        text, max_words, overlap_words, include_header=True,
        tokenizer=tokenizer if tokenizer_aware_windows else None,
        max_len=max_len if tokenizer_aware_windows else None,
        prefix_subword_cap=prefix_subword_cap,
    ):
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
            veto_threshold = _threshold_for_type(
                type_threshold, (ner_type_thresholds or {}).get("span_veto"), bio_type
            )
            disagreement_gate = _threshold_for_type(
                disagreement_threshold,
                (ner_type_thresholds or {}).get("disagreement"),
                bio_type,
            )
            if span_type_id == 0:
                if span_confidence >= veto_threshold:
                    continue
                final_type = bio_type
                confidence = candidate["bio_confidence"]
                decision = "bio_overrode_span_none"
            else:
                span_type = types[span_type_id - 1]
                if span_type == bio_type:
                    final_type = bio_type
                    confidence = (candidate["bio_confidence"] + span_confidence) / 2
                    decision = "heads_agree"
                else:
                    if max(candidate["bio_confidence"], span_confidence) < disagreement_gate:
                        continue
                    final_type = (
                        span_type if span_confidence > candidate["bio_confidence"] else bio_type
                    )
                    confidence = max(candidate["bio_confidence"], span_confidence)
                    decision = "high_confidence_head_wins"
            start, end = candidate["position"]
            assertion_probabilities_for_entity = []
            if final_type in ASSERTION_TYPES:
                assertion_probabilities_for_entity = [
                    float(assertion_probabilities[index, assertion_index])
                    for assertion_index in range(len(assertions))
                ]
            predictions.append({
                "text": text[start:end],
                "position": [start, end],
                "type": final_type,
                "assertions": [],
                "candidates": [],
                "_confidence": confidence,
                "_assertion_probabilities": assertion_probabilities_for_entity,
                "_bio_confidence": candidate["bio_confidence"],
                "_span_confidence": span_confidence,
                "_span_type": None if span_type_id == 0 else types[span_type_id - 1],
                "_decision": decision,
            })
    # Hợp nhất overlap từ sliding windows. Same span/type giữ confidence cao nhất;
    # khác span cạnh tranh thì ưu tiên confidence cao, sau đó span dài hơn.
    merged = _merge_predictions(
        predictions,
        assertions,
        per_assertion_thresholds,
        assertion_policy,
        assertion_aggregation,
        assertion_type_thresholds,
        keep_diagnostics,
    )
    if historical_education_firewall:
        merged, _ = apply_optional_historical_education_firewall(text, merged)
    return merged


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
    parser.add_argument("--prefix-subword-cap", type=int, default=64)
    parser.add_argument(
        "--legacy-word-windows", action="store_true",
        help="Giữ cắt 180 từ rồi truncate như luồng cũ; chỉ dùng để ablation",
    )
    parser.add_argument("--type-threshold", type=float, default=0.60)
    parser.add_argument("--disagreement-threshold", type=float, default=0.85)
    parser.add_argument("--assertion-threshold", type=float, default=0.60)
    parser.add_argument(
        "--historical-threshold", type=float,
        help="Threshold riêng cho isHistorical; mặc định dùng --assertion-threshold",
    )
    parser.add_argument(
        "--negated-threshold", type=float,
        help="Threshold riêng cho isNegated; mặc định dùng --assertion-threshold",
    )
    parser.add_argument(
        "--family-threshold", type=float,
        help="Threshold riêng cho isFamily; policy part3 vẫn luôn loại isFamily",
    )
    parser.add_argument(
        "--assertion-threshold-map",
        help=(
            "JSON threshold theo assertion và entity type; ví dụ "
            "{isHistorical: {CHẨN_ĐOÁN: 0.5}}. Giá trị này ưu tiên hơn các flag chung."
        ),
    )
    parser.add_argument(
        "--ner-threshold-map",
        help=(
            "JSON threshold theo type cho span_veto/disagreement; ví dụ "
            "{span_veto: {TRIỆU_CHỨNG: 0.65}, disagreement: {...}}"
        ),
    )
    parser.add_argument(
        "--confidence-out",
        help="Thư mục sidecar confidence phục vụ calibration/audit; không đổi JSON nộp bài",
    )
    parser.add_argument(
        "--historical-education-firewall", action="store_true",
        help="Rule thử nghiệm, mặc định TẮT; chỉ bật sau gate v67 + external reviewed",
    )
    parser.add_argument(
        "--assertion-policy", choices=POLICIES, default="part3",
        help="part3 áp firewall đã được probe xác nhận; legacy giữ output model nguyên trạng",
    )
    parser.add_argument(
        "--assertion-aggregation", choices=ASSERTION_AGGREGATIONS, default="selected",
        help="selected giữ cửa sổ NER-confidence cao nhất; max gộp max probability assertion qua overlap",
    )
    args = parser.parse_args()
    assertion_type_thresholds = (
        load_type_thresholds(Path(args.assertion_threshold_map))
        if args.assertion_threshold_map else {}
    )
    ner_type_thresholds = {}
    if args.ner_threshold_map:
        ner_type_thresholds = json.loads(
            Path(args.ner_threshold_map).read_text(encoding="utf-8")
        )
        unknown = set(ner_type_thresholds) - {"span_veto", "disagreement"}
        if unknown:
            parser.error(f"Nhóm threshold NER không hợp lệ: {sorted(unknown)}")
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
    if args.confidence_out:
        os.makedirs(args.confidence_out, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.txt")),
        key=lambda path: int(Path(path).stem) if Path(path).stem.isdigit() else Path(path).stem,
    )
    for path in files:
        text = Path(path).read_text(encoding="utf-8")
        entities = predict_text(
            text, model, tokenizer, meta, device,
            max_len=max_len,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            type_threshold=args.type_threshold,
            disagreement_threshold=args.disagreement_threshold,
            assertion_threshold=args.assertion_threshold,
            assertion_thresholds={
                "isHistorical": args.historical_threshold,
                "isNegated": args.negated_threshold,
                "isFamily": args.family_threshold,
            },
            assertion_type_thresholds=assertion_type_thresholds,
            assertion_policy=args.assertion_policy,
            assertion_aggregation=args.assertion_aggregation,
            ner_type_thresholds=ner_type_thresholds,
            prefix_subword_cap=args.prefix_subword_cap,
            tokenizer_aware_windows=not args.legacy_word_windows,
            keep_diagnostics=bool(args.confidence_out),
            historical_education_firewall=args.historical_education_firewall,
        )
        if args.confidence_out:
            diagnostics = []
            for entity in entities:
                diagnostics.append({
                    "text": entity["text"],
                    "position": entity["position"],
                    "type": entity["type"],
                    "assertions": entity["assertions"],
                    "final_confidence": entity.get("_confidence"),
                    "bio_confidence": entity.get("_bio_confidence"),
                    "span_confidence": entity.get("_span_confidence"),
                    "span_type": entity.get("_span_type"),
                    "decision": entity.get("_decision"),
                    "assertion_probabilities": dict(zip(
                        meta.get("assertions") or [],
                        entity.get("_assertion_probabilities", []),
                    )),
                })
                for key in tuple(entity):
                    if key.startswith("_"):
                        entity.pop(key, None)
            (Path(args.confidence_out) / f"{Path(path).stem}.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (Path(args.out_dir) / f"{Path(path).stem}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{Path(path).name}: {len(entities)}")


if __name__ == "__main__":
    main()
