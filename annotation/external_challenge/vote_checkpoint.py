"""Use a trained HybridNER checkpoint to adjudicate the external LLM draft.

The source LLM labels are never overwritten.  The default ``labels/`` output is conservative
strict voting: an entity survives only when model and LLM agree on the exact RAW span and type;
assertions use intersection.  Alternative model-only, LLM-only and relaxed-overlap outputs plus
per-record audits are kept for human review.

Example::

    CUDA_VISIBLE_DEVICES=1 python -m annotation.external_challenge.vote_checkpoint \
      --data-dir annotation/data/external_benhvien108_qa_v1 \
      --checkpoint runs/part3_n2_v3_v67_xlmr_large_30e_lr2e5/best \
      --out-dir annotation/data/external_benhvien108_qa_v1_model_vote_n2 \
      --assertion-policy part3

This is an evaluation/review aid only.  It does not add the external set to training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer

from annotation.part3_eval.align import validate_labels
from training.assertion_policy import filter_assertions
from training.model_v2 import HybridNER
from training.predict_v2 import encoder_capacity, predict_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "annotation/data/external_benhvien108_qa_v1"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _position(entity: dict[str, Any]) -> tuple[int, int]:
    start, end = entity["position"]
    return int(start), int(end)


def _key(entity: dict[str, Any]) -> tuple[tuple[int, int], str]:
    return _position(entity), str(entity.get("type"))


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    start = max(_position(left)[0], _position(right)[0])
    end = min(_position(left)[1], _position(right)[1])
    return max(0, end - start)


def _overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    common = _overlap(left, right)
    if not common:
        return 0.0
    left_len = _position(left)[1] - _position(left)[0]
    right_len = _position(right)[1] - _position(right)[0]
    return common / max(1, min(left_len, right_len))


def _clean_entity(entity: dict[str, Any], policy: str) -> dict[str, Any]:
    output = {
        "text": entity["text"],
        "position": list(entity["position"]),
        "type": entity["type"],
        "assertions": filter_assertions(
            entity["type"], entity.get("assertions") or (), policy
        ),
        "candidates": [],
    }
    return output


def _dedupe(labels: Iterable[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], str]] = set()
    for entity in labels:
        cleaned = _clean_entity(entity, policy)
        key = _key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return sorted(output, key=lambda row: (_position(row), row["type"]))


def _assertion_intersection(
    llm: dict[str, Any], model: dict[str, Any], policy: str,
) -> list[str]:
    model_assertions = set(model.get("assertions") or ())
    llm_assertions = filter_assertions(
        llm["type"], llm.get("assertions") or (), policy
    )
    return [name for name in llm_assertions if name in model_assertions]


def _match_audit(
    llm_labels: list[dict[str, Any]], model_labels: list[dict[str, Any]], policy: str,
) -> dict[str, Any]:
    model_by_key = {_key(row): row for row in model_labels}
    llm_by_key = {_key(row): row for row in llm_labels}
    exact = []
    llm_only = []
    model_only = []
    assertion_disagreements = []
    type_conflicts = []
    boundary_conflicts = []

    for llm in llm_labels:
        key = _key(llm)
        model = model_by_key.get(key)
        if model is not None:
            exact.append({"llm": llm, "model": model})
            llm_assertions = filter_assertions(llm["type"], llm.get("assertions") or (), policy)
            model_assertions = filter_assertions(model["type"], model.get("assertions") or (), policy)
            if llm_assertions != model_assertions:
                assertion_disagreements.append({
                    "position": list(_position(llm)),
                    "type": llm["type"],
                    "llm": llm_assertions,
                    "model": model_assertions,
                })
            continue

        same_type = [
            row for row in model_labels
            if row.get("type") == llm.get("type") and _overlap(llm, row) > 0
        ]
        if same_type:
            best = max(same_type, key=lambda row: (_overlap_ratio(llm, row), -abs(
                (_position(row)[1] - _position(row)[0])
                - (_position(llm)[1] - _position(llm)[0])
            )))
            boundary_conflicts.append({
                "llm": llm,
                "model": best,
                "overlap_ratio": round(_overlap_ratio(llm, best), 4),
            })
        else:
            same_span = [row for row in model_labels if _overlap(llm, row) > 0]
            if same_span:
                type_conflicts.append({"llm": llm, "model": same_span[0]})
            llm_only.append(llm)

    for model in model_labels:
        if _key(model) not in llm_by_key:
            model_only.append(model)

    return {
        "llm_count": len(llm_labels),
        "model_count": len(model_labels),
        "exact_agreements": len(exact),
        "llm_only": len(llm_only),
        "model_only": len(model_only),
        "boundary_conflicts": len(boundary_conflicts),
        "type_conflicts": len(type_conflicts),
        "assertion_disagreements": len(assertion_disagreements),
        "exact": exact,
        "llm_only_entities": llm_only,
        "model_only_entities": model_only,
        "boundary_conflicts_detail": boundary_conflicts,
        "type_conflicts_detail": type_conflicts,
        "assertion_disagreements_detail": assertion_disagreements,
    }


def _strict_vote(
    llm_labels: list[dict[str, Any]], model_labels: list[dict[str, Any]], policy: str,
) -> list[dict[str, Any]]:
    model_by_key = {_key(row): row for row in model_labels}
    output = []
    for llm in llm_labels:
        model = model_by_key.get(_key(llm))
        if model is None:
            continue
        entity = _clean_entity(llm, policy)
        entity["assertions"] = _assertion_intersection(llm, model, policy)
        output.append(entity)
    return _dedupe(output, policy)


def _relaxed_vote(
    llm_labels: list[dict[str, Any]], model_labels: list[dict[str, Any]],
    policy: str, min_overlap: float,
) -> list[dict[str, Any]]:
    """Keep exact agreements and high-overlap same-type matches, using model boundary."""
    used: set[int] = set()
    output = []
    for llm in llm_labels:
        exact = next((
            (index, row) for index, row in enumerate(model_labels)
            if index not in used and _key(row) == _key(llm)
        ), None)
        if exact is not None:
            index, model = exact
        else:
            candidates = [
                (index, row) for index, row in enumerate(model_labels)
                if index not in used and row.get("type") == llm.get("type")
                and _overlap_ratio(llm, row) >= min_overlap
            ]
            if not candidates:
                continue
            index, model = max(candidates, key=lambda item: _overlap_ratio(llm, item[1]))
        used.add(index)
        entity = _clean_entity(model, policy)
        entity["assertions"] = _assertion_intersection(llm, model, policy)
        output.append(entity)
    return _dedupe(output, policy)


def _copy_source_layout(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Output đã tồn tại: {target}; dùng --overwrite nếu muốn thay")
        shutil.rmtree(target)
    (target / "notes").mkdir(parents=True)
    for note in sorted((source / "notes").glob("*.txt")):
        shutil.copy2(note, target / "notes" / note.name)
    if (source / "metadata.jsonl").exists():
        shutil.copy2(source / "metadata.jsonl", target / "metadata.jsonl")
    if (source / "manifest.json").exists():
        shutil.copy2(source / "manifest.json", target / "source_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument("--type-threshold", type=float, default=0.60)
    parser.add_argument("--disagreement-threshold", type=float, default=0.85)
    parser.add_argument("--assertion-threshold", type=float, default=0.60)
    parser.add_argument("--assertion-policy", choices=("legacy", "part3"), default="part3")
    parser.add_argument("--assertion-aggregation", choices=("selected", "max"), default="selected")
    parser.add_argument("--relaxed-overlap", type=float, default=0.80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    target = args.out_dir.resolve()
    if not (source / "notes").is_dir() or not (source / "labels").is_dir():
        raise FileNotFoundError(f"External dataset thiếu notes/labels: {source}")
    if not (checkpoint / "hybrid_meta.json").is_file():
        raise FileNotFoundError(f"Checkpoint không phải HybridNER hoặc thiếu hybrid_meta.json: {checkpoint}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng torch không thấy GPU")
    model, meta = HybridNER.load(str(checkpoint), device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    max_len = min(args.max_len, encoder_capacity(model))

    _copy_source_layout(source, target, args.overwrite)
    for subdir in (
        "labels", "model_labels", "llm_labels", "relaxed_labels", "audits",
    ):
        (target / subdir).mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    metadata_path = source / "metadata.jsonl"
    if metadata_path.exists():
        metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata_by_id = {str(row.get("file_id")): row for row in metadata_rows}

    notes = sorted(
        (source / "notes").glob("*.txt"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    summary = Counter()
    for sequence, note_path in enumerate(notes, 1):
        file_id = note_path.stem
        raw = note_path.read_text(encoding="utf-8")
        llm = _dedupe(_load(source / "labels" / f"{file_id}.json"), args.assertion_policy)
        model_labels = predict_text(
            raw, model, tokenizer, meta, device,
            max_len=max_len,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
            type_threshold=args.type_threshold,
            disagreement_threshold=args.disagreement_threshold,
            assertion_threshold=args.assertion_threshold,
            assertion_policy=args.assertion_policy,
            assertion_aggregation=args.assertion_aggregation,
        )
        model_labels = _dedupe(model_labels, args.assertion_policy)
        strict = _strict_vote(llm, model_labels, args.assertion_policy)
        relaxed = _relaxed_vote(llm, model_labels, args.assertion_policy, args.relaxed_overlap)
        audit = _match_audit(llm, model_labels, args.assertion_policy)
        audit.update({
            "file_id": file_id,
            "raw_characters": len(raw),
            "strict_vote_count": len(strict),
            "relaxed_vote_count": len(relaxed),
            "strict_validation_errors": validate_labels(raw, strict),
            "relaxed_validation_errors": validate_labels(raw, relaxed),
            "model_validation_errors": validate_labels(raw, model_labels),
        })
        _dump(target / "llm_labels" / f"{file_id}.json", llm)
        _dump(target / "model_labels" / f"{file_id}.json", model_labels)
        _dump(target / "labels" / f"{file_id}.json", strict)
        _dump(target / "relaxed_labels" / f"{file_id}.json", relaxed)
        _dump(target / "audits" / f"{file_id}.json", audit)

        row = deepcopy(metadata_by_id.get(file_id, {"file_id": file_id}))
        prior_strata = list(row.get("strata") or [])
        new_strata = ["model_vote"]
        if audit["llm_only"] or audit["model_only"] or audit["boundary_conflicts"] or audit["type_conflicts"]:
            new_strata.append("entity_disagreement")
        if audit["assertion_disagreements"]:
            new_strata.append("assertion_disagreement")
        row.update({
            "strata": sorted(set(prior_strata + new_strata)),
            "vote_strict_entities": len(strict),
            "vote_relaxed_entities": len(relaxed),
            "model_entities": len(model_labels),
            "llm_entities": len(llm),
            "exact_agreements": audit["exact_agreements"],
            "llm_only": audit["llm_only"],
            "model_only": audit["model_only"],
            "assertion_disagreements": audit["assertion_disagreements"],
        })
        metadata_by_id[file_id] = row
        summary.update({
            "records": 1,
            "llm_entities": len(llm),
            "model_entities": len(model_labels),
            "strict_entities": len(strict),
            "relaxed_entities": len(relaxed),
            "exact_agreements": audit["exact_agreements"],
            "llm_only": audit["llm_only"],
            "model_only": audit["model_only"],
            "assertion_disagreements": audit["assertion_disagreements"],
        })
        print(
            f"[{sequence}/{len(notes)}] {file_id}: llm={len(llm)} model={len(model_labels)} "
            f"strict={len(strict)} relaxed={len(relaxed)} "
            f"agree={audit['exact_agreements']}",
            flush=True,
        )

    _write_jsonl(
        target / "metadata.jsonl",
        [metadata_by_id[key] for key in sorted(
            metadata_by_id, key=lambda value: int(value) if value.isdigit() else value
        )],
    )
    _dump(target / "_reviewed.json", [])
    source_manifest = _load(source / "manifest.json") if (source / "manifest.json").exists() else {}
    manifest = {
        "schema_version": 1,
        "stage": "external_challenge_checkpoint_vote_pending_human_review",
        "source_dir": str(source),
        "source_manifest_stage": source_manifest.get("stage"),
        "checkpoint": str(checkpoint),
        "checkpoint_meta": meta,
        "device": device,
        "inference": {
            "max_len": max_len,
            "max_words": args.max_words,
            "overlap_words": args.overlap_words,
            "type_threshold": args.type_threshold,
            "disagreement_threshold": args.disagreement_threshold,
            "assertion_threshold": args.assertion_threshold,
            "assertion_policy": args.assertion_policy,
            "assertion_aggregation": args.assertion_aggregation,
        },
        "voting": {
            "default_labels": "labels/ = strict exact span+type intersection; assertion intersection",
            "relaxed_labels": f"relaxed_labels/ = same-type overlap >= {args.relaxed_overlap}; model boundary; assertion intersection",
            "model_labels": "model_labels/ = checkpoint output",
            "llm_labels": "llm_labels/ = copied original draft",
            "human_review_required": True,
            "training_use": False,
        },
        "summary": dict(summary),
    }
    _dump(target / "manifest.json", manifest)
    print(json.dumps({"output": str(target), "summary": dict(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
