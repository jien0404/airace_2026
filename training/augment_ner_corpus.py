"""Append reviewed targeted pilots to an existing record-level NER corpus.

The targeted pilots are deliberately Track C/contaminated: some surfaces are sampled from
manual Part 3 errors.  This utility keeps the parent's validation and test files byte-for-byte
and appends only mechanically valid drafts to TRAIN.  It is intentionally separate from
``build_ner_corpus.build`` so a targeted ablation cannot silently reshuffle the validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .assertion_policy import filter_assertions
from .build_ner_corpus import describe, load_synthetic

ROOT = Path(__file__).resolve().parents[1]


def _repo(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pilot_gate(pilot: Path) -> dict[str, Any]:
    manifest_path = pilot / "pilot_manifest.json"
    audit_path = pilot / "audit_report.json"
    drafts_path = pilot / "drafts.jsonl"
    if not manifest_path.exists() or not audit_path.exists() or not drafts_path.exists():
        raise FileNotFoundError(f"pilot thiếu manifest/audit/drafts: {pilot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("targeted_scope") != "WER_and_assertion_only":
        raise RuntimeError(f"pilot không khóa scope WER/assertion: {pilot}")
    if manifest.get("candidates_included") or manifest.get("candidates_optimized"):
        raise RuntimeError(f"pilot có cờ candidate, không được đưa vào corpus: {pilot}")
    if not audit.get("valid"):
        raise RuntimeError(f"audit không PASS: {pilot}")
    drafts = _read_jsonl(drafts_path)
    expected = audit.get("draft_count")
    if expected is not None and len(drafts) != expected:
        raise RuntimeError(
            f"draft_count không khớp audit ở {pilot}: file={len(drafts)} audit={expected}"
        )
    invalid = [
        draft.get("draft_id")
        for draft in drafts
        if not str(draft.get("status") or "").startswith("mechanically_valid")
    ]
    if invalid:
        raise RuntimeError(f"pilot có draft không mechanically_valid: {pilot}: {invalid[:5]}")
    return {
        "manifest": manifest,
        "audit": audit,
        "drafts": drafts,
        "drafts_sha256": _sha256(drafts_path),
    }


def _normalize_assertions(records: list[dict[str, Any]]) -> Counter:
    """Re-apply the current business-rule assertion firewall at corpus ingestion."""
    corrections: Counter = Counter()
    for record in records:
        for entity in record.get("entities") or []:
            before = list(entity.get("assertions") or [])
            after = filter_assertions(entity.get("type"), before, "part3")
            if after != before:
                corrections[
                    f"{entity.get('type')}:" + ",".join(sorted(set(before) - set(after)))
                ] += 1
                entity["assertions"] = after
    return corrections


def _targeted_records(
    pilot: Path, drafts: list[dict[str, Any]], *, mask_assertions: bool = False,
) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    # Use the same canonical cleaning path as the normal corpus builder.  This verifies offsets
    # once more and drops no data silently: the caller compares the resulting count below.
    # Importing load_synthetic keeps the exact occurrence/assertion conventions in one place.
    loaded = load_synthetic(pilot, stats, mask_assertions=mask_assertions)
    if len(loaded) != len(drafts):
        raise RuntimeError(
            f"cleaning làm mất draft ở {pilot}: loaded={len(loaded)} drafts={len(drafts)} "
            f"stats={dict(stats)}"
        )
    corrections = _normalize_assertions(loaded)
    for record in loaded:
        raw_id = record["id"]
        record["id"] = f"{pilot.name}:{raw_id}"
        record["source"] = "synthetic_targeted_part3"
        record["targeted_pilot"] = pilot.name
        record["targeted_scope"] = "WER_and_assertion_only"
        record["candidates_included"] = False
    return loaded, corrections


def build(parent_dir: Path, pilot_dirs: list[Path], out_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    required = [parent_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"parent corpus thiếu split: {parent_dir}")
    parent_manifest_path = parent_dir / "manifest.json"
    if not parent_manifest_path.exists():
        raise FileNotFoundError(f"parent corpus thiếu manifest: {parent_manifest_path}")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"đích đã tồn tại: {out_dir}; dùng --overwrite nếu cần")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    parent_train = _read_jsonl(parent_dir / "train.jsonl")
    parent_validation_bytes = (parent_dir / "validation.jsonl").read_bytes()
    parent_test_bytes = (parent_dir / "test.jsonl").read_bytes()
    parent_ids = {record["id"] for record in parent_train}
    if len(parent_ids) != len(parent_train):
        raise RuntimeError("parent train có id trùng")

    all_added: list[dict[str, Any]] = []
    pilot_reports: list[dict[str, Any]] = []
    for pilot in pilot_dirs:
        checked = _pilot_gate(pilot)
        mode = checked["manifest"].get("mode")
        added, assertion_corrections = _targeted_records(
            pilot,
            checked["drafts"],
            mask_assertions=mode in {"wer", "k_wer"},
        )
        collisions = parent_ids & {record["id"] for record in added}
        if collisions:
            raise RuntimeError(f"id targeted đụng parent: {sorted(collisions)[:5]}")
        pilot_reports.append({
            "pilot": str(pilot),
            "pilot_id": pilot.name,
            "drafts_sha256": checked["drafts_sha256"],
            "records_added": len(added),
            "audit": checked["audit"].get("distribution", {}),
            "direct_replay": checked["manifest"].get("direct_replay", 0),
            "analogue": checked["manifest"].get("analogue", 0),
            "profiles": checked["manifest"].get("profiles", {}),
            "error_kinds": checked["manifest"].get("error_kinds", {}),
            "genres": checked["manifest"].get("genres", {}),
            "assertion_supervision_masked": mode in {"wer", "k_wer"},
            "assertion_corrections_at_ingest": dict(assertion_corrections),
        })
        all_added.extend(added)
        parent_ids.update(record["id"] for record in added)

    if not all_added:
        raise RuntimeError("không có targeted record nào để append")
    train = parent_train + all_added
    _write_jsonl(out_dir / "train.jsonl", train)
    (out_dir / "validation.jsonl").write_bytes(parent_validation_bytes)
    (out_dir / "test.jsonl").write_bytes(parent_test_bytes)

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    direct_replay_records = sum(
        int(report.get("direct_replay") or 0) for report in pilot_reports
    )
    manifest = {
        "schema_version": 1,
        "track": "A",
        "corpus_kind": "parent_plus_targeted_part3_patch",
        "parent_corpus": str(parent_dir),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "parent_manifest": parent_manifest.get("sources", {}),
        "targeted_scope": "WER_and_assertion_only",
        "candidates_included": False,
        "candidates_optimized": False,
        "contaminated": True,
        "contamination_note": (
            "Train có Track C được thiết kế từ error family của Part 3; "
            + (
                f"có {direct_replay_records} direct replay. "
                if direct_replay_records else
                "không có direct replay hay câu chép nguyên văn từ Part 3. "
            )
            + "Chỉ dùng để vá WER/assertion và không dùng local Part 3 để kết luận "
              "generalization lên private test."
        ),
        "direct_replay_records": direct_replay_records,
        "validation_test_policy": "copied_parent_byte_for_byte",
        "validation_sha256": _sha256(out_dir / "validation.jsonl"),
        "test_sha256": _sha256(out_dir / "test.jsonl"),
        "parent_validation_sha256": _sha256(parent_dir / "validation.jsonl"),
        "parent_test_sha256": _sha256(parent_dir / "test.jsonl"),
        "added_pilots": pilot_reports,
        "added_records": len(all_added),
        "splits": {"train": describe(train), "validation": describe(_read_jsonl(out_dir / "validation.jsonl")),
                   "test": describe(_read_jsonl(out_dir / "test.jsonl"))},
    }
    if manifest["validation_sha256"] != manifest["parent_validation_sha256"]:
        raise RuntimeError("validation không còn byte-for-byte giống parent")
    if manifest["test_sha256"] != manifest["parent_test_sha256"]:
        raise RuntimeError("test không còn byte-for-byte giống parent")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", required=True, help="parent track_a corpus")
    parser.add_argument("--pilot", action="append", required=True, help="reviewed pilot; repeatable")
    parser.add_argument("--out", required=True, help="new track_a corpus")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build(
        _repo(args.parent), [_repo(path) for path in args.pilot], _repo(args.out),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
