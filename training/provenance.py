"""Fail-fast provenance checks and immutable fingerprints for training datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifact_helpers():
    """Load Part 3 artifact helpers only when an artifact check is requested.

    A normal training run only needs the dataset manifest.  Keeping this import
    lazy allows portable dataset/training bundles to run without carrying the
    ignored, large Part 3 artifact ZIP.  The strict SHA check still fails with a
    useful message if that artifact is explicitly requested but unavailable.
    """
    try:
        from business_rules.artifacts import (  # pylint: disable=import-outside-toplevel
            current_labels_zip,
            manifest as artifact_manifest,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "business_rules.artifacts":
            raise
        raise RuntimeError(
            "Thiếu business_rules/artifacts. Chỉ cần chép thư mục artifact "
            "hoặc bỏ --require-part3-sha256 khi chạy training-only."
        ) from exc
    return current_labels_zip, artifact_manifest


def inspect_dataset(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else None
    )
    splits = {}
    for name in ("train", "validation", "test"):
        path = data_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset thiếu {path}")
        splits[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    report_path = data_dir / "build_report.json"
    build_report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file() else None
    )
    return {
        "data_dir": str(data_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()) if manifest is not None else None,
        "manifest": manifest,
        "build_report": build_report,
        "splits": splits,
        "validation_reuses_test": (
            splits["validation"]["sha256"] == splits["test"]["sha256"]
        ),
    }


def verify_dataset(
    data_dir: Path,
    *,
    require_manifest: bool = False,
    allowed_variants: Iterable[str] = (),
    required_part3_sha256: str | None = None,
) -> dict[str, Any]:
    report = inspect_dataset(data_dir)
    manifest = report["manifest"]
    if require_manifest and manifest is None:
        raise RuntimeError(f"Dataset {data_dir} thiếu manifest.json")
    if allowed_variants:
        variant = (manifest or {}).get("variant")
        allowed = set(allowed_variants)
        if variant not in allowed:
            raise RuntimeError(
                f"Sai dataset variant: {variant!r}; chỉ chấp nhận {sorted(allowed)}"
            )
    if required_part3_sha256:
        current_labels_zip, artifact_manifest = _load_artifact_helpers()
        dataset_sha = (
            ((manifest or {}).get("part3_artifact") or {}).get("sha256")
            or ((manifest or {}).get("source_sha256") or {}).get("part3_labels")
        )
        current = artifact_manifest()["current"]
        current_sha = current.get("sha256")
        actual_sha = sha256(current_labels_zip())
        failures = []
        if dataset_sha != required_part3_sha256:
            failures.append(f"dataset={dataset_sha}")
        if current_sha != required_part3_sha256:
            failures.append(f"MANIFEST.current={current_sha}")
        if actual_sha != required_part3_sha256:
            failures.append(f"artifact_file={actual_sha}")
        if failures:
            raise RuntimeError(
                "Part3 provenance không khớp SHA bắt buộc "
                f"{required_part3_sha256}: " + ", ".join(failures)
            )
    return report
