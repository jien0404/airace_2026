# -*- coding: utf-8 -*-
"""Đường dẫn artifact Part 3 hiện hành, đọc từ `MANIFEST.json`.

Trước đây tên file được ghi cứng ở sáu chỗ (`training/build_ner_corpus`,
`dataset_factory/build_archive`, `configs/dataset_v1.json`, `entity_clean/derived_rules.json`…),
nên đổi artifact là phải sửa tay từng chỗ và rất dễ sót một chỗ trỏ vào bản cũ mà không ai biết.
`MANIFEST.json` vốn đã là nguồn duy nhất đúng — module này chỉ làm cho code đọc được nó.
"""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACTS_DIR.parents[1]
MANIFEST_PATH = ARTIFACTS_DIR / "MANIFEST.json"


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def current_labels_zip() -> Path:
    """ZIP nhãn Part 3 hiện hành. Là NHÃN BÀI NỘP TỐT NHẤT, không phải gold ban tổ chức."""
    path = REPO_ROOT / manifest()["current"]["path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact hiện hành không tồn tại: {path}. "
            "Chạy `python business_rules/scripts/verify_current.py` để kiểm."
        )
    return path


def current_name() -> str:
    return manifest()["current"]["name"]
