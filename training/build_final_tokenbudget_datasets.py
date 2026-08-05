"""Build portable Final-N corpora with tokenizer-budgeted, gap-free windows.

No data is generated here.  The script only re-windows the already frozen v3-v67 raw corpora
with the XLM-R tokenizer and fails on provenance drift, subword overflow, or any newly clipped
gold entity.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

from .build_dataset_v2 import build_track
from .provenance import sha256, verify_dataset


ROOT = Path(__file__).resolve().parents[1]
PART3_SHA256 = "e803003cee5a43d740bf4024e65b82d7a6a7214389e3338c0d83352333959b9b"

SPECS = {
    "n1_control": {
        "variant": "n1",
        "raw": "ner_N1_final_generalization_v3_v67_part3_heldout_20260804",
        "legacy": "ner_N1_final_generalization_v3_v67_part3_heldout_windows_20260804",
        "out": "ner_N1_v3_v67_tb256_ok35_windows_20260805",
        "o_keep": 0.35,
    },
    "n1_regularized": {
        "variant": "n1",
        "raw": "ner_N1_final_generalization_v3_v67_part3_heldout_20260804",
        "legacy": "ner_N1_final_generalization_v3_v67_part3_heldout_windows_20260804",
        "out": "ner_N1_v3_v67_tb256_ok50_windows_20260805",
        "o_keep": 0.50,
    },
    "n2_balanced": {
        "variant": "n2",
        "raw": "ner_N2_final_generalization_v3_v67_part3_split_20260804",
        "legacy": "ner_N2_final_generalization_v3_v67_part3_split_windows_20260804",
        "out": "ner_N2_v3_v67_tb256_ok45_windows_20260805",
        "o_keep": 0.45,
    },
    "n3_control": {
        "variant": "n3",
        "raw": "ner_N3_final_generalization_v3_v67_private_refit_20260804",
        "legacy": "ner_N3_final_generalization_v3_v67_private_refit_windows_20260804",
        "out": "ner_N3_v3_v67_tb256_ok35_windows_20260805",
        "o_keep": 0.35,
    },
    "n3_regularized": {
        "variant": "n3",
        "raw": "ner_N3_final_generalization_v3_v67_private_refit_20260804",
        "legacy": "ner_N3_final_generalization_v3_v67_private_refit_windows_20260804",
        "out": "ner_N3_v3_v67_tb256_ok50_windows_20260805",
        "o_keep": 0.50,
    },
}
DEFAULT_SPECS = ("n1_control", "n1_regularized", "n3_control", "n3_regularized")


def build_one(key: str, spec: dict, datasets_dir: Path, tokenizer, tokenizer_name: str) -> dict:
    raw = datasets_dir / spec["raw"] / "track_a"
    legacy = datasets_dir / spec["legacy"] / "track_a"
    output = datasets_dir / spec["out"] / "track_a"
    provenance = verify_dataset(
        raw,
        require_manifest=True,
        allowed_variants=(spec["variant"],),
        required_part3_sha256=PART3_SHA256,
    )
    manifest = provenance["manifest"]
    report = build_track(
        raw, output,
        max_words=180,
        overlap_words=45,
        header_dropout=0.35,
        o_keep=spec["o_keep"],
        standalone_ratio=0.60,
        seed=int(manifest.get("seed", 20260804)),
        source_masses=manifest["sampling_masses"],
        tokenizer=tokenizer,
        max_len=256,
        prefix_subword_cap=64,
    )
    legacy_report_path = legacy / "build_report.json"
    legacy_errors = set()
    if legacy_report_path.is_file():
        legacy_errors = set(json.loads(
            legacy_report_path.read_text(encoding="utf-8")
        ).get("coverage_errors") or [])
    new_errors = set(report["coverage_errors"])
    introduced = sorted(new_errors - legacy_errors)
    overflow = sum(
        int(stats.get("subword_overflow_windows") or 0)
        for stats in report["splits"].values()
    )
    if introduced:
        raise RuntimeError(f"{key}: tokenizer windows clip entity mới: {introduced[:10]}")
    if overflow:
        raise RuntimeError(f"{key}: có {overflow} window vượt max_len=256")

    enriched = {
        **manifest,
        "window_artifact": {
            "schema_version": 2,
            "strategy": "tokenizer_budget_gap_free_v1",
            "tokenizer": tokenizer_name,
            "max_len": 256,
            "max_words_cap": 180,
            "overlap_raw_words": 45,
            "boundary_lookback_words": 45,
            "context_raw_words": 32,
            "prefix_subword_cap": 64,
            "header_dropout": 0.35,
            "o_keep": spec["o_keep"],
            "subword_overflow_windows": overflow,
            "legacy_coverage_errors": len(legacy_errors),
            "coverage_errors_after": len(new_errors),
            "new_coverage_errors": len(introduced),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = ROOT / f"{spec['out']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        artifact_root = datasets_dir / spec["out"]
        for path in sorted(artifact_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(artifact_root))
    return {
        "key": key,
        "data": str(output),
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "window_artifact": enriched["window_artifact"],
        "splits": report["splits"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--datasets-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--only", action="append", choices=tuple(SPECS))
    parser.add_argument("--out-report", type=Path, default=ROOT / "result/final_tokenbudget_build.json")
    args = parser.parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    selected = args.only or list(DEFAULT_SPECS)
    result = {
        key: build_one(key, SPECS[key], args.datasets_dir.resolve(), tokenizer, args.model)
        for key in selected
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
