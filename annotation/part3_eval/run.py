"""Chạy benchmark nghiệp vụ + prompt LLM trên 10 file Part 3.

Mặc định một lệnh sẽ:
1. xác minh artifact/rulebook;
2. tạo reference V66ab chỉ trên 10 file;
3. chạy bốn prompt variants, có cache/resume;
4. align RAW, lọc candidate không tồn tại, chạy gate;
5. tạo năm ZIP cùng manifest/report.

    python -m annotation.part3_eval.run

Dry-run không gọi API:

    python -m annotation.part3_eval.run --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from annotation.evidence.diff_labels import diff_file, summarize

from .align import align_predictions, validate_labels
from .prompts import build_messages, templates


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PACKAGE_DIR / "runs" / "v1_10files"
BENCHMARK_PATH = PACKAGE_DIR / "benchmark.json"
MANIFEST_PATH = ROOT / "business_rules" / "artifacts" / "MANIFEST.json"
CANONICAL_PATH = ROOT / "business_rules" / "CANONICAL.md"
NOTES_DIR = ROOT / "input_turn2"
ICD_PATH = ROOT / "dictionary" / "icd10_tt06_full.json"
RXNORM_PATH = (
    ROOT
    / "dictionary"
    / "RxNorm_full_prescribe_07062026"
    / "rrf"
    / "RXNCONSO.RRF"
)
NO_PROXY = "localhost,127.0.0.1,::1,.viettelpost.vn"

VARIANTS = (
    "official_only",
    "canonical_dump",
    "compiled",
    "compiled_verified",
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "part3_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line_id": {"type": "integer"},
                            "text": {"type": "string"},
                            "occurrence": {"type": "integer"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "TRIỆU_CHỨNG",
                                    "CHẨN_ĐOÁN",
                                    "TÊN_XÉT_NGHIỆM",
                                    "KẾT_QUẢ_XÉT_NGHIỆM",
                                    "THUỐC",
                                ],
                            },
                            "assertions": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "isNegated",
                                        "isHistorical",
                                        "isFamily",
                                    ],
                                },
                            },
                            "candidates": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "line_id",
                            "text",
                            "occurrence",
                            "type",
                            "assertions",
                            "candidates",
                        ],
                        "additionalProperties": False,
                    },
                },
                "self_check": {
                    "type": "object",
                    "properties": {
                        "scanned_all_lines": {"type": "boolean"},
                        "reviewed_missing_mentions": {"type": "boolean"},
                        "reviewed_overlap": {"type": "boolean"},
                    },
                    "required": [
                        "scanned_all_lines",
                        "reviewed_missing_mentions",
                        "reviewed_overlap",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": ["entities", "self_check"],
            "additionalProperties": False,
        },
    },
}
ZIP_NAMES = {
    "reference_v66ab": "00_reference_v66ab_10.zip",
    "official_only": "01_official_only_10.zip",
    "canonical_dump": "02_canonical_dump_10.zip",
    "compiled": "03_compiled_10.zip",
    "compiled_verified": "04_compiled_verified_10.zip",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def enable_proxy(proxy_url: str | None) -> None:
    if not proxy_url:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = proxy_url
    os.environ["NO_PROXY"] = NO_PROXY
    os.environ["no_proxy"] = NO_PROXY


def load_reference(
    artifact_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    labels: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(artifact_path) as archive:
        for name in archive.namelist():
            match = re.fullmatch(r"labels/(\d+)\.json", name)
            if match:
                labels[match.group(1)] = json.loads(archive.read(name))
    return labels


def verify_inputs(
    benchmark: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    manifest = load_json(MANIFEST_PATH)
    artifact = ROOT / manifest["current"]["path"]
    rulebook = ROOT / manifest["primary_rulebook"]["path"]
    errors: list[str] = []
    for label, path, expected in (
        ("artifact", artifact, manifest["current"]["sha256"]),
        ("rulebook", rulebook, manifest["primary_rulebook"]["sha256"]),
    ):
        if not path.is_file():
            errors.append(f"{label}: thiếu {path}")
        elif sha256_path(path) != expected:
            errors.append(f"{label}: checksum lệch MANIFEST.json")
    selected = benchmark.get("selected_files")
    if (
        not isinstance(selected, list)
        or len(selected) != 10
        or len(set(selected)) != 10
        or any(not isinstance(file_id, int) or not 1 <= file_id <= 100 for file_id in selected)
    ):
        errors.append("benchmark.json phải chứa đúng 10 selected_files duy nhất trong 1..100")
    for file_id in selected or []:
        if not (NOTES_DIR / f"{file_id}.txt").is_file():
            errors.append(f"thiếu RAW input_turn2/{file_id}.txt")
    if errors:
        raise RuntimeError("\n".join(errors))
    return manifest, artifact, rulebook


def load_valid_codes() -> tuple[set[str], set[str]]:
    if not ICD_PATH.is_file():
        raise FileNotFoundError(f"Thiếu ICD source: {ICD_PATH}")
    if not RXNORM_PATH.is_file():
        raise FileNotFoundError(f"Thiếu RxNorm source: {RXNORM_PATH}")

    valid_icd = {
        str(row.get("code", "")).upper()
        for row in load_json(ICD_PATH)
        if isinstance(row, dict) and row.get("code")
    }
    valid_rxnorm: set[str] = set()
    with RXNORM_PATH.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            code = line.partition("|")[0].strip()
            if code.isdigit():
                valid_rxnorm.add(code)
    return valid_icd, valid_rxnorm


def selection_stats(
    selected: list[int],
    reference: dict[str, list[dict[str, Any]]],
    artifact_path: Path,
) -> dict[str, Any]:
    from annotation.evidence.diff_labels import diff, load_labels

    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    archive_dir = ROOT / "business_rules" / "artifacts" / "archive"
    archive_diffs: dict[str, Any] = {}
    for archive in sorted(archive_dir.glob("*.zip")):
        changes = diff(load_labels(str(archive)), reference)
        archive_diffs[archive.name] = {
            "files_changed": len(changes),
            "ops": summarize(changes),
        }
        for file_id, operations in changes.items():
            by_file[file_id]["artifact_total"] += len(operations)
            by_file[file_id]["artifact_noncode"] += sum(
                operation["op"] != "CODE" for operation in operations
            )

    history_path = ROOT / "annotation" / "evidence" / "ops_index.jsonl"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            file_id = str(row["file"])
            by_file[file_id]["historical_ops"] += 1
            if (row.get("delta") or 0) > 0:
                by_file[file_id]["positive_ops"] += 1

    ranking: list[dict[str, Any]] = []
    for file_id in range(1, 101):
        counts = by_file[str(file_id)]
        rank_score = (
            3 * counts["artifact_noncode"]
            + counts["historical_ops"]
            + 2 * counts["positive_ops"]
        )
        ranking.append({
            "file_id": file_id,
            "rank_score": rank_score,
            **dict(counts),
            "selected": file_id in selected,
        })
    ranking.sort(key=lambda row: (-row["rank_score"], row["file_id"]))

    selected_rows: list[dict[str, Any]] = []
    for file_id in selected:
        labels = reference[str(file_id)]
        types = Counter(entity["type"] for entity in labels)
        selected_rows.append({
            "file_id": file_id,
            "rank": next(
                index for index, row in enumerate(ranking, 1)
                if row["file_id"] == file_id
            ),
            "rank_score": next(
                row["rank_score"] for row in ranking if row["file_id"] == file_id
            ),
            "characters": len((NOTES_DIR / f"{file_id}.txt").read_text(encoding="utf-8")),
            "reference_entities": len(labels),
            "reference_candidates": sum(bool(entity.get("candidates")) for entity in labels),
            "reference_asserted": sum(bool(entity.get("assertions")) for entity in labels),
            "reference_by_type": dict(types),
        })
    return {
        "artifact": str(artifact_path.relative_to(ROOT)),
        "archive_diffs": archive_diffs,
        "ranking_formula": (
            "3*artifact_noncode + historical_ops + 2*positive_ops"
        ),
        "selected": selected_rows,
        "top_25": ranking[:25],
    }


def _zip_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def write_partial_submission(
    path: Path,
    selected: set[int],
    labels: dict[str, list[dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_id in range(1, 101):
            value = labels.get(str(file_id), []) if file_id in selected else []
            info = zipfile.ZipInfo(f"labels/{file_id}.json")
            info.date_time = (2026, 7, 30, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, _zip_json_bytes(value))


def validate_submission(
    path: Path,
    selected: set[int],
    notes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    expected = {f"labels/{file_id}.json" for file_id in range(1, 101)}
    with zipfile.ZipFile(path) as archive:
        actual = set(archive.namelist())
        if actual != expected:
            errors.append(
                f"ZIP entries lệch: thiếu={sorted(expected-actual)} dư={sorted(actual-expected)}"
            )
        for file_id in range(1, 101):
            name = f"labels/{file_id}.json"
            if name not in actual:
                continue
            value = json.loads(archive.read(name))
            if not isinstance(value, list):
                errors.append(f"{name}: root không phải list")
                continue
            if file_id not in selected and value:
                errors.append(f"{name}: ngoài subset nhưng không rỗng")
            if file_id in selected:
                errors.extend(
                    f"{name}: {error}"
                    for error in validate_labels(notes[str(file_id)], value)
                )
    return errors


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list):
        raise ValueError("LLM output phải là object có entities:list")
    return value


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, key, None) is not None
    }


def _chat_create(client: Any, kwargs: dict[str, Any]) -> Any:
    """Tương thích deployment cũ bằng fallback hẹp, không đổi nội dung prompt."""
    current = dict(kwargs)
    for _ in range(5):
        try:
            return client.chat.completions.create(**current)
        except Exception as error:
            message = str(error).lower()
            if "temperature" in message and "temperature" in current:
                current.pop("temperature", None)
                continue
            if "max_completion_tokens" in message and "max_completion_tokens" in current:
                current["max_tokens"] = current.pop("max_completion_tokens")
                continue
            if (
                ("reasoning_effort" in message or "reasoning effort" in message)
                and "reasoning_effort" in current
            ):
                current.pop("reasoning_effort", None)
                continue
            response_format = current.get("response_format", {})
            if (
                ("json_schema" in message or "response_format" in message)
                and response_format.get("type") == "json_schema"
            ):
                current["response_format"] = {"type": "json_object"}
                continue
            raise
    raise RuntimeError("Không tìm được request shape tương thích với deployment")


def call_llm_json(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    reasoning_effort: str | None,
    max_retries: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    last_error: Exception | None = None
    repair_messages = messages
    for attempt in range(1, max_retries + 1):
        attempt_started = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": repair_messages,
            "response_format": RESPONSE_FORMAT,
            "max_completion_tokens": 16000,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            response = _chat_create(client, kwargs)
            content = response.choices[0].message.content or ""
            try:
                parsed = _parse_json_content(content)
            except (json.JSONDecodeError, ValueError) as parse_error:
                if attempt >= max_retries:
                    raise
                print(
                    f"\n    [retry {attempt}/{max_retries}] JSON không hợp lệ sau "
                    f"{time.monotonic() - attempt_started:.1f}s",
                    flush=True,
                )
                repair_messages = messages + [
                    {"role": "assistant", "content": content[:20000]},
                    {
                        "role": "user",
                        "content": (
                            "Output trước không đúng JSON schema. Hãy làm lại từ RAW và chỉ "
                            "trả một JSON object có entities:list theo OUTPUT CONTRACT."
                        ),
                    },
                ]
                last_error = parse_error
                continue
            return parsed, content, _usage_dict(response)
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            print(
                f"\n    [retry {attempt}/{max_retries}] {type(error).__name__} sau "
                f"{time.monotonic() - attempt_started:.1f}s",
                flush=True,
            )
            time.sleep(min(8, 2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def preflight(client: Any, deployment: str, timeout_seconds: float = 30.0) -> float:
    """Một request rất ngắn để tách lỗi kết nối khỏi latency của annotation."""
    started = time.monotonic()
    probe_client = client.with_options(timeout=timeout_seconds, max_retries=0)
    response = _chat_create(probe_client, {
        "model": deployment,
        "messages": [
            {
                "role": "user",
                "content": 'Chỉ trả JSON object: {"ok":true}',
            }
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 64,
    })
    if not getattr(response, "choices", None):
        raise RuntimeError("Azure preflight không có choices")
    return time.monotonic() - started


def cached_call(
    client: Any,
    deployment: str,
    model_label: str,
    messages: list[dict[str, str]],
    variant: str,
    file_id: int,
    run_dir: Path,
    temperature: float | None,
    reasoning_effort: str | None,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    request = {
        "deployment": deployment,
        "model_label": model_label,
        "messages": messages,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    fingerprint = sha256_bytes(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    cache_path = (
        run_dir
        / "cache"
        / variant
        / f"{file_id}_{fingerprint[:16]}.json"
    )
    if cache_path.is_file():
        cached = load_json(cache_path)
        return cached["parsed"], cached.get("usage", {}), True

    parsed, raw_content, usage = call_llm_json(
        client,
        deployment,
        messages,
        temperature,
        reasoning_effort,
        max_retries,
    )
    dump_json(cache_path, {
        "request_sha256": fingerprint,
        "variant": variant,
        "file_id": file_id,
        "deployment": deployment,
        "model_label": model_label,
        "usage": usage,
        "parsed": parsed,
        "raw_content": raw_content,
    })
    return parsed, usage, False


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, int) and "token" in key:
                out[key] += value
    return dict(out)


def variant_stats(labels: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    entities = [entity for rows in labels.values() for entity in rows]
    return {
        "files": len(labels),
        "entities": len(entities),
        "by_type": dict(Counter(entity["type"] for entity in entities)),
        "with_assertion": sum(bool(entity.get("assertions")) for entity in entities),
        "with_candidates": sum(bool(entity.get("candidates")) for entity in entities),
    }


def write_variant_diff(
    path: Path,
    selected: list[int],
    reference: dict[str, list[dict[str, Any]]],
    labels: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for file_id in selected:
        operations = diff_file(reference[str(file_id)], labels[str(file_id)])
        if operations:
            by_file[str(file_id)] = operations
    summary = summarize(by_file)
    payload = {"summary": summary, "files_changed": len(by_file), "by_file": by_file}
    dump_json(path, payload)
    return payload


def write_report(
    run_dir: Path,
    benchmark: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    selected = benchmark["selected_files"]
    reasons = benchmark["file_reasons"]
    lines = [
        "# Part 3 rule/prompt evaluation — run report",
        "",
        "Mục tiêu: kiểm định chất lượng nghiệp vụ và cách prompt khai thác nghiệp vụ, "
        "không tối ưu trực tiếp V66ab.",
        "",
        "## Tập 10 file",
        "",
        "| File | Lý do |",
        "|---:|---|",
    ]
    for file_id in selected:
        lines.append(f"| {file_id} | {reasons[str(file_id)]} |")
    lines.extend([
        "",
        "## ZIP cần nộp",
        "",
        "Mỗi ZIP có đủ `labels/1.json`…`labels/100.json`; chỉ 10 file trên có nhãn, "
        "90 file còn lại là `[]`. Vì vậy điểm có trần xấp xỉ 10 và chỉ so sánh với "
        "các ZIP trong run này, không so trực tiếp với 50.4346 của full V66ab.",
        "",
        "| Thứ tự | Variant | ZIP | Entity | Candidate | Assertion | Gate |",
        "|---:|---|---|---:|---:|---:|---|",
    ])
    order = ["reference_v66ab", *VARIANTS]
    for index, variant in enumerate(order):
        row = manifest["outputs"].get(variant, {})
        stats = row.get("stats", {})
        lines.append(
            f"| {index} | `{variant}` | `{row.get('zip', 'CHƯA TẠO')}` | "
            f"{stats.get('entities', '-')} | {stats.get('with_candidates', '-')} | "
            f"{stats.get('with_assertion', '-')} | "
            f"{'PASS' if row.get('gate_pass') else 'FAIL/CHƯA CHẠY'} |"
        )
    lines.extend([
        "",
        "Ý nghĩa so sánh:",
        "",
        "- `official_only → canonical_dump`: giá trị thông tin của nghiệp vụ tổng hợp.",
        "- `canonical_dump → compiled`: giá trị của việc biên dịch nghiệp vụ thành workflow.",
        "- `compiled → compiled_verified`: giá trị của lượt critic độc lập.",
        "- `compiled_verified → reference_v66ab`: khoảng cách còn lại với artifact tốt nhất.",
        "",
        "Khi nộp, ghi cả `score`, `WER`, `J_assertion`, `J_candidates` vào "
        "`scores_template.csv`. Candidate là stage riêng; khi đánh giá prompt NER phải đọc "
        "WER/J_assertion, không chỉ nhìn tổng điểm.",
        "",
        "## Reproduce/resume",
        "",
        "```bash",
        "python -m annotation.part3_eval.run",
        "```",
        "",
        "Chạy lại cùng lệnh sẽ dùng cache theo hash request và chỉ gọi lại phần còn thiếu.",
    ])
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scores_template(run_dir: Path, outputs: dict[str, Any]) -> None:
    path = run_dir / "scores_template.csv"
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("variant"):
                    existing[row["variant"]] = row
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "variant",
            "zip",
            "score",
            "WER",
            "J_assertion",
            "J_candidates",
            "notes",
        ])
        for variant in ["reference_v66ab", *VARIANTS]:
            previous = existing.get(variant, {})
            writer.writerow([
                variant,
                outputs.get(variant, {}).get("zip", ""),
                previous.get("score", ""),
                previous.get("WER", ""),
                previous.get("J_assertion", ""),
                previous.get("J_candidates", ""),
                previous.get("notes", ""),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark nghiệp vụ + prompt LLM trên 10 file Part 3"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--variants",
        default="all",
        help="all hoặc danh sách cách nhau bởi dấu phẩy",
    )
    parser.add_argument("--dry-run", action="store_true", help="không gọi LLM")
    parser.add_argument(
        "--proxy-url",
        default=None,
        help=(
            "chỉ set proxy khi truyền rõ URL; mặc định giữ nguyên môi trường và gọi "
            "Azure trực tiếp giống annotation/llm_annotate.py"
        ),
    )
    parser.add_argument("--deployment", default=None, help="override Azure deployment")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
        help=(
            "mặc định none để tránh ngưỡng chờ ~30 giây của Azure/network; "
            "tự fallback nếu deployment không hỗ trợ"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="tổng số attempt/request, mặc định 2 (SDK retry nội bộ đã tắt)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="timeout cứng cho mỗi request annotation, mặc định 300 giây",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="bỏ request kiểm tra Azure 30 giây trước batch",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    benchmark = load_json(BENCHMARK_PATH)
    manifest_source, artifact_path, rulebook_path = verify_inputs(benchmark)
    selected: list[int] = benchmark["selected_files"]
    selected_set = set(selected)
    notes = {
        str(file_id): (NOTES_DIR / f"{file_id}.txt").read_text(encoding="utf-8")
        for file_id in selected
    }
    reference = load_reference(artifact_path)
    if set(reference) != {str(file_id) for file_id in range(1, 101)}:
        raise RuntimeError("Artifact V66ab không có đúng 100 file")

    run_dir.mkdir(parents=True, exist_ok=True)
    canonical_text = CANONICAL_PATH.read_text(encoding="utf-8")
    prompt_dir = run_dir / "prompt_templates"
    for name, content in templates(canonical_text).items():
        path = prompt_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")

    selection = selection_stats(selected, reference, artifact_path)
    dump_json(run_dir / "selection_report.json", selection)

    outputs: dict[str, Any] = {}
    reference_zip = run_dir / "submissions" / ZIP_NAMES["reference_v66ab"]
    write_partial_submission(reference_zip, selected_set, reference)
    reference_errors = validate_submission(reference_zip, selected_set, notes)
    if reference_errors:
        raise RuntimeError("Reference ZIP lỗi:\n" + "\n".join(reference_errors))
    reference_selected = {str(file_id): reference[str(file_id)] for file_id in selected}
    outputs["reference_v66ab"] = {
        "zip": str(reference_zip.relative_to(run_dir)),
        "sha256": sha256_path(reference_zip),
        "gate_pass": True,
        "stats": variant_stats(reference_selected),
    }
    print(
        f"[reference] {reference_zip.name}: "
        f"{outputs['reference_v66ab']['stats']['entities']} entities · gate PASS"
    )

    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark["benchmark_id"],
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_files": selected,
        "fingerprints": {
            "benchmark": sha256_path(BENCHMARK_PATH),
            "canonical": sha256_path(CANONICAL_PATH),
            "rulebook": sha256_path(rulebook_path),
            "artifact": sha256_path(artifact_path),
        },
        "source_manifest": manifest_source,
        "outputs": outputs,
    }

    if args.dry_run:
        dump_json(run_dir / "manifest.json", base_manifest)
        write_scores_template(run_dir, outputs)
        write_report(run_dir, benchmark, base_manifest)
        print(f"[dry-run] OK -> {run_dir}")
        return 0

    if args.variants == "all":
        requested = list(VARIANTS)
    else:
        requested = [item.strip() for item in args.variants.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(VARIANTS))
        if unknown:
            raise ValueError(f"Variant không hỗ trợ: {unknown}")
        if "compiled_verified" in requested and "compiled" not in requested:
            requested.insert(0, "compiled")

    # Azure OpenAI được gọi trực tiếp như annotation/llm_annotate.py. Proxy Viettel
    # chỉ dành cho tải dataset/model và không được tự bật ở đây.
    enable_proxy(args.proxy_url)
    sys.path.insert(0, str(ROOT))
    from data_gen.config import get_azure_client

    client, env_deployment, model_label = get_azure_client()
    deployment = args.deployment or env_deployment
    # SDK OpenAI mặc định có timeout/retry dài. Tắt retry nội bộ để chỉ còn một
    # lớp retry có log/counter rõ ràng trong call_llm_json.
    client = client.with_options(timeout=args.request_timeout, max_retries=0)
    print(f"[config] deployment={deployment} model_label={model_label}")
    print(f"[config] selected={selected} variants={requested}")
    print(
        f"[config] request_timeout={args.request_timeout:.0f}s "
        f"outer_attempts={args.max_retries} reasoning_effort={args.reasoning_effort}"
    )
    if not args.skip_preflight:
        print("[preflight] Azure connection: ", end="", flush=True)
        try:
            elapsed = preflight(client, deployment)
        except Exception as error:
            print(f"FAIL sau tối đa 30s — {type(error).__name__}: {str(error)[:300]}")
            print(
                "[preflight] Không chạy batch. Kiểm tra kết nối/API bằng "
                "annotation.llm_annotate.py hoặc cấu hình .env."
            )
            return 3
        print(f"PASS ({elapsed:.1f}s)")

    valid_icd, valid_rxnorm = load_valid_codes()
    print(
        f"[codes] ICD={len(valid_icd):,} · RxNorm={len(valid_rxnorm):,} "
        "(candidate ngoài DB sẽ bị bỏ và ghi audit)"
    )

    variant_raw: dict[str, dict[str, list[dict[str, Any]]]] = {}
    incomplete: dict[str, list[int]] = defaultdict(list)
    for variant in requested:
        print(f"\n[variant] {variant}")
        labels_by_file: dict[str, list[dict[str, Any]]] = {}
        raw_by_file: dict[str, list[dict[str, Any]]] = {}
        audits: dict[str, Any] = {}
        usage_rows: list[dict[str, Any]] = []

        for sequence, file_id in enumerate(selected, 1):
            raw = notes[str(file_id)]
            draft = None
            if variant == "compiled_verified":
                draft = variant_raw.get("compiled", {}).get(str(file_id))
                if draft is None:
                    incomplete[variant].append(file_id)
                    print(
                        f"  [{sequence:02d}/10] file {file_id}: SKIP "
                        "(thiếu compiled draft; chạy lại sẽ resume)"
                    )
                    continue
            messages = build_messages(variant, raw, canonical_text, draft=draft)
            started = time.monotonic()
            print(
                f"  [{sequence:02d}/10] file {file_id} "
                f"(timeout {args.request_timeout:.0f}s): ",
                end="",
                flush=True,
            )
            try:
                parsed, usage, from_cache = cached_call(
                    client=client,
                    deployment=deployment,
                    model_label=model_label,
                    messages=messages,
                    variant=variant,
                    file_id=file_id,
                    run_dir=run_dir,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                    max_retries=args.max_retries,
                )
            except Exception as error:
                incomplete[variant].append(file_id)
                elapsed = time.monotonic() - started
                print(
                    f"ERROR sau {elapsed:.1f}s {type(error).__name__}: "
                    f"{str(error)[:300]}"
                )
                continue

            predictions = parsed["entities"]
            labels, audit, aligned_metadata = align_predictions(
                raw,
                predictions,
                valid_icd,
                valid_rxnorm,
            )
            errors = validate_labels(raw, labels)
            audit["validation_errors"] = errors
            audit["llm_self_check"] = parsed.get("self_check")
            audit["usage"] = usage
            audit["cache_hit"] = from_cache
            audit["aligned_metadata"] = aligned_metadata
            labels_by_file[str(file_id)] = labels
            raw_by_file[str(file_id)] = predictions
            audits[str(file_id)] = audit
            usage_rows.append(usage)
            elapsed = time.monotonic() - started
            print(
                f"proposed={audit['proposed']} kept={audit['kept']} "
                f"align={audit['alignment_rate']:.0%} "
                f"{'cache' if from_cache else 'API'} {elapsed:.1f}s"
            )

        variant_raw[variant] = raw_by_file
        dump_json(run_dir / "audits" / f"{variant}.json", audits)
        for file_id, labels in labels_by_file.items():
            dump_json(run_dir / "labels" / variant / f"{file_id}.json", labels)

        complete = not incomplete[variant] and set(labels_by_file) == {
            str(file_id) for file_id in selected
        }
        nonempty = complete and all(labels_by_file[str(file_id)] for file_id in selected)
        validation_errors = [
            f"file {file_id}: {error}"
            for file_id in selected
            for error in audits.get(str(file_id), {}).get("validation_errors", [])
        ]
        gate_pass = bool(complete and nonempty and not validation_errors)
        row: dict[str, Any] = {
            "gate_pass": gate_pass,
            "complete": complete,
            "nonempty_selected_files": nonempty,
            "incomplete_files": incomplete[variant],
            "validation_errors": validation_errors,
            "stats": variant_stats(labels_by_file),
            "usage": _sum_usage(usage_rows),
        }
        if complete:
            diff_payload = write_variant_diff(
                run_dir / "diffs" / f"{variant}_vs_reference.json",
                selected,
                reference,
                labels_by_file,
            )
            row["diff_vs_reference"] = diff_payload["summary"]
        if gate_pass:
            zip_path = run_dir / "submissions" / ZIP_NAMES[variant]
            write_partial_submission(zip_path, selected_set, labels_by_file)
            zip_errors = validate_submission(zip_path, selected_set, notes)
            if zip_errors:
                row["gate_pass"] = False
                row["validation_errors"].extend(zip_errors)
            else:
                row["zip"] = str(zip_path.relative_to(run_dir))
                row["sha256"] = sha256_path(zip_path)
                print(
                    f"[zip] {zip_path.name}: {row['stats']['entities']} entities · gate PASS"
                )
        if not row["gate_pass"]:
            print(
                f"[gate] {variant}: FAIL — không tạo ZIP hoặc ZIP không an toàn; "
                f"chạy lại cùng lệnh để resume"
            )
        outputs[variant] = row

    final_manifest = {
        **base_manifest,
        "model": {
            "provider": "Azure OpenAI",
            "deployment": deployment,
            "model_label": model_label,
            "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort,
            "request_timeout_seconds": args.request_timeout,
            "sdk_max_retries": 0,
            "outer_max_attempts": args.max_retries,
        },
        "candidate_validation": {
            "icd_source": str(ICD_PATH.relative_to(ROOT)),
            "rxnorm_source": str(RXNORM_PATH.relative_to(ROOT)),
            "valid_icd_count": len(valid_icd),
            "valid_rxnorm_count": len(valid_rxnorm),
        },
        "outputs": outputs,
    }
    dump_json(run_dir / "manifest.json", final_manifest)
    write_scores_template(run_dir, outputs)
    write_report(run_dir, benchmark, final_manifest)
    all_pass = all(outputs.get(variant, {}).get("gate_pass") for variant in requested)
    print(f"\n[done] report: {run_dir / 'REPORT.md'}")
    print(f"[done] submissions: {run_dir / 'submissions'}")
    if not all_pass:
        print("[done] Có variant chưa PASS; chạy lại cùng lệnh để resume cache.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
