"""Crawl and label the Benh Vien 108 public Q&A external challenge set.

The output is deliberately in the same ``notes/<id>.txt`` + ``labels/<id>.json`` format as the
annotation UI.  This corpus is an evaluation/review artifact only; no training builder imports
it automatically.

Examples from the repository root::

    python -m annotation.external_challenge.benhvien108 crawl
    python -m annotation.external_challenge.benhvien108 annotate --limit 10
    python -m annotation.app --data_dir annotation/data/external_benhvien108_qa_v1 --port 5000

The crawler obeys a small delay between requests and keeps HTML responses in a cache so an
interrupted run can resume without repeatedly hitting the site.  HTTP(S)_PROXY is intentionally
read from the environment; the caller may set the proxy supplied for the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from annotation.part3_eval.align import align_predictions, validate_labels
from annotation.part3_eval.prompts import build_messages, templates
from annotation.part3_eval.run import (
    CANONICAL_PATH,
    cached_call,
    load_valid_codes,
    sha256_path,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "annotation/data/external_benhvien108_qa_v1"
DEFAULT_RUN_DIR = DEFAULT_DATA_DIR / "part3_eval_run"
BASE_URL = "https://www.benhvien108.vn"
LIST_PATH = "/hoi-dap.htm"
SOURCE_URL = f"{BASE_URL}{LIST_PATH}"
DEFAULT_TOTAL_PAGES = 16
USER_AGENT = "Viettel-AI-Race external challenge crawler/1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalise_space(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def _clean_html_text(value: str) -> str:
    """Keep meaningful line breaks from lists/paragraphs while removing layout whitespace."""
    output: list[str] = []
    blank_pending = False
    for raw_line in value.replace("\xa0", " ").splitlines():
        line = _normalise_space(raw_line)
        if not line:
            if output:
                blank_pending = True
            continue
        if blank_pending and output and output[-1] != "":
            output.append("")
        blank_pending = False
        output.append(line)
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output)


def _request(session: requests.Session, url: str, cache_dir: Path, refresh: bool) -> str:
    cache_path = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.html"
    if cache_path.is_file() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    response = session.get(url, timeout=(20, 120))
    response.raise_for_status()
    # The page declares UTF-8; requests can otherwise guess ISO-8859-1 from old headers.
    response.encoding = "utf-8"
    html = response.text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


def _page_number(html: str) -> int | None:
    match = re.search(r"var\s+totalPage\s*=\s*(\d+)", html, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _list_items(html: str, page: int) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in soup.select("article.item"):
        link = None
        for candidate in article.select('a[href*="/chi-tiet-hoi-dap.htm?id="]'):
            text = _normalise_space(candidate.get_text(" ", strip=True))
            if text and text.lower() not in {"xem chi tiết", "chi tiết"}:
                link = candidate
                break
        if link is None:
            continue
        detail_url = urljoin(BASE_URL, link.get("href", ""))
        if detail_url in seen:
            continue
        seen.add(detail_url)
        rows.append({
            "page": page,
            "order_on_page": len(rows) + 1,
            "detail_url": detail_url,
            "question_preview": _normalise_space(link.get_text(" ", strip=True)),
        })
    return rows


def _detail_record(
    html: str, item: dict[str, Any], record_number: int,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    article = None
    for candidate in soup.select("article.item"):
        if candidate.find(string=re.compile(r"Hỏi\s*:", re.IGNORECASE)):
            article = candidate
            break
    if article is None:
        raise RuntimeError(f"Không tìm thấy article hỏi đáp: {item['detail_url']}")

    question_link = None
    for candidate in article.select("h3 a"):
        text = _normalise_space(candidate.get_text(" ", strip=True))
        if text and text.lower() not in {"xem chi tiết", "chi tiết"}:
            question_link = candidate
            break
    question = _normalise_space(
        question_link.get_text(" ", strip=True) if question_link else ""
    )
    answer_heading = None
    for heading in article.find_all("h3"):
        if re.match(r"\s*Trả lời\s*:?", heading.get_text(" ", strip=True), re.I):
            answer_heading = heading
            break
    answer = ""
    if answer_heading is not None:
        title_box = answer_heading.find_parent("div", class_="title")
        answer_box = title_box.find_next_sibling() if title_box else None
        if answer_box is not None:
            answer = _clean_html_text(answer_box.get_text("\n", strip=True))
    if not question:
        raise RuntimeError(f"Không đọc được câu hỏi: {item['detail_url']}")

    raw = f"HỎI:\n{question}\n\nTRẢ LỜI:\n{answer}".rstrip() + "\n"
    return {
        "number": record_number,
        "id": str(record_number),
        "text": raw,
        "page": item["page"],
        "order_on_page": item["order_on_page"],
        "detail_url": item["detail_url"],
        "list_url": f"{SOURCE_URL}?p={item['page']}",
        "question_preview": item["question_preview"],
        "title": question[:160],
        "source_id": item["detail_url"].split("id=", 1)[-1],
    }


def crawl(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    cache_dir = data_dir / "crawl_cache"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.8"})

    first_url = f"{SOURCE_URL}?p={args.start_page}"
    first_html = _request(session, first_url, cache_dir, args.refresh)
    discovered_pages = _page_number(first_html) or DEFAULT_TOTAL_PAGES
    end_page = min(args.end_page or discovered_pages, discovered_pages)
    if args.start_page < 1 or end_page < args.start_page:
        raise ValueError(f"Khoảng trang không hợp lệ: {args.start_page}..{end_page}")

    items: list[dict[str, Any]] = []
    for page in range(args.start_page, end_page + 1):
        url = f"{SOURCE_URL}?p={page}"
        html = first_html if page == args.start_page else _request(
            session, url, cache_dir, args.refresh
        )
        page_items = _list_items(html, page)
        items.extend(page_items)
        print(f"[crawl list] page={page}/{end_page} items={len(page_items)}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    unique: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in items:
        if item["detail_url"] not in seen_urls:
            seen_urls.add(item["detail_url"])
            unique.append(item)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for number, item in enumerate(unique, 1):
        try:
            html = _request(session, item["detail_url"], cache_dir, args.refresh)
            records.append(_detail_record(html, item, number))
            print(f"[crawl detail] {number}/{len(unique)} OK", flush=True)
        except Exception as error:
            errors.append({"url": item["detail_url"], "error": str(error)})
            print(f"[crawl detail] {number}/{len(unique)} ERROR: {error}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    notes_dir = data_dir / "notes"
    labels_dir = data_dir / "labels"
    notes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for record in records:
        number = record["number"]
        note_path = notes_dir / f"{number}.txt"
        label_path = labels_dir / f"{number}.json"
        note_path.write_text(record["text"], encoding="utf-8")
        if not label_path.exists() or args.refresh:
            _json_dump(label_path, [])
        metadata.append({
            "file_id": str(number),
            "source": "benhvien108_public_qa",
            "source_url": record["detail_url"],
            "list_url": record["list_url"],
            "source_id": record["source_id"],
            "page": record["page"],
            "order_on_page": record["order_on_page"],
            "title": record["title"],
            "characters": len(record["text"]),
            "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
            "crawled_at": _now(),
        })
    _jsonl_dump(data_dir / "metadata.jsonl", metadata)
    _json_dump(data_dir / "_reviewed.json", [])
    manifest = {
        "schema_version": 1,
        "stage": "external_challenge_crawled_pending_human_review",
        "source": "Bệnh viện Trung ương Quân đội 108 — Tư vấn/Hỏi đáp",
        "source_url": SOURCE_URL,
        "crawl_started_page": args.start_page,
        "crawl_end_page": end_page,
        "discovered_total_pages": discovered_pages,
        "list_items": len(items),
        "unique_items": len(unique),
        "records_written": len(records),
        "errors": errors,
        "cache_dir": str(cache_dir.relative_to(data_dir)),
        "created_at": _now(),
        "labeling": {
            "method": "annotation.part3_eval compiled prompt + deterministic RAW alignment",
            "human_review_required": True,
            "training_use": False,
        },
    }
    _json_dump(data_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if records and not errors else 2


def _read_metadata(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = data_dir / "metadata.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["file_id"])] = row
    return rows


def annotate(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else data_dir / "part3_eval_run"
    notes_dir = data_dir / "notes"
    labels_dir = data_dir / "labels"
    if not notes_dir.is_dir() or not (data_dir / "metadata.jsonl").is_file():
        raise FileNotFoundError(f"Chưa crawl dataset: {data_dir}")
    metadata = _read_metadata(data_dir)
    file_ids = sorted(
        (path.stem for path in notes_dir.glob("*.txt")),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    if args.start > 0:
        file_ids = file_ids[args.start:]
    if args.limit is not None:
        file_ids = file_ids[:args.limit]
    if not file_ids:
        raise RuntimeError("Không có note nào để annotate")

    canonical_text = CANONICAL_PATH.read_text(encoding="utf-8")
    prompt_dir = run_dir / "prompt_templates"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name, content in templates(canonical_text).items():
        (prompt_dir / name).write_text(content.rstrip() + "\n", encoding="utf-8")

    # Azure client configuration is shared with annotation.part3_eval.  Proxy for the website
    # is inherited from the environment; do not silently invent a proxy for the API.
    from data_gen.config import get_azure_client

    client, env_deployment, model_label = get_azure_client()
    deployment = args.deployment or env_deployment
    client = client.with_options(timeout=args.request_timeout, max_retries=0)
    valid_icd, valid_rxnorm = load_valid_codes()
    print(
        f"[annotate] files={len(file_ids)} deployment={deployment} "
        f"variant=compiled timeout={args.request_timeout:.0f}s",
        flush=True,
    )

    stats = Counter()
    failures: list[dict[str, Any]] = []
    canonical_hash = sha256_path(CANONICAL_PATH)
    for sequence, file_id in enumerate(file_ids, 1):
        raw = (notes_dir / f"{file_id}.txt").read_text(encoding="utf-8")
        cache_key = f"external_{file_id}"
        started = time.monotonic()
        print(f"[annotate] {sequence}/{len(file_ids)} file={file_id} ...", flush=True)
        try:
            messages = build_messages("compiled", raw, canonical_text)
            parsed, usage, from_cache = cached_call(
                client=client,
                deployment=deployment,
                model_label=model_label,
                messages=messages,
                variant="compiled",
                file_id=cache_key,
                run_dir=run_dir,
                temperature=0.0,
                reasoning_effort=args.reasoning_effort,
                max_retries=args.max_retries,
            )
            labels, audit, aligned = align_predictions(
                raw, parsed["entities"], valid_icd, valid_rxnorm
            )
            errors = validate_labels(raw, labels)
            audit.update({
                "file_id": file_id,
                "source_url": metadata.get(file_id, {}).get("source_url"),
                "llm_self_check": parsed.get("self_check"),
                "usage": usage,
                "cache_hit": from_cache,
                "aligned_metadata": aligned,
                "validation_errors": errors,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            })
            _json_dump(run_dir / "audits" / f"{file_id}.json", audit)
            _json_dump(labels_dir / f"{file_id}.json", labels)
            metadata_row = metadata.setdefault(file_id, {"file_id": file_id})
            metadata_row.update({
                "draft_entities": len(labels),
                "draft_proposed": audit["proposed"],
                "alignment_rate": audit["alignment_rate"],
                "unaligned_count": len(audit.get("unaligned", [])),
                "draft_validation_errors": len(audit.get("validation_errors", [])),
                "review_priority": (
                    "high_empty_draft" if not labels else
                    "high_alignment_attention" if audit["alignment_rate"] < 1.0 else
                    "normal_draft_review"
                ),
                "strata": (
                    ["empty_llm_draft"] if not labels else
                    ["alignment_attention"] if audit["alignment_rate"] < 1.0 else
                    ["compiled_draft"]
                ),
            })
            stats["files_ok"] += 1
            stats["entities"] += len(labels)
            stats["cache_hits"] += int(from_cache)
            for row in labels:
                stats[f"type:{row['type']}"] += 1
            print(
                f"  kept={len(labels)} align={audit['alignment_rate']:.0%} "
                f"{'cache' if from_cache else 'API'} {audit['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({"file_id": file_id, "error": repr(error)})
            stats["files_failed"] += 1
            metadata_row = metadata.setdefault(file_id, {"file_id": file_id})
            metadata_row.update({
                "review_priority": "critical_annotation_failed",
                "strata": ["annotation_failed"],
                "annotation_error": repr(error),
            })
            print(f"  ERROR: {error}", flush=True)

    _jsonl_dump(
        data_dir / "metadata.jsonl",
        [metadata[file_id] for file_id in sorted(
            metadata, key=lambda value: int(value) if value.isdigit() else value
        )],
    )
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["stage"] = (
        "external_challenge_llm_draft_ready_pending_human_review"
        if not failures else "external_challenge_llm_draft_partial_pending_human_review"
    )
    manifest["labeling"].update({
        "method": "annotation.part3_eval compiled prompt + deterministic RAW alignment",
        "variant": "compiled",
        "deployment": deployment,
        "model_label": model_label,
        "canonical_sha256": canonical_hash,
        "run_dir": str(run_dir),
        "files_requested": len(file_ids),
        "failures": failures,
        "stats": dict(stats),
        "human_review_required": True,
        "training_use": False,
    })
    _json_dump(data_dir / "manifest.json", manifest)
    _json_dump(run_dir / "manifest.json", {
        "source_manifest": str((data_dir / "manifest.json").relative_to(run_dir.parent)),
        "created_at": _now(),
        "files_requested": file_ids,
        "failures": failures,
        "stats": dict(stats),
    })
    return 0 if not failures else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    crawl_parser = sub.add_parser("crawl", help="crawl list/detail pages")
    crawl_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    crawl_parser.add_argument("--start-page", type=int, default=1)
    crawl_parser.add_argument("--end-page", type=int)
    crawl_parser.add_argument("--delay", type=float, default=0.4)
    crawl_parser.add_argument("--refresh", action="store_true")

    annotate_parser = sub.add_parser("annotate", help="LLM draft via part3_eval compiled prompt")
    annotate_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    annotate_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    annotate_parser.add_argument("--start", type=int, default=0)
    annotate_parser.add_argument("--limit", type=int)
    annotate_parser.add_argument("--deployment")
    annotate_parser.add_argument("--request-timeout", type=float, default=300.0)
    annotate_parser.add_argument("--max-retries", type=int, default=2)
    annotate_parser.add_argument(
        "--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"),
        default="none",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "crawl":
        return crawl(args)
    return annotate(args)


if __name__ == "__main__":
    raise SystemExit(main())
