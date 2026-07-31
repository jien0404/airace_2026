"""Rà soát nhãn part1/gt2 bằng nghiệp vụ, có LLM phân xử và checkpoint.

Luồng: sàng bằng luật (`screen.py`) → LLM phân xử từng nghi ngờ → ghi `findings.jsonl` để
người review quyết định trong `annotation.label_audit.app`.

Nguyên tắc: **chỉ bắt lỗi rất rõ ràng**. LLM được yêu cầu trả `keep` khi còn lăn tăn; nghi ngờ
cơ học (assertion trên type bị cấm, offset lệch) không cần hỏi LLM.

Không sửa gì vào nhãn gốc. Bản sửa nằm ở thư mục riêng, do lệnh `apply` tạo sau khi review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dataset_factory.schema import ASSERTION_TYPES

from .screen import (
    CONTEXT_CHARS, load_labeled_dir, screen_delete_candidates, screen_records,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES = {
    "part1": "annotation/data/best_54.92",
    "gt2": "annotation/data/groundtruth_part2",
}

def _input_signature(prompt: str) -> str:
    """Dấu vân tay của MỌI thứ quyết định đầu vào của LLM: prompt + cấu hình context.

    Trước đây đây là một con số đếm tay, và nó đã hỏng đúng một lần: tôi nới cửa sổ 320→400 và
    thêm heading nhưng quên tăng số, nên phán xử cũ (sinh với context khác hẳn) vẫn được tái dùng
    im lặng. Băm tự động thì không quên được.
    """
    material = "\n".join([
        prompt,
        f"context_chars={CONTEXT_CHARS}",
        f"heading_field={HEADING_IN_PAYLOAD}",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


HEADING_IN_PAYLOAD = True

DELETE_PROMPT = """Bạn soát lại QUYẾT ĐỊNH GÁN của nhãn NER y khoa tiếng Việt. Với mỗi entity đã
được gán, hãy nói nó có thực sự đáng là entity hay không.

Quy ước (CANONICAL §1.1): file được chấm đúng như nó tồn tại, KHÔNG xoá chỉ vì đoạn đó là tư vấn,
giáo dục hay nói về ca khác. Nhưng không phải mọi thuật ngữ y khoa xuất hiện đều là entity.

THƯỜNG BỎ:
- danh từ meta đứng trần: `triệu chứng`, `thuốc`, `xét nghiệm`, `bệnh`, `chẩn đoán`;
- cơ chế bệnh sinh, giải thích sinh học;
- checklist biểu hiện điển hình nêu như kiến thức chung;
- tác dụng phụ / chống chỉ định / nguy cơ chỉ nêu chung, chưa xảy ra với người bệnh;
- câu giả định `có thể gây`, `nếu... thì...`;
- thuốc chỉ nêu làm ví dụ minh hoạ;
- heading, ô thuộc tính, sự kiện hành chính.

VẪN GIỮ, kể cả trong đoạn giáo dục:
- tên bệnh chủ đề và subtype;
- chẩn đoán/finding là nội dung chuyên môn chính;
- tên xét nghiệm, phương pháp điều trị, tên nhóm thuốc;
- kết quả cụ thể gắn với một phép đo/thăm dò.

KHÔNG BAO GIỜ xoá vì các lý do sau — đây là lỗi hay gặp nhất:
- **Vì nó bị phủ định.** `không mô tả ảo tưởng`, `không sốt`, `phủ nhận đau ngực` — finding bị
  phủ định VẪN là entity, chỉ mang thêm `isNegated`. Xoá là mất nhãn đúng.
- **Vì nó thuộc tiền sử hoặc người thân.** Cũng vẫn là entity, chỉ mang `isHistorical`/`isFamily`.
- **Vì span cắt hơi lệch.** `lên vai trái,` cắt xấu là chuyện của ranh giới span, không phải lý do
  xoá cả entity. Gặp span xấu thì trả `keep`.
- **Vì nó là dấu hiệu sinh tồn.** `Mạch: 100 lần/phút`, `HA 130/70 mmHg` — quy ước cho vital
  hiện CHƯA CHỐT (CANONICAL §2.4) và gold được phép giữ nguyên. Luôn trả `keep` cho vital.
- **Vì nó nằm trong đoạn tư vấn/giáo dục.** Chỉ xoá khi nội dung ĐÓ là kiến thức chung; tên bệnh
  chủ đề, xét nghiệm, nhóm thuốc trong đoạn giáo dục vẫn giữ.

Nguyên tắc phán xử — RẤT QUAN TRỌNG:
- Mặc định là `keep`. Chỉ trả `delete` khi đọc câu là thấy RÕ nó thuộc nhóm THƯỜNG BỎ.
- Bạn đang soát nhãn tay của người có kinh nghiệm; phần lớn nhãn là ĐÚNG. Nếu bạn thấy mình
  muốn xoá quá nhiều, gần như chắc chắn bạn đang khắt khe sai.
- Không đề xuất thêm entity, không đổi span, không đổi type, không đổi assertion.
- `muc_dang_o` là heading của mục, dùng để hiểu thể thức phát ngôn; heading một mình KHÔNG đủ
  để xoá — vẫn phải đọc câu.

Trả JSON duy nhất:
{"verdicts": [{"id": "...", "verdict": "delete" | "keep", "reason": "ngắn gọn tiếng Việt"}]}
Mỗi id trong đầu vào phải xuất hiện đúng một lần."""

SYSTEM_PROMPT = """Bạn là người soát nhãn NER y khoa tiếng Việt. Bạn KHÔNG gán nhãn mới; bạn chỉ
phán xử xem một nhãn đang có là SAI RÕ RÀNG hay CHẤP NHẬN ĐƯỢC.

Quy ước đang áp dụng:
- Ba assertion: isNegated (bị phủ định trực tiếp trong cùng mệnh đề), isHistorical (occurrence
  thực sự thuộc thời điểm trước đợt hiện tại), isFamily (thuộc người thân, không phải bệnh nhân).
- TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM LUÔN có assertions rỗng, không ngoại lệ.
- Assertion quyết định theo TỪNG occurrence, dựa vào bằng chứng trong cùng mệnh đề/context gần.
  Không suy từ heading, không lan từ occurrence khác cùng surface.
- `nghi ngờ`, `theo dõi`, `khả năng`, `chưa loại trừ` KHÔNG phải phủ định.
- Một mục "Tiền sử" vẫn có thể chứa triệu chứng hiện tại; không gán isHistorical chỉ vì heading.
- `muc_dang_o` là heading gần nhất phía trước entity, gửi kèm vì nó thường nằm ngoài cửa sổ
  context. Dùng nó để hiểu thể thức phát ngôn của mục, KHÔNG dùng nó một mình để suy assertion.
- Với THUỐC, isHistorical cần rất thận trọng: thuốc đã ngừng/đã hết không mặc nhiên là historical.
- Phủ định tiếng Việt có thể đứng trước hoặc sau concept: "không đau ngực", "sốt: không".

Nguyên tắc phán xử — QUAN TRỌNG:
- Chỉ trả `fix` khi lỗi RÕ RÀNG và mâu thuẫn TRỰC TIẾP với quy ước trên, đọc ngay trong context
  đã cho là thấy.
- Còn lăn tăn, thiếu context, hoặc chỉ là khác biệt phong cách gán nhãn → trả `keep`.
- Không đề xuất thêm hay bớt entity. Chỉ được sửa ba thứ của entity đang xét: `assertions`,
  `type`, và ranh giới span.
- `assertions` đề xuất là danh sách con của [isNegated, isFamily, isHistorical], có thể rỗng.

Về TYPE — chỉ đổi khi chắc chắn:
- Type quyết định theo vai trò của cụm trong CHÍNH câu này, không theo từ điển. Cùng một chữ có
  thể là type khác nhau ở hai chỗ khác nhau, và điều đó hợp lệ.
- Chỉ trả `fix` cho type khi đọc context thấy rõ vai trò mâu thuẫn với nhãn: ví dụ một trị số kèm
  đơn vị đứng sau tên xét nghiệm mà bị gán TRIỆU_CHỨNG.
- Đa số corpus chỉ là gợi ý. Nếu context ở đây thật sự khác, trả `keep` dù thiểu số.

Về SPAN — chỉ cắt, không nới:
- Từ phủ định (`không`, `chưa`, `phủ nhận`) là BẰNG CHỨNG cho isNegated, không phải phần của
  concept. `"Không đau đầu"` phải là span `"đau đầu"` + `isNegated`.
- Chỉ đề xuất span khi nó là khúc con LIỀN MẠCH của span hiện tại. Không mở rộng span.
- KẾT_QUẢ_XÉT_NGHIỆM dạng tường thuật (`"không ghi nhận bất thường"`) là ngoại lệ: cả câu chính
  là kết quả, giữ nguyên span và trả `keep`.
- Nếu đề xuất đổi span, **BẮT BUỘC** trả trường `span` chứa ĐÚNG chuỗi con mới. Viết lý do mà
  không trả `span` thì hệ thống không sửa được gì cả.

Ví dụ đúng cho `negation_inside_span`:
  vào:  type=TRIỆU_CHỨNG, surface="không chóng mặt", assertions=[]
  ra:   {"id": "...", "verdict": "fix", "assertions": ["isNegated"], "span": "chóng mặt",
         "reason": "Cắt từ phủ định khỏi concept, chuyển thành isNegated."}

Ví dụ đúng cho `type_minority_vs_corpus`:
  vào:  type=CHẨN_ĐOÁN, surface="khó thở", context "Bệnh nhân khó thở khi gắng sức"
  ra:   {"id": "...", "verdict": "fix", "assertions": [], "type": "TRIỆU_CHỨNG",
         "reason": "Ở câu này là than phiền của bệnh nhân, không phải kết luận bệnh."}

Trả JSON duy nhất:
{"verdicts": [{"id": "...", "verdict": "fix" | "keep", "assertions": ["..."],
  "type": "giữ nguyên nếu không đổi", "span": "chỉ đưa khi cắt span", "reason": "ngắn gọn tiếng Việt"}]}
Mỗi id trong đầu vào phải xuất hiện đúng một lần."""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def finding_id(finding: dict[str, Any]) -> str:
    return f"{finding['file']}#{finding['entity_index']}"


def _payload(finding: dict[str, Any]) -> dict[str, Any]:
    context = finding["context"]
    return {
        "id": finding_id(finding),
        "type": finding["entity"]["type"],
        "surface": finding["entity"]["text"],
        "assertions_hien_tai": finding["current"]["assertions"],
        "nghi_ngo": finding["kind"],
        "luat": finding["rule"],
        "ly_do_sang_loc": finding["reason"],
        "muc_dang_o": context.get("heading") or "(không có heading phía trước)",
        "de_xuat_cua_tang_sang": {
            key: value for key, value in (finding.get("proposed") or {}).items()
            if key in ("assertions", "type", "text")
        },
        "context": (
            context["before"] + "⟦" + context["surface"] + "⟧" + context["after"]
        ),
    }


def _resolve_change(
    finding: dict[str, Any], decision: dict[str, Any],
) -> dict[str, Any] | None:
    """Chuyển phán xử của LLM thành thay đổi cụ thể, hoặc None nếu rốt cuộc không đổi gì.

    Span mới phải là khúc con LIỀN MẠCH của span cũ — nếu không, bỏ phần span và chỉ giữ thay đổi
    assertion/type. Ràng buộc này giữ cho offset luôn tính lại được từ span gốc, và chặn việc LLM
    lặng lẽ nới rộng entity.
    """
    entity = finding["entity"]
    current_assertions = list(finding["current"]["assertions"])
    current_type = finding["current"]["type"]
    proposed_assertions = list(decision.get("assertions") or [])
    proposed_type = decision.get("type") or current_type
    change: dict[str, Any] = {}

    span = (decision.get("span") or "").strip()
    if span and span != entity["text"]:
        offset = entity["text"].find(span)
        if offset >= 0 and span:
            start = entity["position"][0] + offset
            change["proposed_text"] = span
            change["proposed_position"] = [start, start + len(span)]
    if proposed_type != current_type:
        change["proposed_type"] = proposed_type
    if proposed_type not in ASSERTION_TYPES:
        proposed_assertions = []
    if proposed_assertions != current_assertions:
        change["proposed_assertions"] = proposed_assertions
    if not change:
        return None
    change.setdefault("proposed_assertions", current_assertions)
    return change


def screen_delete_all(sources: list[str]) -> list[dict[str, Any]]:
    findings = []
    for source in sources:
        for record in load_labeled_dir(REPO_ROOT / SOURCES[source], source):
            for row in screen_delete_candidates(record):
                row["source"] = source
                findings.append(row)
    return findings


def screen_all(sources: list[str]) -> list[dict[str, Any]]:
    # Sàng CẢ HAI nguồn trong một lượt: `type_minority_vs_corpus` cần thống kê type đa số trên
    # toàn bộ nhãn tay, tách từng nguồn thì gt2 và part1 không soi được cho nhau.
    records = []
    for source in sources:
        records.extend(load_labeled_dir(REPO_ROOT / SOURCES[source], source))
    findings = []
    for row in screen_records(records):
        row["source"] = row["file"].split(":", 1)[0]
        findings.append(row)
    return findings


def adjudicate(
    findings: list[dict[str, Any]],
    out_dir: Path,
    batch_size: int,
    concurrency: int,
    timeout: float,
    attempts: int,
    mode: str = "fix",
) -> dict[str, Any]:
    from data_gen.config import get_azure_client

    client, deployment, model = get_azure_client()
    client = client.with_options(timeout=timeout, max_retries=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / ("verdicts_delete.jsonl" if mode == "delete" else "verdicts.jsonl")
    signature = _input_signature(DELETE_PROMPT if mode == "delete" else SYSTEM_PROMPT)
    journal = _read_jsonl(journal_path)
    verdicts: dict[str, dict[str, Any]] = {
        row["id"]: row for row in journal if row.get("input_signature") == signature
    }
    stale = len(journal) - len(verdicts)
    if stale:
        print(
            f"[plan] bỏ qua {stale} phán xử sinh với prompt/context khác "
            f"(chữ ký hiện tại {signature})",
            flush=True,
        )
    pending = [
        finding for finding in findings
        if not finding["mechanical"] and finding_id(finding) not in verdicts
    ]
    print(
        f"[plan] {len(findings)} nghi ngờ · {sum(1 for f in findings if f['mechanical'])} cơ học "
        f"(không cần LLM) · {len(pending)} cần phân xử · đã có {len(verdicts)}",
        flush=True,
    )
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    lock = threading.Lock()

    def run_batch(index: int, batch: list[dict[str, Any]]) -> None:
        payload = {"can_phan_xu": [_payload(item) for item in batch]}
        last_error = None
        for _ in range(max(1, attempts)):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": DELETE_PROMPT if mode == "delete" else SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    reasoning_effort="none",
                )
                data = json.loads(response.choices[0].message.content)
                rows = {row["id"]: row for row in data.get("verdicts") or []}
                fresh = []
                for item in batch:
                    key = finding_id(item)
                    row = rows.get(key)
                    if not row:
                        continue
                    if mode == "delete":
                        fresh.append({
                            "id": key,
                            "verdict": "delete" if row.get("verdict") == "delete" else "keep",
                            "reason": str(row.get("reason") or "")[:400],
                            "model": model,
                            "input_signature": signature,
                        })
                        continue
                    fresh.append({
                        "id": key,
                        "verdict": "fix" if row.get("verdict") == "fix" else "keep",
                        "assertions": [
                            name for name in (row.get("assertions") or [])
                            if name in {"isNegated", "isFamily", "isHistorical"}
                        ],
                        "type": (row.get("type") or "").strip() or None,
                        "span": (row.get("span") or "").strip() or None,
                        "reason": str(row.get("reason") or "")[:400],
                        "model": model,
                        "input_signature": signature,
                    })
                with lock:
                    for row in fresh:
                        verdicts[row["id"]] = row
                    with journal_path.open("a", encoding="utf-8") as stream:
                        for row in fresh:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                wanted = "delete" if mode == "delete" else "fix"
                hits = sum(1 for row in fresh if row["verdict"] == wanted)
                label = "đề xuất xoá" if mode == "delete" else "đề xuất sửa"
                print(
                    f"[{index:04d}/{len(batches):04d}] {len(fresh)}/{len(batch)} phán xử, "
                    f"{hits} {label}",
                    flush=True,
                )
                return
            except Exception as exc:
                last_error = exc
        print(f"[{index:04d}/{len(batches):04d}] LỖI {type(last_error).__name__}", flush=True)

    if batches:
        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [
                executor.submit(run_batch, index, batch)
                for index, batch in enumerate(batches, 1)
            ]
            for future in as_completed(futures):
                future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    rows = []
    for finding in findings:
        key = finding_id(finding)
        verdict = verdicts.get(key)
        if finding["mechanical"]:
            decision = {
                "verdict": "fix",
                "assertions": (finding["proposed"] or {}).get("assertions", []),
                "reason": finding["reason"],
                "model": "rule_only",
            }
        elif verdict is None:
            continue
        else:
            decision = verdict
        if mode == "delete":
            if decision["verdict"] != "delete":
                continue
            rows.append({
                **{key2: value for key2, value in finding.items() if key2 != "entity"},
                "entity": finding["entity"],
                "llm": decision,
                "proposed_delete": True,
                "proposed_assertions": finding["current"]["assertions"],
            })
            continue
        if decision["verdict"] != "fix":
            continue
        change = _resolve_change(finding, decision)
        if change is None:
            continue
        rows.append({
            **{key2: value for key2, value in finding.items() if key2 != "entity"},
            "entity": finding["entity"],
            "llm": decision,
            **change,
        })
    _write_jsonl(
        out_dir / ("findings_delete.jsonl" if mode == "delete" else "findings.jsonl"), rows
    )
    report = {
        "stage": "label_audit_findings",
        "model": model,
        "screened": len(findings),
        "mechanical": sum(1 for f in findings if f["mechanical"]),
        "adjudicated": len(verdicts),
        "confirmed_fixes": len(rows),
        "by_kind": dict(Counter(row["kind"] for row in rows)),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "labels_modified": 0,
        "next_stage_runs_automatically": False,
    }
    (out_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_all_findings(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Gộp hai luồng. XOÁ THẮNG khi một entity vừa bị đề xuất sửa vừa bị đề xuất xoá.

    Sửa assertion cho một entity lẽ ra không nên tồn tại là việc vô nghĩa; và phán xử "đây không
    phải entity" là phán xử cơ bản hơn phán xử "assertion của nó sai". Đo được 261 ca chồng lấn.
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(out_dir / "findings.jsonl"):
        merged[row["file"] + "#" + str(row["entity_index"])] = row
    for row in _read_jsonl(out_dir / "findings_delete.jsonl"):
        key = row["file"] + "#" + str(row["entity_index"])
        if key in merged:
            row = {**row, "supersedes_fix": merged[key].get("kind")}
        merged[key] = row
    return merged


def apply_decisions(out_dir: Path, fixed_root: Path) -> dict[str, Any]:
    """Ghi bản nhãn ĐÃ SỬA sang thư mục riêng; không đụng vào nhãn gốc."""
    findings = load_all_findings(out_dir)
    decisions = json.loads((out_dir / "decisions.json").read_text(encoding="utf-8"))
    accepted = {key for key, value in decisions.items() if value == "accept"}
    if not accepted:
        raise RuntimeError("Chưa có quyết định 'accept' nào trong decisions.json")
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source, path in SOURCES.items():
        records = load_labeled_dir(REPO_ROOT / path, source)
        by_source[source] = {record["file_stem"]: record for record in records}
    changed = Counter()
    # Xoá phải làm SAU cùng và theo chỉ số giảm dần, nếu không mọi entity phía sau bị lệch index.
    to_delete: dict[tuple[str, str], list[int]] = {}
    for key in sorted(accepted):
        finding = findings.get(key)
        if finding is None:
            continue
        source = finding["source"]
        stem = finding["file"].split(":", 1)[1]
        record = by_source[source][stem]
        if finding.get("proposed_delete"):
            to_delete.setdefault((source, stem), []).append(finding["entity_index"])
            changed[f"{source}:xoá"] += 1
            changed[source] += 1
            continue
        entity = record["entities"][finding["entity_index"]]
        entity["assertions"] = list(finding["proposed_assertions"])
        if finding.get("proposed_type"):
            entity["type"] = finding["proposed_type"]
        if finding.get("proposed_position"):
            start, end = finding["proposed_position"]
            # Kiểm lại trên RAW trước khi ghi: span mới phải trích đúng nguyên văn.
            if record["text"][start:end] == finding["proposed_text"]:
                entity["position"] = [start, end]
                entity["text"] = finding["proposed_text"]
            else:
                changed[f"{source}:span bị bỏ vì lệch offset"] += 1
                continue
        changed[source] += 1
    for (source, stem), indexes in to_delete.items():
        entities = by_source[source][stem]["entities"]
        for index in sorted(set(indexes), reverse=True):
            del entities[index]
    fixed_root.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    for source, records in by_source.items():
        if not changed[source]:
            continue
        target = fixed_root / source
        (target / "notes").mkdir(parents=True, exist_ok=True)
        (target / "labels").mkdir(parents=True, exist_ok=True)
        for stem, record in records.items():
            (target / "notes" / f"{stem}.txt").write_text(record["text"], encoding="utf-8")
            (target / "labels" / f"{stem}.json").write_text(
                json.dumps(record["entities"], ensure_ascii=False, indent=1), encoding="utf-8"
            )
        manifest_files[source] = len(records)
    manifest = {
        "stage": "label_audit_applied",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_of_truth": "nhãn gốc KHÔNG bị sửa; đây là bản sao đã áp dụng quyết định người dùng",
        "accepted_fixes": dict(changed),
        "rejected": sum(1 for value in decisions.values() if value == "reject"),
        "files_written": manifest_files,
        "originals": SOURCES,
    }
    (fixed_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    screen_cmd = sub.add_parser("screen", help="Chỉ sàng bằng luật, không gọi LLM")
    screen_cmd.add_argument("--source", action="append", choices=list(SOURCES), default=[])
    screen_cmd.add_argument("--out", default="annotation/data/label_audit")

    run_cmd = sub.add_parser("adjudicate", help="Sàng bằng luật rồi để LLM phân xử")
    run_cmd.add_argument("--source", action="append", choices=list(SOURCES), default=[])
    run_cmd.add_argument("--out", default="annotation/data/label_audit")
    run_cmd.add_argument("--batch-size", type=int, default=8)
    run_cmd.add_argument("--concurrency", type=int, default=6)
    run_cmd.add_argument("--request-timeout", type=float, default=120)
    run_cmd.add_argument("--attempts", type=int, default=3)
    run_cmd.add_argument("--limit", type=int, default=0, help="chỉ phân xử N nghi ngờ đầu (thử nhanh)")
    run_cmd.add_argument("--execute-llm", action="store_true")
    run_cmd.add_argument(
        "--mode", choices=["fix", "delete"], default="fix",
        help="fix = sửa assertion/type/span (mặc định) · delete = soát lại quyết định gán",
    )
    run_cmd.add_argument("--sample", type=int, default=0, help="lấy mẫu ngẫu nhiên N ứng viên")
    run_cmd.add_argument("--seed", type=int, default=20260731)

    apply_cmd = sub.add_parser("apply", help="Ghi nhãn đã sửa sang thư mục riêng")
    apply_cmd.add_argument("--out", default="annotation/data/label_audit")
    apply_cmd.add_argument("--fixed-out", default="annotation/data/label_fixed")

    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if args.command == "screen":
        findings = screen_all(args.source or list(SOURCES))
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "screened.jsonl", findings)
        print(json.dumps({
            "screened": len(findings),
            "by_kind": dict(Counter(row["kind"] for row in findings)),
            "by_source": dict(Counter(row["source"] for row in findings)),
            "mechanical": sum(1 for row in findings if row["mechanical"]),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "adjudicate":
        if not args.execute_llm:
            raise SystemExit("Thiếu --execute-llm; không gọi Azure ngầm")
        findings = (
            screen_delete_all(args.source or list(SOURCES)) if args.mode == "delete"
            else screen_all(args.source or list(SOURCES))
        )
        if args.sample:
            rng = random.Random(args.seed)
            findings = rng.sample(findings, min(args.sample, len(findings)))
        if args.limit:
            mechanical = [row for row in findings if row["mechanical"]]
            rest = [row for row in findings if not row["mechanical"]][: args.limit]
            findings = mechanical + rest
        report = adjudicate(
            findings, out_dir, args.batch_size, args.concurrency,
            args.request_timeout, args.attempts, args.mode,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    fixed_root = (
        REPO_ROOT / args.fixed_out if not Path(args.fixed_out).is_absolute()
        else Path(args.fixed_out)
    )
    print(json.dumps(apply_decisions(out_dir, fixed_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
