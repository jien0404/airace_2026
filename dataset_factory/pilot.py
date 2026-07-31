from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from entity_clean.load import load_clean

from .casebook import (
    DEFAULT_CASEBOOK,
    is_word_bounded,
    load_cases,
    materialize_case,
    occurrence_position,
    validate_cases,
)
from .config import REPO_ROOT, sha256_file
from .llm_writer import azure_client
from .schema import ASSERTIONS, ASSERTION_TYPES, TYPES
from .sources import EntityCatalog, entity_allowed

MAX_PILOT_DRAFTS = 6_000
MAX_VARIANTS_PER_CASE = 40
MAX_CONCURRENCY = 16
DEFAULT_ENTITY_SEED = 20260730
CLEAN_MANIFEST = REPO_ROOT / "dictionary" / "clean" / "_manifest.json"

SYSTEM_PROMPT = """Bạn tạo MỘT trích đoạn tài liệu y khoa tiếng Việt để KIỂM ĐỊNH nghiệp vụ NER.
Đây chỉ là draft cho con người duyệt, KHÔNG phải dữ liệu train đã được chấp nhận.

Yêu cầu:
- `entity_contract.entity_plan` là DANH SÁCH ĐÓNG. Phải dùng đúng các surface đã cấp, theo đúng
  số occurrence, label decision, type và assertion. Không đổi, rút gọn hay mở rộng surface.
- Surface phải là toàn bộ concept y khoa. Không gắn thêm từ làm đổi nghĩa concept nhưng lại để
  ngoài span, ví dụ seed `tập trung` không được viết thành `khó tập trung`. Nếu một surface không
  thể đặt tự nhiên mà không cần mở rộng, hãy viết lại câu; không được tự sửa surface.
- TUYỆT ĐỐI không tự thêm bất kỳ nội dung nào có thể là TRIỆU_CHỨNG, CHẨN_ĐOÁN,
  TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM hoặc THUỐC ngoài entity_plan — kể cả mention phủ định,
  tiền sử, ví dụ, chỉ số sinh tồn, con số/đơn vị, thuốc phụ hay xét nghiệm dự kiến.
- Chỉ được thêm từ nối và khung phi-entity cần thiết để tạo context tự nhiên: chủ thể, thời gian,
  quan hệ diễn ngôn, hành chính và hành động chung. Nếu thiếu entity để viết dài, viết ngắn hơn;
  không được bịa entity để lấp độ dài.
- Tuân thủ generation_brief về thể loại, độ dài, heading/section và ngân sách entity.
- Nếu `scope` là `toàn văn bản nhiều mục`, hãy viết một tài liệu hoàn chỉnh với 3-5 mục liên tiếp;
  dùng `document_headers` làm các heading đầu mục. Context của từng entity có thể phụ thuộc heading
  và các mục trước, nhưng không được thêm surface y khoa ngoài entity_plan.
- Viết giống tài liệu y khoa thật. Có thể dùng heading, dòng trường-giá trị, bullet hoặc hội thoại;
  không bắt buộc mọi nội dung thành câu văn xuôi. Không dùng bảng.
- Heading và section_path là một phần context. Tuy nhiên không suy assertion chỉ từ heading:
  occurrence vẫn cần bằng chứng đủ rõ trong nội dung.
- Tuân thủ header_mode: `none` thì không tự thêm tiêu đề và trả section_path=[]; `inline` thì
  dùng heading_text như nhãn mục cùng dòng; `explicit_title/explicit_upper` dùng một dòng heading.
- Khi có heading, giữ nguyên `heading_text` và dùng sentence case tự nhiên; không tự chuyển thành
  CHỮ HOA TOÀN BỘ. Khi `header_mode=none`, không được tạo bất kỳ dòng tiêu đề thay thế nào.
- Giữ đúng kiểu quyết định của reference case. Surface đa dạng đã được sampler cấp sẵn.
- `mentions` phải có đúng một dòng cho mỗi seed_id trong entity_plan, không hơn, không thiếu;
  intentional-O vẫn phải liệt kê với should_label=false.
- Mỗi surface trong entity_plan chỉ được xuất hiện đúng số occurrence đã cấp trên TOÀN BỘ section.
  Không nhắc lại ở phần tóm tắt/kết luận, heading, hoặc câu dặn dò. Không sao chép lại danh sách
  entity để làm cho văn bản dài hơn.
- Quyết định từng occurrence độc lập; không dùng dictionary để quyết định type.
- Assertion phải dựa vào bằng chứng trong cùng mệnh đề/context.
- TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn assertions=[].
- Không tự gán candidate ICD/RxNorm.
- Không viết câu giải thích annotation như “đây chỉ là ví dụ/minh họa”, “không mô tả bệnh nhân”,
  “không cần gán nhãn”, “thuật ngữ/entity”, “nội dung này nhằm minh họa/tham khảo”.
  Context tự nhiên phải tự thể hiện vai trò discourse.
- Không thêm filler y khoa chỉ để đủ số lượng. Entity phụ phải hợp với thể loại và section.
- Không tạo span lồng/overlap. Chỉ dùng assertion: isNegated, isFamily, isHistorical.
- Không gán các danh từ meta đứng riêng như “thuốc”, “xét nghiệm”, “triệu chứng”, “chẩn đoán”.
- Trong pilot core hiện tại, không gán isHistorical cho THUỐC vì nghiệp vụ này còn OPEN.

Trả JSON duy nhất:
{
  "source_case_id": "...",
  "document_format": "... đúng generation_brief ...",
  "section_path": ["heading xuất hiện nguyên văn trong text"] hoặc [] nếu brief không có header,
  "text": "...",
  "mentions": [
    {
      "seed_id": "đúng seed_id trong entity_plan",
      "text": "nguyên văn",
      "occurrence_index": 0,
      "should_label": true,
      "type": "một trong năm type hoặc null",
      "assertions": [],
      "context_evidence": "cụm bằng chứng nguyên văn, liên tục và chứa đúng occurrence",
      "rationale": "lý do ngắn"
    }
  ]
}
Với intentional-O: should_label=false, type=null, assertions=[].
Không trả offset; hệ thống sẽ exact-align trên RAW."""

SYSTEM_PROMPT_GATE = """Bạn tạo MỘT trích đoạn tài liệu y khoa tiếng Việt để huấn luyện NER.
Đây là draft cho con người duyệt, KHÔNG phải dữ liệu train đã được chấp nhận.

QUAN TRỌNG NHẤT — mỗi occurrence là một quyết định riêng:
- KHÔNG áp dụng luật "một khái niệm xuất hiện nhiều lần thì lần nào cũng là entity".
- `entity_plan` liệt kê TỪNG occurrence phải viết ra, kể cả occurrence CỐ Ý KHÔNG GÁN.
- Occurrence có `occurrence_role=repeat_omitted` phải xuất hiện trong văn bản, cùng surface y hệt,
  nhưng đặt trong ngữ cảnh mà quy ước KHÔNG gán: câu dặn dò/phòng ngừa, đoạn giáo dục chung, ví dụ
  minh họa, câu giả định, hướng dẫn xử trí chung, hoặc lặp thừa trong ngoặc. Lý do cụ thể nằm ở
  `gate_reason`. Ngữ cảnh phải tự nhiên và tự nó đủ để người đọc thấy đây không phải sự kiện của
  bệnh nhân.
- Occurrence có `occurrence_role=repeat_labeled` là lần nhắc lại VẪN được gán, ở một ngữ cảnh
  bệnh nhân khác (mốc thời gian khác, mệnh đề khác). Hai lần này có thể mang assertion khác nhau.
BỐ CỤC — bệnh án thật là DANH SÁCH VÀ BẢNG, không phải văn xuôi:
- Đo được trên bộ test: 31 dòng/tài liệu, 50% số dòng ngắn hơn 40 ký tự. Draft phải giống vậy:
  tối thiểu 20 dòng, và ít nhất 40% số dòng là dòng ngắn.
- Dùng thật sự các hình thức sau, không chỉ viết đoạn văn dài: heading ngắn đứng riêng một dòng;
  mục gạch đầu dòng `- ...`; dòng `Tên trường: giá trị`; bảng xét nghiệm mỗi chỉ số một dòng.
- Câu văn xuôi vẫn được dùng cho bệnh sử và diễn biến, nhưng không được chiếm cả tài liệu.

CUE ASSERTION phải theo `assertion_cue_style` của từng entity, KHÔNG chỉ dùng cụm từ:
- `prose` — cue bằng cụm từ trong câu: `từng có`, `đã điều trị`, `phủ nhận`, `mẹ bệnh nhân`.
- `list_marker` — cue nằm ở CẤU TRÚC mục liệt kê, ví dụ dòng `- Tăng huyết áp - đã điều trị ổn`
  trong mục Tiền sử: chính vị trí trong mục và phần chú sau dấu gạch làm nên nghĩa tiền sử.
- `column_field` — cue là tên trường của dòng, ví dụ `Tiền sử bản thân: ...`,
  `Thuốc đã ngừng: ...`, `Tiền sử gia đình: ...`.
- Không lặp đi lặp lại một cụm cue: không cụm nào được chiếm quá 15% số ca trong tài liệu.

- THỨ TỰ occurrence do `order_hint` quyết định, KHÔNG phải "bản được gán luôn đứng trước":
  `order_hint=before_primary` nghĩa là occurrence KHÔNG gán phải nằm TRƯỚC occurrence được gán
  trong văn bản; `after_primary` thì ngược lại. Đây là điểm bắt buộc: nếu lần nào occurrence được
  gán cũng là lần xuất hiện đầu tiên, dữ liệu chỉ dạy được luật "lấy lần đầu" thay vì đọc ngữ cảnh.
- Occurrence có `occurrence_role=covered_by_longer` KHÔNG được viết riêng ở bất kỳ đâu: nó chỉ
  tồn tại bên trong entity dài ghi ở `covered_by`, và vì đã được entity dài cover nên không gán
  và không bọc marker. Chỉ cần viết entity dài đúng nguyên văn là occurrence này tự có mặt.
  Ví dụ: plan có `viêm phổi kẽ` (gán) và `viêm phổi` (covered_by_longer). Viết
  `<E id="s1">viêm phổi kẽ</E> tiến triển chậm` là ĐÚNG. Viết thêm câu `theo dõi viêm phổi`
  ở chỗ khác là SAI vì `viêm phổi` đã xuất hiện ngoài entity dài.

Đánh dấu occurrence được gán bằng marker, KHÔNG tự đếm offset:
- Bọc đúng chuỗi surface của mọi occurrence `should_label=true` bằng `<E id="SEED_ID">…</E>`,
  với SEED_ID là `seed_id` trong entity_plan.
- Occurrence `should_label=false` (repeat_omitted và intentional-O) viết BÌNH THƯỜNG, KHÔNG marker.
- Marker không được lồng nhau, không bọc thừa dấu cách hay dấu câu, và phải bọc TRỌN concept.
- Ngoài các marker này, không dùng ký tự `<` hay `>` ở bất kỳ đâu.

Yêu cầu nội dung:
- `entity_contract.entity_plan` là DANH SÁCH ĐÓNG. Phải dùng đúng surface đã cấp, đúng số lần đã
  cấp, đúng label decision, type và assertion. Không đổi, rút gọn hay mở rộng surface.
- Surface phải là toàn bộ concept y khoa; không gắn thêm từ làm đổi nghĩa concept nhưng để ngoài
  span, ví dụ seed `tập trung` không được viết thành `khó tập trung`.
- TUYỆT ĐỐI không tự thêm nội dung nào khác có thể là TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM,
  KẾT_QUẢ_XÉT_NGHIỆM hoặc THUỐC ngoài entity_plan — kể cả mention phủ định, tiền sử, ví dụ, chỉ số
  sinh tồn, con số/đơn vị, thuốc phụ hay xét nghiệm dự kiến.
- Chỉ được thêm từ nối và khung phi-entity: chủ thể, thời gian, quan hệ diễn ngôn, hành chính.
- Tuân thủ generation_brief về thể loại, độ dài, heading/section và ngân sách entity.
- ASSERTION PHẢI CÓ CUE TRONG CÙNG MỆNH ĐỀ. Nếu plan ghi `isNegated` thì câu phải có phủ định trực
  tiếp ("không", "chưa", "phủ nhận"). Nếu ghi `isHistorical` thì phải có mốc quá khứ rõ ("tiền sử",
  "trước đây", "đã từng", "năm 2019"). Nếu ghi `isFamily` thì phải nêu người thân ("bố", "mẹ").
  NGƯỢC LẠI, occurrence có `assertions: []` thì TUYỆT ĐỐI không được đặt sau các cue đó — nếu cần
  ngữ cảnh quá khứ hay phủ định thì phải viết lại câu cho thành sự kiện hiện tại, khẳng định.
- `nghi ngờ`, `theo dõi`, `khả năng` KHÔNG phải phủ định; đừng dùng chúng để diễn đạt isNegated.
- Tuân thủ header_mode: `none` thì không tự thêm tiêu đề và trả section_path=[]; `inline` dùng
  heading_text như nhãn mục cùng dòng; `explicit_title` dùng một dòng heading. Giữ nguyên
  `heading_text`, dùng sentence case tự nhiên, không CHỮ HOA TOÀN BỘ.
- Viết giống tài liệu y khoa thật: heading, dòng trường-giá trị, bullet hoặc hội thoại đều được.
  Không dùng bảng.
- TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn assertions=[].
- Không gán các danh từ meta đứng riêng như "thuốc", "xét nghiệm", "triệu chứng", "chẩn đoán".
- Không viết câu giải thích annotation như "đây chỉ là ví dụ/minh họa", "không cần gán nhãn".
- Không tự gán candidate ICD/RxNorm.

Trả JSON duy nhất:
{
  "source_case_id": "...",
  "document_format": "... đúng generation_brief ...",
  "section_path": ["heading xuất hiện nguyên văn trong text"] hoặc [],
  "marked_text": "toàn bộ văn bản, occurrence được gán bọc trong <E id=\\"...\\">...</E>",
  "mentions": [
    {
      "seed_id": "đúng seed_id trong entity_plan",
      "text": "nguyên văn surface",
      "should_label": true,
      "type": "một trong năm type hoặc null",
      "assertions": [],
      "context_evidence": "cụm bằng chứng nguyên văn, liên tục, chứa đúng occurrence này; trích từ văn bản SAU KHI đã bỏ marker, KHÔNG được chứa <E hay </E>",
      "rationale": "lý do ngắn cho quyết định gán/không gán occurrence này"
    }
  ]
}
`mentions` phải có đúng một dòng cho mỗi seed_id trong entity_plan, không hơn không thiếu.
Với occurrence không gán: should_label=false, type=null, assertions=[]."""

META_TEXT_PATTERNS = (
    "không mô tả tình trạng",
    "không nói về đơn thuốc",
    "không nói về triệu chứng",
    "chỉ là thông tin minh họa",
    "chỉ mang tính minh họa",
    "nhằm minh họa",
    "thông tin này chỉ là",
    "chỉ là lưu ý chung",
    "lưu ý chung",
    "không cần gán nhãn",
    "intentional-o",
    "gán nhãn",
    "thuật ngữ này",
    "entity này",
    # Nói VỀ NGÔN TỪ thay vì về y khoa: cùng một mùi meta, làm văn bản mất tự nhiên.
    "các cụm như",
    "cụm từ",
    "cách nói",
    "cách gọi",
)

_EVIDENCE_SEPARATOR = re.compile(r"\s*(?:\.\.\.|…|;)\s*")
_MEASUREMENT_PATTERN = re.compile(
    r"(?i)(?<!\w)\d+(?:[.,]\d+)?\s*"
    r"(?:%|°\s*c|độ\s*c|mg|g|kg|ml|l|mmhg|mmol/l|µmol/l|u/l|l/phút|l/p)(?!\w)"
)
_META_GUARD_TERMS = {
    "bệnh nhân",
    "bác sĩ",
    "thuốc",
    "xét nghiệm",
    "triệu chứng",
    "chẩn đoán",
    "kết quả",
}
_PILOT_SURFACE_BLOCKLIST = (
    "cần thêm thông tin",
    "triệu chứng của",
    "không phẫu thuật",
    "tập trung",
)
_AMBIGUOUS_CONTEXT_PREFIXES = (
    "bệnh sử",
    "cảm giác",
    "có ",
    "đi lại",
    "gắng sức",
    "hoạt động",
    "khám",
    "làm việc",
    "lo lắng",
    "người",
    "quan sát",
    "sinh hoạt",
    "theo dõi",
)


_MARKER_RE = re.compile(r'<E\s+id="([^"]{1,64})"\s*>(.*?)</E>', re.DOTALL)
_MARKER_LEFTOVER_RE = re.compile(r"</?E\b|<E|</E>")

# Cue đủ mạnh để một occurrence KHÔNG THỂ mang assertions rỗng, và ngược lại, để một
# assertion đã lên kế hoạch phải có bằng chứng. Cố ý giữ hẹp: chỉ nhận cue đứng ngay
# trước occurrence, tránh phạt oan các câu dài nhiều mệnh đề.
ASSERTION_CUES: dict[str, tuple[str, ...]] = {
    "isNegated": (
        "không", "chưa", "phủ nhận", "loại trừ", "hết", "âm tính với", "không còn",
        "không có", "không ghi nhận", "không thấy",
    ),
    "isHistorical": (
        "tiền sử", "trước đây", "trước đó", "trước kia", "đã từng", "từng bị",
        "từng được", "từng mắc", "từng có", "từng", "trong quá khứ", "bệnh án cũ",
        "hồ sơ cũ", "nhiều năm trước", "cách đây", "hồi", "lúc nhỏ", "đợt trước",
        "năm ngoái", "đã điều trị ổn định", "trong quá trình theo dõi trước",
    ),
    "isFamily": (
        "bố", "mẹ", "cha", "ba", "má", "ông", "bà", "anh trai", "chị gái", "em trai",
        "em gái", "người nhà", "gia đình", "con trai", "con gái", "cậu", "dì", "chú",
        "bác", "anh", "chị", "em", "họ hàng", "bên nội", "bên ngoại",
    ),
}
# Cue mâu thuẫn: dùng để bắt lỗi "văn bản nói tiền sử/phủ định nhưng plan để assertions rỗng".
# Cố ý HẸP và phải đứng sát occurrence, vì hướng này loại cả draft nên sai một lần là đắt.
CONFLICT_CUES: dict[str, tuple[str, ...]] = {
    "isNegated": ("không", "chưa", "phủ nhận", "loại trừ", "không ghi nhận", "không có"),
    "isHistorical": (
        "tiền sử", "trước đây", "đã từng", "từng bị", "từng mắc", "từng", "bệnh án cũ",
        "hồ sơ cũ",
    ),
}
# "từng" là cue tiền sử mạnh ("người bệnh từng đau ngực") nhưng cũng là lượng từ
# ("điều trị từng đợt"). Loại các danh từ đi sau khiến nó không còn nghĩa thời gian.
_TUNG_NOT_HISTORICAL = (
    "đợt", "bước", "loại", "người", "ngày", "lần", "cái", "mục", "phần",
    # lượng từ/phân loại: "run rẩy từng cơn", "từng giai đoạn", "từng tuần"
    "cơn", "giai đoạn", "khoảng", "phút", "giờ", "tuần", "tháng", "năm", "chặng",
    "nhịp", "chút", "ít", "vùng", "bên", "chi",
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SUPPORT_WINDOW_BEFORE = 220
_SUPPORT_WINDOW_AFTER = 26
_CONFLICT_WINDOW = 30
_CONFLICT_MAX_GAP = 18


def _clause_before(text: str, start: int, window: int, hard_only: bool) -> str:
    """Lấy đoạn ngay trước occurrence, cắt tại ranh giới câu (và mệnh đề nếu hard_only)."""
    chunk = text[max(0, start - window):start].casefold()
    separators = (". ", "\n", "! ", "? ")
    if hard_only:
        separators = separators + ("; ", ", ", ": ")
    for separator in separators:
        cut = chunk.rfind(separator)
        if cut >= 0:
            chunk = chunk[cut + len(separator):]
    return chunk


def _word_bounded_hits(chunk: str, cue: str, max_gap: int | None) -> bool:
    index = chunk.rfind(cue)
    while index >= 0:
        before_char = chunk[index - 1] if index else " "
        after = chunk[index + len(cue):]
        boundary_ok = not re.match(r"^[0-9a-zà-ỹ]", before_char) and not (
            after and re.match(r"^[0-9a-zà-ỹ]", after[0])
        )
        if boundary_ok and (max_gap is None or len(after) <= max_gap):
            return True
        index = chunk.rfind(cue, 0, index)
    return False


def _cues_support(text: str, position: list[int]) -> set[str]:
    """Cue ĐỦ để một assertion đã lên kế hoạch được coi là có bằng chứng (rộng tay).

    Quét ngược tới đầu câu chứ không phải một cửa sổ ngắn: trong "bố bị tăng huyết áp;
    ngoài ra còn có X và từng than phiền Y", chủ thể "bố" chi phối cả X lẫn Y.
    """
    start, end = position
    before = _clause_before(text, start, _SUPPORT_WINDOW_BEFORE, hard_only=False)
    after = text[end:end + _SUPPORT_WINDOW_AFTER].casefold()
    after = re.split(r"[.!?\n]", after)[0]
    found = set()
    for name, cues in ASSERTION_CUES.items():
        for cue in cues:
            if _word_bounded_hits(before, cue, None):
                found.add(name)
                break
        else:
            # Phủ định tiếng Việt có thể đứng SAU concept: "sốt không có", "ho: không".
            if name == "isNegated" and any(
                _word_bounded_hits(after, cue, None)
                for cue in ("không", "chưa", "âm tính")
            ):
                found.add(name)
    if _YEAR_RE.search(before) or "năm" in before:
        found.add("isHistorical")
    return found


def _cues_before(text: str, start: int) -> set[str]:
    """Cue MÂU THUẪN: hẹp, phải sát occurrence, dùng để chặn nhãn rỗng đặt sai chỗ."""
    chunk = _clause_before(text, start, _CONFLICT_WINDOW, hard_only=True)
    found = set()
    for name, cues in CONFLICT_CUES.items():
        for cue in cues:
            if not _word_bounded_hits(chunk, cue, _CONFLICT_MAX_GAP):
                continue
            if cue == "từng":
                index = chunk.rfind("từng")
                tail = chunk[index + len("từng"):].lstrip()
                if tail.split(" ")[0] in _TUNG_NOT_HISTORICAL:
                    continue
            found.add(name)
            break
    return found


def _parse_marked_text(marked: str) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Bỏ marker và tính offset trên RAW sinh ra. Không normalize gì thêm sau bước này."""
    errors: list[str] = []
    parts: list[str] = []
    marks: list[dict[str, Any]] = []
    cursor = 0
    length = 0
    for match in _MARKER_RE.finditer(marked):
        chunk = marked[cursor:match.start()]
        parts.append(chunk)
        length += len(chunk)
        surface = match.group(2)
        if not surface or surface != surface.strip():
            errors.append(
                f"marker {match.group(1)!r} bọc thừa khoảng trắng hoặc rỗng: {surface!r}"
            )
        marks.append({
            "seed_id": match.group(1),
            "text": surface,
            "position": [length, length + len(surface)],
        })
        parts.append(surface)
        length += len(surface)
        cursor = match.end()
    parts.append(marked[cursor:])
    raw = "".join(parts)
    if _MARKER_LEFTOVER_RE.search(raw):
        errors.append("còn sót marker hỏng trong text sau khi bỏ marker")
    return raw, marks, errors


def _all_exact_occurrences(text: str, surface: str) -> list[list[int]]:
    positions: list[list[int]] = []
    cursor = 0
    while surface:
        start = text.find(surface, cursor)
        if start < 0:
            break
        end = start + len(surface)
        if is_word_bounded(text, start, end):
            positions.append([start, end])
        cursor = start + max(1, len(surface))
    return positions


def _context_evidence_supports_occurrence(
    text: str,
    surface: str,
    position: list[int],
    evidence: str,
) -> bool:
    """Require evidence to resolve to the same occurrence, not merely exist elsewhere."""
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    start, end = position
    fragments = [evidence.strip()]
    fragments.extend(
        fragment.strip(" \t\r\n,.:")
        for fragment in _EVIDENCE_SEPARATOR.split(evidence)
        if fragment.strip(" \t\r\n,.:")
    )
    unique_fragments = []
    for fragment in fragments:
        if surface in fragment and fragment not in unique_fragments:
            unique_fragments.append(fragment)

    for fragment in unique_fragments:
        # A bare surface cannot disambiguate multiple occurrences.
        if fragment == surface and len(_all_exact_occurrences(text, surface)) > 1:
            continue
        cursor = 0
        while True:
            evidence_start = text.find(fragment, cursor)
            if evidence_start < 0:
                break
            evidence_end = evidence_start + len(fragment)
            if evidence_start <= start and end <= evidence_end:
                return True
            cursor = evidence_start + 1
    return False


def _surface_conflicts(surface: str, existing: set[str]) -> bool:
    folded = surface.casefold()
    for other in existing:
        other_folded = other.casefold()
        if folded == other_folded:
            return True
        for short, long in ((folded, other_folded), (other_folded, folded)):
            start = long.find(short)
            if start >= 0 and is_word_bounded(long, start, start + len(short)):
                return True
    return False


_NUMERIC_SURFACE_RE = re.compile(r"^[\d.,/+\-]+$")


def _gate_surface_safe(surface: str) -> bool:
    """Chặn surface không kiểm soát được số occurrence trong hợp đồng đóng.

    Số trần như `115`, `1.3` là KẾT_QUẢ hợp lệ trong gold, nhưng ở generator thì mọi con số
    khác trong văn bản đều có thể trùng, làm phép đếm occurrence sai và loại oan cả draft.
    Ký tự đơn cũng vậy. Các viết tắt chữ như `HA`, `XQ`, `ho` vẫn được giữ.
    """
    stripped = surface.strip()
    return len(stripped) >= 2 and not _NUMERIC_SURFACE_RE.match(stripped)


def _pilot_surface_safe(surface: str) -> bool:
    folded = surface.casefold().strip()
    return (
        not folded.startswith("theo dõi ")
        and not any(pattern in folded for pattern in _PILOT_SURFACE_BLOCKLIST)
    )


def _semantic_type_for_decision(
    case: dict[str, Any],
    decision: dict[str, Any],
    catalog: EntityCatalog,
) -> str:
    if decision.get("type") in TYPES:
        return decision["type"]
    matches = catalog.semantic_types(decision["text"])
    if len(matches) == 1:
        return matches[0]
    fallback = {
        ("discourse.drug_example.001", "d3"): "TRIỆU_CHỨNG",
        ("discourse.heading_meta.001", "d1"): "TRIỆU_CHỨNG",
        ("discourse.mechanism_selective.001", "d2"): "TRIỆU_CHỨNG",
    }
    resolved = fallback.get((case["case_id"], decision["decision_id"]))
    if resolved:
        return resolved
    raise RuntimeError(
        f"Không suy được semantic_type cho intentional-O "
        f"{case['case_id']}:{decision['decision_id']} {decision['text']!r}; matches={matches}"
    )


def _planned_item(
    seed_id: str,
    text: str,
    semantic_type: str,
    *,
    occurrence_index: int = 0,
    should_label: bool = True,
    typ: str | None = None,
    assertions: Iterable[str] = (),
    role: str,
    provenance: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "seed_id": seed_id,
        "text": text,
        "semantic_type": semantic_type,
        "occurrence_index": occurrence_index,
        "should_label": should_label,
        "type": (typ if should_label else None),
        "assertions": list(assertions) if should_label else [],
        "role": role,
        "provenance": list(provenance),
    }


# Phân phối type của gold V66ab; dùng làm trọng số bù type thay vì chia đều năm type.
GOLD_TYPE_WEIGHTS = {
    "TRIỆU_CHỨNG": 0.39,
    "CHẨN_ĐOÁN": 0.26,
    "TÊN_XÉT_NGHIỆM": 0.15,
    "KẾT_QUẢ_XÉT_NGHIỆM": 0.10,
    "THUỐC": 0.10,
}
# Gold có 16,6% entity mang assertion, chia isHistorical 62% / isNegated 36% / isFamily 2%.
# Chỉ TRIỆU/CHẨN/THUỐC được mang assertion nên xác suất trên mỗi entity đủ điều kiện cao hơn.
# Trọng số BỐC (khác bảng gold ở trên): THUỐC và KẾT_QUẢ_XÉT_NGHIỆM chỉ có mặt trong một
# phần các coverage profile, nên nếu bốc đúng tỷ lệ gold thì sản lượng cuối chỉ còn ~7%.
SAMPLING_TYPE_WEIGHTS = {
    "TRIỆU_CHỨNG": 0.34,
    "CHẨN_ĐOÁN": 0.26,
    # TÊN_XÉT_NGHIỆM thấp hơn gold vì mỗi cặp xét nghiệm-kết quả đã tự kèm một tên.
    "TÊN_XÉT_NGHIỆM": 0.06,
    "KẾT_QUẢ_XÉT_NGHIỆM": 0.18,
    "THUỐC": 0.115,
}
ASSERTION_TARGET_RATE = 0.22
ASSERTION_SPLIT = (("isHistorical", 0.62), ("isNegated", 0.36), ("isFamily", 0.02))
# Nhóm surface lặp: bao nhiêu phần trăm là "mixed" (có occurrence bị bỏ). Gold Part 2 đo được
# 20-33% tuỳ cách đếm; train được phép oversample lên 35-40% theo DATASET_CONTRACT §3b.
REPEAT_GATE_RATE = 0.38
# Tỷ lệ occurrence BỊ BỎ được viết TRƯỚC occurrence được gán. Không có tham số này, LLM luôn
# viết bản gán trước rồi mới tới bản bỏ, và dataset quay về đúng luật cũ "chỉ lấy lần đầu"
# (đo trên mẻ 5.000: 99,3% nhóm mixed gán occurrence đầu, trong khi gold gt2 chỉ 27,6%).
REPEAT_OMITTED_FIRST_RATE = 0.55
# Mẻ bổ sung chuyên occurrence gate: bật bằng `prepare --gate-focus`. Dùng để cân lại một mẻ đã
# sinh bị lệch (nhiều nhóm gate hơn mỗi bản ghi, nên cần ít bản ghi hơn để kéo tỷ lệ về mốc gold).
GATE_FOCUS = False
GATE_FOCUS_GATE_RATE = 0.62
# Đặt >0 để ép tỷ lệ before_primary khác mặc định — dùng khi mẻ bổ sung phải KÉO NGƯỢC một mẻ cũ
# đã lệch về "chỉ lấy lần đầu".
OMITTED_FIRST_RATE_OVERRIDE = 0.0
GATE_REASONS = (
    ("nhắc lại trong câu dặn dò/phòng ngừa cho người bệnh", 0.28),
    ("nhắc lại trong đoạn giáo dục kiến thức chung", 0.22),
    ("nêu như ví dụ minh họa, không phải sự kiện của người bệnh", 0.18),
    ("câu giả định 'nếu…' hoặc 'có thể gây…'", 0.17),
    ("hướng dẫn xử trí chung, không mô tả diễn biến người bệnh", 0.15),
)
# Ngoặc lặp nguyên văn chỉ tự nhiên trong lời kể/ASR; ép vào bệnh án trơn tru sẽ ra
# "viêm túi mật cấp (viêm túi mật cấp)" — đúng luật nhưng văn bản giả.
GATE_REASON_PARENTHETICAL = "lặp thừa trong ngoặc ngay sau lần nhắc đã được gán (chỉ hợp với lời kể/ASR)"
_ASR_FORMAT_MARKERS = ("asr", "transcript", "lời kể", "dialogue", "telehealth")


def _gate_reason(rng: Any, document_format: str | None) -> str:
    names, weights = zip(*GATE_REASONS)
    folded = (document_format or "").casefold()
    if any(marker in folded for marker in _ASR_FORMAT_MARKERS) and rng.random() < 0.35:
        return GATE_REASON_PARENTHETICAL
    return rng.choices(names, weights=weights, k=1)[0]


def _weighted_type_mix(type_mix: list[str], rng: Any) -> list[str]:
    """Xáo type theo trọng số gold: type nặng có xác suất đứng trước cao hơn."""
    remaining = list(type_mix)
    ordered = []
    while remaining:
        weights = [max(GOLD_TYPE_WEIGHTS.get(typ, 0.1), 0.01) for typ in remaining]
        chosen = rng.choices(remaining, weights=weights, k=1)[0]
        ordered.append(chosen)
        remaining.remove(chosen)
    return ordered


_HISTORICAL_DISCOURSE = ("tiền sử", "past_medical_history", "bệnh sử", "hồ sơ cũ")
_FAMILY_DISCOURSE = ("gia đình", "family_history")


def _discourse_assertion_bias(case: dict[str, Any], brief: dict[str, Any]) -> str | None:
    """Thể loại quyết định assertion mặc định của cả tài liệu.

    Một mục "Tiền sử bản thân" thì gần như mọi bệnh trong đó đều là isHistorical. Nếu plan
    vẫn bốc assertion độc lập 22% thì LLM buộc phải viết bệnh hiện tại trong mục tiền sử —
    đó chính là nguồn của 11 lỗi "đứng sau cue isHistorical nhưng plan để assertions rỗng".
    """
    haystack = " ".join(str(value).casefold() for value in (
        case.get("case_id"), brief.get("document_format"), brief.get("heading_text"),
        brief.get("required_heading"),
    ) if value)
    # Kiểm gia đình TRƯỚC: "Tiền sử gia đình" chứa cả hai dấu hiệu nhưng chủ thể mới là
    # thứ quyết định, và isFamily hẹp hơn isHistorical.
    if case.get("case_id", "").startswith("assertion.family") or any(
        marker in haystack for marker in _FAMILY_DISCOURSE
    ):
        return "isFamily"
    if case.get("case_id", "").startswith("assertion.historical") or any(
        marker in haystack for marker in _HISTORICAL_DISCOURSE
    ):
        return "isHistorical"
    return None


SELF_NEGATED_SURFACE = re.compile(r"^\s*(không|chưa|phủ nhận|ko)\b", re.IGNORECASE)
# Cue assertion phải đến từ NHIỀU hình thức, không chỉ cụm từ văn xuôi. Đo trên Part 3: cue
# `isHistorical` phổ biến nhất là `tính -`, `áp -`, `phì -` — tức tiền sử suy từ MỤC GẠCH ĐẦU
# DÒNG (`Tăng huyết áp - đã điều trị`), trong khi synthetic chỉ dạy `từng có`/`đã từng`/`tiền sử`.
# 10 bigram phổ biến nhất của synthetic phủ 66% ca, của Part 3 chỉ 33%.
# Đo trực tiếp trên Part 3, phân loại theo DÒNG chứa entity:
#   isHistorical: mục gạch đầu dòng 78% · văn xuôi 13% · dòng `Tên trường: giá trị` 9%
#   isNegated:    mục gạch đầu dòng 58% · văn xuôi 32% · dòng trường 10%
#   isFamily:     mục gạch đầu dòng 67% · văn xuôi 22% · dòng trường 11%
# Casebook hiện tại dạy 100% văn xuôi (31/32 case chỉ có một dòng), nên generator chưa bao giờ
# thấy hình thức chiếm đa số ngoài thực tế.
ASSERTION_CUE_STYLES = {
    "isHistorical": (("list_marker", 0.72), ("prose", 0.16), ("column_field", 0.12)),
    "isNegated": (("list_marker", 0.55), ("prose", 0.32), ("column_field", 0.13)),
    "isFamily": (("list_marker", 0.62), ("prose", 0.24), ("column_field", 0.14)),
}


def _cue_style(rng: Any, assertion_name: str) -> str:
    table = ASSERTION_CUE_STYLES.get(assertion_name) or ASSERTION_CUE_STYLES["isNegated"]
    names, weights = zip(*table)
    return rng.choices(names, weights=weights, k=1)[0]


def _assign_supplement_assertions(
    plan: list[dict[str, Any]],
    rng: Any,
    discourse_bias: str | None = None,
    target_rate: float | None = None,
) -> None:
    """Cấp assertion cho entity bổ sung theo tỷ lệ gold.

    Mẻ v1 chỉ có 0,49% entity mang assertion (gold 16,6%) vì entity ngẫu nhiên luôn được
    phát với `assertions=[]`; tệ hơn, LLM vẫn viết ngữ cảnh tiền sử quanh chúng nên sinh ra
    242 nhãn historical sai. Cấp assertion ngay từ plan để văn bản và nhãn cùng một nguồn.
    """
    for item in plan:
        if item["role"] != "supplemental_random" or not item["should_label"]:
            continue
        if item["type"] not in ASSERTION_TYPES:
            continue
        if discourse_bias:
            # Trong tài liệu tiền sử/gia đình, assertion là MẶC ĐỊNH của thể loại chứ không
            # phải ngoại lệ hiếm.
            if rng.random() >= 0.88:
                continue
            name = discourse_bias
        else:
            if rng.random() >= (
                target_rate if target_rate is not None else ASSERTION_TARGET_RATE
            ):
                continue
            names, weights = zip(*ASSERTION_SPLIT)
            name = rng.choices(names, weights=weights, k=1)[0]
        if name == "isHistorical" and item["type"] == "THUỐC":
            # Nghiệp vụ THUỐC-historical còn OPEN; không sinh hàng loạt.
            continue
        if name == "isNegated" and SELF_NEGATED_SURFACE.match(item["text"]):
            # Surface tự nó đã phủ định (`không có máu`); gán thêm isNegated đẻ ra phủ định kép
            # kiểu `không có Không dung nạp thức ăn` — 42 ca trong mẻ 5.000.
            continue
        item["assertions"] = [name]
        item["assertion_cue_style"] = _cue_style(rng, name)


# Ngưỡng bố cục tối thiểu (bộ test đo được 31 dòng, 50% dòng ngắn). Đặt dưới mức thật để vòng
# sửa chữa không phải viết lại quá nhiều draft, nhưng đủ để loại hẳn dạng văn xuôi thuần.
MIN_LINES_PER_RECORD = 18
MIN_SHORT_LINE_RATIO = 0.35
# Dưới ngưỡng này thì không xét bố cục — trích đoạn ngắn không nói được gì về layout.
LAYOUT_CHECK_MIN_CHARS = 600

COVERED_OCCURRENCE_RATE = 0.18
# Mật độ entity theo TỪNG bản ghi, khớp phân bố gold Part 3 (p10 5,1 · median 12,1 · p90 24,4;
# 31/100 file dưới 8/1k). Một khoảng cố định [10,16] cho ra median đúng nhưng mất hết phương
# sai, nên dataset không có tài liệu thưa — đúng loại tài liệu tư vấn/giáo dục của bộ test.
# Mỗi bucket mang PROFILE RIÊNG, không chỉ mật độ. Đo trên 27 tài liệu Part 3 không trùng Part 1
# (lớp tài liệu thưa) so với 73 tài liệu còn lại: mật độ 8,7/1k vs 15,1/1k, CHẨN_ĐOÁN 37% vs 23%,
# TRIỆU_CHỨNG 23% vs 43%, assertion 7,7% vs 18,6%. Mẻ 5.000 dùng chung một bảng trọng số type cho
# mọi bucket nên không hề có lớp tài liệu thưa-nặng-chẩn-đoán; đúng lớp đó precision của cả ba
# model sụp còn 43–47%.
DENSITY_BUCKETS = (
    ("sparse", (6, 10), 0.32, {
        "TRIỆU_CHỨNG": 0.22, "CHẨN_ĐOÁN": 0.38, "TÊN_XÉT_NGHIỆM": 0.06,
        "KẾT_QUẢ_XÉT_NGHIỆM": 0.19, "THUỐC": 0.15,
    }, 0.10),
    ("normal", (11, 16), 0.44, None, 0.22),
    ("dense", (17, 25), 0.24, None, 0.28),
)
# Bản ghi hoàn toàn không có nhãn: spec §8.1 yêu cầu có, gold cũng có file gần như trống.
EMPTY_RECORD_RATE = 0.06
INTENTIONAL_O_REASONS = (
    "kiến thức chung, không mô tả tình trạng của người bệnh này",
    "checklist biểu hiện dùng để giáo dục, không phải sự kiện của người bệnh",
    "tác dụng phụ/biến chứng nêu chung, chưa xảy ra với người bệnh",
    "cơ chế bệnh sinh, giải thích sinh học",
    "nêu như ví dụ minh họa trong câu tư vấn",
)


def _density_target(
    rng: Any, low_yield: bool, expected_chars: float,
) -> tuple[int, dict[str, float] | None, float]:
    """Bốc số entity theo mật độ mục tiêu, kèm profile type và tỷ lệ assertion của bucket.

    Trả về `(số entity, trọng số type của bucket hoặc None, tỷ lệ assertion)`.
    """
    if low_yield:
        # Thể loại tư vấn/giáo dục luôn nằm ở nhóm thưa; đôi khi không có nhãn nào.
        if rng.random() < EMPTY_RECORD_RATE * 4:
            return 0, None, ASSERTION_TARGET_RATE
        low, high = 3, 8
        type_weights, assertion_rate = None, 0.12
    else:
        index = rng.choices(
            range(len(DENSITY_BUCKETS)),
            weights=[bucket[2] for bucket in DENSITY_BUCKETS], k=1,
        )[0]
        _, (low, high), _, type_weights, assertion_rate = DENSITY_BUCKETS[index]
    density = rng.uniform(low, high)
    return (
        max(0, round(density * expected_chars / 1000)), type_weights, assertion_rate,
    )


def _plan_intentional_o(
    plan: list[dict[str, Any]],
    catalog: EntityCatalog,
    surfaces: set[str],
    count: int,
    type_mix: list[str],
) -> None:
    """Thêm thuật ngữ y khoa CÓ MẶT nhưng KHÔNG được gán vì vai trò diễn ngôn.

    Đây là thứ làm nên tài liệu thưa nhãn thật: đoạn tư vấn dày thuật ngữ mà gold gán rất ít.
    """
    for index in range(count):
        typ = catalog.rng.choice(type_mix or list(TYPES))
        if typ == "KẾT_QUẢ_XÉT_NGHIỆM":
            typ = "TRIỆU_CHỨNG"
        try:
            entity = catalog.sample(
                typ, exclude=surfaces, exclude_flagged=True, min_source_count=2,
            )
        except RuntimeError:
            return
        if _surface_conflicts(entity.text, surfaces) or not _gate_surface_safe(entity.text):
            continue
        if not _pilot_surface_safe(entity.text):
            continue
        surfaces.add(entity.text)
        plan.append({
            **_planned_item(
                f"discourse_o:{index + 1:02d}",
                entity.text,
                entity.type,
                should_label=False,
                role="intentional_o_discourse",
                provenance=entity.provenance,
            ),
            "occurrence_role": "intentional_o",
            "gate_reason": catalog.rng.choice(INTENTIONAL_O_REASONS),
        })


def _covered_sub_surface(
    long_text: str,
    catalog: EntityCatalog,
) -> tuple[str, str] | None:
    """Tìm một surface NGẮN nằm trọn trong một entity dài, theo ranh giới từ.

    Gold Part 2 có 2.939 occurrence kiểu này và quy ước cấm gán entity ngắn lồng trong
    entity dài đã cover nó. `_surface_conflicts` cấm hai surface lồng nhau nên nếu không
    dựng riêng thì dataset không bao giờ có mẫu này.
    """
    words = long_text.split()
    if len(words) < 2:
        return None
    for size in range(len(words) - 1, 0, -1):
        for start in range(0, len(words) - size + 1):
            phrase = " ".join(words[start:start + size]).strip(" ,.;:()")
            if len(phrase) < 4 or phrase == long_text:
                continue
            types = catalog.semantic_types(phrase)
            if types:
                return phrase, types[0]
    return None


def _plan_covered_occurrence(
    plan: list[dict[str, Any]],
    catalog: EntityCatalog,
) -> list[dict[str, Any]]:
    if catalog.rng.random() >= COVERED_OCCURRENCE_RATE:
        return plan
    candidates = [
        item for item in plan
        if item["should_label"] and len(item["text"].split()) >= 2
    ]
    catalog.rng.shuffle(candidates)
    for source in candidates:
        found = _covered_sub_surface(source["text"], catalog)
        if not found:
            continue
        phrase, semantic_type = found
        plan.append({
            **_planned_item(
                "covered:01",
                phrase,
                semantic_type,
                should_label=False,
                role="covered_by_longer_entity",
                provenance=source.get("provenance") or (),
            ),
            "occurrence_role": "covered_by_longer",
            "covered_by": source["seed_id"],
            "gate_reason": (
                f"nằm trọn trong entity dài hơn {source['text']!r} nên KHÔNG gán riêng"
            ),
        })
        return plan
    return plan


def _plan_repeat_occurrences(
    plan: list[dict[str, Any]],
    rng: Any,
    document_format: str | None = None,
) -> list[dict[str, Any]]:
    """Thêm occurrence lặp của surface đã có, gồm cả lặp ĐƯỢC GÁN và lặp BỊ BỎ.

    Đây là phần dạy occurrence gate: cùng một surface, occurrence này lấy, occurrence kia bỏ.
    Không có nó, dataset chỉ dạy nhận diện surface (mẻ v1: 0,03 nhóm lặp/bản ghi).
    """
    for item in plan:
        item.setdefault("occurrence_role", "primary")
    eligible = [
        item for item in plan
        if item["should_label"] and item["type"] in ASSERTION_TYPES
        and len(item["text"]) >= 4
    ]
    if not eligible:
        return plan
    if GATE_FOCUS:
        counts, weights, gate_rate = (2, 3, 4), (0.3, 0.45, 0.25), GATE_FOCUS_GATE_RATE
    else:
        counts, weights, gate_rate = (1, 2, 3), (0.5, 0.35, 0.15), REPEAT_GATE_RATE
    wanted = min(len(eligible), rng.choices(counts, weights=weights, k=1)[0])
    chosen = rng.sample(eligible, wanted)
    extra: list[dict[str, Any]] = []
    for index, source in enumerate(chosen, 1):
        gated = rng.random() < gate_rate
        if gated:
            extra.append({
                **_planned_item(
                    f"repeat:{index:02d}",
                    source["text"],
                    source["semantic_type"],
                    should_label=False,
                    role="occurrence_gate_negative",
                    provenance=source.get("provenance") or (),
                ),
                "occurrence_role": "repeat_omitted",
                "gate_reason": _gate_reason(rng, document_format),
                "repeat_of": source["seed_id"],
                # Không neo occurrence được gán vào vị trí đầu: xem REPEAT_OMITTED_FIRST_RATE.
                "order_hint": (
                    "before_primary"
                    if rng.random() < (OMITTED_FIRST_RATE_OVERRIDE or REPEAT_OMITTED_FIRST_RATE)
                    else "after_primary"
                ),
            })
            continue
        # Lặp vẫn được gán: đổi assertion để dạy "cùng surface, assertion khác nhau".
        current = list(source["assertions"])
        options = [name for name in ("isHistorical", "isNegated") if name not in current]
        if source["type"] == "THUỐC":
            options = [name for name in options if name != "isHistorical"]
        if SELF_NEGATED_SURFACE.match(source["text"]):
            # Cùng lý do như ở `_assign_supplement_assertions`: surface tự phủ định rồi.
            options = [name for name in options if name != "isNegated"]
        assertions = [rng.choice(options)] if options and rng.random() < 0.6 else []
        extra.append({
            **_planned_item(
                f"repeat:{index:02d}",
                source["text"],
                source["semantic_type"],
                typ=source["type"],
                assertions=assertions,
                role="occurrence_repeat_labeled",
                provenance=source.get("provenance") or (),
            ),
            "occurrence_role": "repeat_labeled",
            "repeat_of": source["seed_id"],
        })
    return plan + extra


def build_entity_contract(
    case: dict[str, Any],
    brief: dict[str, Any],
    catalog: EntityCatalog,
    entity_seed: int,
) -> dict[str, Any]:
    """Create a closed, provenance-carrying entity plan before asking the LLM to write."""
    gate_design = brief.get("design") == "occurrence_gate_v6"
    wide_lexicon = gate_design
    plan: list[dict[str, Any]] = []
    surfaces: set[str] = set()
    for decision in case["decisions"]:
        semantic_type = _semantic_type_for_decision(case, decision, catalog)
        plan.append(_planned_item(
            f"core:{decision['decision_id']}",
            decision["text"],
            semantic_type,
            occurrence_index=decision["occurrence_index"],
            should_label=decision["should_label"],
            typ=decision.get("type"),
            assertions=decision.get("assertions") or (),
            role="hard_case",
            provenance=case.get("provenance", {}).get("business_sources") or (),
        ))
        surfaces.add(decision["text"])

    case_id = case["case_id"]
    bucket_type_weights: dict[str, float] | None = None
    bucket_assertion_rate = ASSERTION_TARGET_RATE
    if brief.get("scope") == "toàn văn bản nhiều mục":
        target_mentions = max(len(plan), 10)
    elif case_id.startswith(("discourse.drug_example", "discourse.education")):
        # Các case intentional-O cần context diễn ngôn sạch; không trộn entity ngẫu nhiên
        # không liên quan chỉ để tăng mật độ.
        target_mentions = len(plan)
    elif case_id.startswith("type.qualitative_result"):
        target_mentions = max(len(plan), 8)
    elif gate_design:
        # Ngân sách độ dài × 0,82 là độ dài thực đo được của LLM.
        low_chars, high_chars = brief.get("length_chars", [900, 1550])
        expected_chars = 0.82 * (low_chars + high_chars) / 2
        low_yield = bool((case.get("coverage_profile") or {}).get("low_yield"))
        density_count, bucket_type_weights, bucket_assertion_rate = _density_target(
            catalog.rng, low_yield, expected_chars
        )
        target_mentions = max(len(plan), density_count)
    else:
        count_range = brief.get("entity_count_range")
        if count_range:
            low, high = count_range
            target_mentions = max(len(plan), catalog.rng.randint(low, high))
        else:
            target_mentions = max(len(plan), 5)

    type_mix = list(dict.fromkeys(brief.get("recommended_type_mix") or TYPES))
    catalog.rng.shuffle(type_mix)
    # Cặp tên xét nghiệm + kết quả cần hai chỗ trống cùng lúc; nếu để vòng lặp tự bốc thì
    # KẾT_QUẢ_XÉT_NGHIỆM luôn bị bỏ ở cuối và tụt còn ~3% (gold 9,9%). Đặt trước một cặp.
    if (
        gate_design
        and "KẾT_QUẢ_XÉT_NGHIỆM" in type_mix
        and target_mentions - len(plan) >= 2
        and catalog.rng.random() < 0.85
    ):
        pair = catalog.sample_lab_pair("ket_qua")
        # Vế phải kiểu "15" hoặc "2" là số trần: word-bounded match sẽ đụng mọi con số khác
        # trong văn bản và đếm occurrence sai. Chỉ nhận kết quả có đơn vị hoặc chữ.
        if pair and not all(_gate_surface_safe(item.text) for item in pair):
            pair = None
        if pair and not any(_surface_conflicts(item.text, surfaces) for item in pair):
            for index, entity in enumerate(pair, 1):
                plan.append(_planned_item(
                    f"labpair:{index:02d}",
                    entity.text,
                    entity.type,
                    typ=entity.type,
                    role="supplemental_random",
                    provenance=entity.provenance,
                ))
                surfaces.add(entity.text)
    if gate_design:
        # Cân theo phân phối gold thay vì chia đều: mẻ v1 lệch THUỐC 19,2% (gold 9,9%) và
        # TÊN_XÉT_NGHIỆM 21,5% (gold 15,4%) vì thuật toán luôn bù type ít nhất.
        type_mix = _weighted_type_mix(type_mix, catalog.rng)
    semantic_counts = Counter(item["semantic_type"] for item in plan)
    supplement_index = 0
    attempts = 0
    while len(plan) < target_mentions and attempts < 200:
        attempts += 1
        if gate_design:
            # Bốc theo trọng số gold thay vì luôn bù type ít nhất, nếu không mọi type trong
            # mix sẽ về xấp xỉ bằng nhau và THUỐC/TÊN_XN bị thổi lên gấp đôi gold.
            weights_table = bucket_type_weights or SAMPLING_TYPE_WEIGHTS
            typ = catalog.rng.choices(
                type_mix,
                weights=[max(weights_table.get(item, 0.1), 0.01) for item in type_mix],
                k=1,
            )[0]
        else:
            typ = min(
                type_mix,
                key=lambda candidate: (
                    semantic_counts[candidate],
                    type_mix.index(candidate),
                ),
            )
        sampled = []
        if typ == "KẾT_QUẢ_XÉT_NGHIỆM":
            pair = catalog.sample_lab_pair("ket_qua")
            if pair and target_mentions - len(plan) >= 2:
                sampled.extend(pair)
            else:
                semantic_counts[typ] += 1
                continue
        else:
            # Từ vựng: mẻ v1 chỉ bốc trong các surface có mặt ở gt2 nên THUỐC còn 69 và
            # CHẨN_ĐOÁN 137 surface duy nhất trên 3.000 draft. Nới sang toàn bộ tier A có
            # từ hai nguồn độc lập (đúng mức A/B của DATASET_CONTRACT §2) để gấp đôi kho.
            sampled.append(
                catalog.sample(
                    typ,
                    exclude=surfaces,
                    require_in_corpus=not wide_lexicon,
                    required_sources=(
                        None if wide_lexicon else {"gt2", "nhãn tay:gt2"}
                    ),
                    exclude_flagged=True,
                    min_source_count=2,
                )
            )

        for entity in sampled:
            if len(plan) >= target_mentions:
                break
            if _surface_conflicts(entity.text, surfaces):
                continue
            if not _pilot_surface_safe(entity.text):
                continue
            if gate_design and not _gate_surface_safe(entity.text):
                continue
            supplement_index += 1
            plan.append(_planned_item(
                f"supplement:{supplement_index:02d}",
                entity.text,
                entity.type,
                typ=entity.type,
                role="supplemental_random",
                provenance=entity.provenance,
            ))
            surfaces.add(entity.text)
            semantic_counts[entity.type] += 1
    if len(plan) < target_mentions:
        raise RuntimeError(
            f"Chỉ lập được {len(plan)}/{target_mentions} entity cho {case_id}"
        )

    if gate_design:
        low_yield = bool((case.get("coverage_profile") or {}).get("low_yield"))
        if low_yield:
            # Tài liệu tư vấn/giáo dục phải DÀY thuật ngữ mà THƯA nhãn; nếu chỉ giảm số
            # entity thì ta chỉ được văn bản nhạt, không phải bài toán gate diễn ngôn.
            _plan_intentional_o(
                plan, catalog, surfaces,
                catalog.rng.randint(3, 6),
                list(brief.get("recommended_type_mix") or TYPES),
            )
        _assign_supplement_assertions(
            plan, catalog.rng, _discourse_assertion_bias(case, brief),
            bucket_assertion_rate,
        )
        plan = _plan_repeat_occurrences(
            plan, catalog.rng, brief.get("document_format")
        )
        plan = _plan_covered_occurrence(plan, catalog)

    labeled = sum(item["should_label"] for item in plan)
    intentional_o = len(plan) - labeled
    brief["entity_budget"] = {
        "labeled": [labeled, labeled],
        "intentional_o": [intentional_o, intentional_o],
        "total_medical_mentions_max": len(plan),
        "primary_hard_pattern": 1,
    }
    if gate_design:
        return {
            "schema_version": 2,
            "policy": "occurrence_gate_v1",
            "entity_seed": entity_seed,
            "allow_unplanned_medical_mentions": False,
            # ⛔ KHÔNG còn "mọi occurrence đã lên kế hoạch đều phải được gán".
            # Plan liệt kê từng occurrence, mỗi occurrence tự mang quyết định gán/không gán.
            "every_occurrence_decided_independently": True,
            "text_carries_markers": True,
            "entity_plan": plan,
        }
    return {
        "schema_version": 1,
        "policy": "closed_allowlist_v1",
        "entity_seed": entity_seed,
        "allow_unplanned_medical_mentions": False,
        "all_planned_occurrences_required": True,
        "entity_plan": plan,
    }


@lru_cache(maxsize=3)
def _catalog_guard_terms(track: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    by_surface: dict[str, set[str]] = {}
    for typ, entries in load_clean(max_tier="A").items():
        for entry in entries:
            surface = entry["text"].strip().casefold()
            if (
                len(surface) < 3
                or surface in _META_GUARD_TERMS
                or "\n" in surface
                or not entity_allowed(entry, track)
            ):
                continue
            by_surface.setdefault(surface, set()).add(typ)
    return tuple(
        (surface, tuple(sorted(types)))
        for surface, types in sorted(
            by_surface.items(), key=lambda item: (-len(item[0]), item[0])
        )
    )


def _unplanned_medical_mentions(
    text: str,
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = request.get("entity_contract")
    if not contract:
        return []
    allowed_spans = []
    for item in contract["entity_plan"]:
        allowed_spans.extend(_all_exact_occurrences(text, item["text"]))
    brief = request.get("generation_brief") or {}
    ignored_spans = []
    heading = brief.get("heading_text") or brief.get("required_heading")
    if heading:
        ignored_spans.extend(_all_exact_occurrences(text, heading))

    def overlaps_planned_or_heading(start: int, end: int) -> bool:
        return any(
            not (end <= left or start >= right)
            for left, right in (*allowed_spans, *ignored_spans)
        )

    def high_precision(surface: str, types: tuple[str, ...]) -> bool:
        folded_surface = surface.casefold().strip()
        if (
            folded_surface in _META_GUARD_TERMS
            or folded_surface.startswith(_AMBIGUOUS_CONTEXT_PREFIXES)
        ):
            return False
        if len(folded_surface.split()) >= 2:
            return True
        return bool(
            set(types)
            & {
                "CHẨN_ĐOÁN",
                "TÊN_XÉT_NGHIỆM",
                "KẾT_QUẢ_XÉT_NGHIỆM",
                "THUỐC",
            }
        )

    folded = text.casefold()
    found = []
    occupied = []
    for surface, types in _catalog_guard_terms(request["track"]):
        cursor = 0
        while True:
            start = folded.find(surface, cursor)
            if start < 0:
                break
            end = start + len(surface)
            cursor = start + max(1, len(surface))
            if (
                not is_word_bounded(folded, start, end)
                or overlaps_planned_or_heading(start, end)
                or any(left <= start and end <= right for left, right in occupied)
                or not high_precision(surface, types)
            ):
                continue
            found.append({
                "text": text[start:end],
                "position": [start, end],
                "catalog_types": list(types),
                "guard": "tier_A_lexicon",
            })
            occupied.append([start, end])
            if len(found) >= 12:
                return found
    for match in _MEASUREMENT_PATTERN.finditer(text):
        if overlaps_planned_or_heading(*match.span()):
            continue
        found.append({
            "text": match.group(),
            "position": list(match.span()),
            "catalog_types": ["KẾT_QUẢ_XÉT_NGHIỆM"],
            "guard": "measurement_pattern",
        })
        if len(found) >= 12:
            break
    return found


_COVERAGE_PROFILES = (
    ("outpatient_followup", "outpatient_followup_note", "Tái khám", "ghi chú tái khám ngoại trú", ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"]),
    ("admission", "admission_note_excerpt", "Nhập viện", "trích đoạn hồ sơ nhập viện", ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]),
    ("nursing", "nursing_progress_note", "Theo dõi điều dưỡng", "ghi chép điều dưỡng theo ca", ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]),
    ("medication", "medication_reconciliation", "Danh sách thuốc", "đối chiếu thuốc", ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"]),
    ("laboratory", "laboratory_result_form", "Phiếu kết quả", "phiếu xét nghiệm dạng dòng", ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]),
    ("imaging", "imaging_request_note", "Chỉ định chẩn đoán hình ảnh", "phiếu chỉ định/thăm dò", ["TÊN_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "KẾT_QUẢ_XÉT_NGHIỆM", "TRIỆU_CHỨNG"]),
    ("telehealth", "telehealth_consultation", "Trao đổi từ xa", "hỏi đáp y tế từ xa", ["TRIỆU_CHỨNG", "THUỐC", "CHẨN_ĐOÁN"]),
    ("referral", "referral_letter_excerpt", "Giấy chuyển tuyến", "đoạn giấy chuyển tuyến", ["CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "THUỐC"]),
    ("procedure", "procedure_preparation_note", "Chuẩn bị thủ thuật", "ghi chú chuẩn bị thủ thuật", ["TÊN_XÉT_NGHIỆM", "THUỐC", "TRIỆU_CHỨNG"]),
    ("discharge", "discharge_instruction", "Dặn dò ra viện", "hướng dẫn sau ra viện", ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG"]),
)

# Thể loại ÍT NHÃN. Gold Part 3 có 31/100 file dưới 8 entity/1k ký tự và 8/100 dưới 5 —
# phần lớn là tư vấn, giáo dục, hỏi đáp. Nếu vòng quay coverage chỉ có bệnh án thì dataset
# không bao giờ dạy được "đoạn dày thuật ngữ y khoa nhưng gán rất ít".
_LOW_YIELD_PROFILES = (
    ("education", "patient_education_leaflet", "Thông tin dành cho người bệnh",
     "tờ thông tin giáo dục sức khoẻ", ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG"]),
    ("counseling", "medication_counseling_leaflet", "Lưu ý khi dùng thuốc",
     "tờ tư vấn dùng thuốc", ["THUỐC", "CHẨN_ĐOÁN"]),
    ("qa_answer", "patient_question_answer", "Hỏi đáp cùng bác sĩ",
     "hỏi đáp y tế thường thức", ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN"]),
    ("prevention", "prevention_guidance_note", "Hướng dẫn phòng ngừa",
     "hướng dẫn phòng ngừa chung", ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"]),
)


def _natural_heading(value: str | None) -> str | None:
    """Chuẩn hóa heading generator về sentence case; không tạo tiêu đề ALL CAPS."""
    if not value:
        return None
    text = value.strip().rstrip(":")
    if text == text.upper() and any(char.isalpha() for char in text):
        text = text.lower()
        text = text[:1].upper() + text[1:]
    return text


def _coverage_case(index: int) -> dict[str, Any]:
    """Case phổ quát không dùng nghiệp vụ/artefact Part 3; chỉ neo genre và entity plan."""
    low_yield = index % 4 == 3
    if low_yield:
        key, document_format, heading, genre, type_mix = _LOW_YIELD_PROFILES[
            (index // 4) % len(_LOW_YIELD_PROFILES)
        ]
    else:
        pool_index = index - (index // 4)
        key, document_format, heading, genre, type_mix = _COVERAGE_PROFILES[
            pool_index % len(_COVERAGE_PROFILES)
        ]
    return {
        "schema_version": 1,
        "case_id": f"coverage.{key}.{index + 1:03d}",
        "title": f"Broad coverage: {genre}",
        "dimension": "label_decision",
        "risk": "low",
        "status": "confirmed",
        "genre": genre,
        "section_role": "general_clinical_document",
        "discourse_role": "general_knowledge" if low_yield else "patient_fact",
        "text": "Khung sinh dữ liệu phổ quát, không lấy từ tài liệu hay test Part 3.",
        "decisions": [],
        "rules": ["BROAD_COVERAGE_ENTITY_CONTRACT"],
        "provenance": {
            "origin": "paraphrased_rule_instantiation",
            "business_sources": ["dictionary/clean", "dataset_factory/broad_coverage"],
            "part3_exact_donor_used": False,
        },
        "generation_constraints": [
            "Chỉ dùng entity do contract cấp", "Không mô phỏng layout hay donor Part 3"
        ],
        "coverage_profile": {
            "document_format": document_format,
            "heading": heading,
            "type_mix": type_mix,
            "low_yield": low_yield,
        },
    }


def build_generation_brief(
    case: dict[str, Any],
    variant_index: int,
    design: str = "hard_stress_v3",
) -> dict[str, Any]:
    case_id = case["case_id"]
    common = {
        "length_chars": [250, 550],
        "entity_budget": {
            "labeled": [3, 6],
            "intentional_o": [0, 2],
            "total_medical_mentions_max": 8,
            "primary_hard_pattern": 1,
        },
        "variant_index": variant_index,
        "no_tables": True,
        "naturalness": "Không giải thích chính sách annotation trong nội dung.",
        "assertion_policy": {"allow_historical_drug": False},
    }
    if case_id.startswith("coverage."):
        profile = case["coverage_profile"]
        common.update({
            "document_format": profile["document_format"],
            "required_heading": profile["heading"],
            "layout": (
                f"một {case['genre']} tự nhiên; không dùng bảng và không mô phỏng test"
            ),
            "length_chars": [220, 500],
            "recommended_type_mix": profile["type_mix"],
        })
    elif case_id.startswith("discourse.education"):
        common.update({
            "document_format": "patient_education_leaflet",
            "required_heading": "THÔNG TIN DÀNH CHO NGƯỜI BỆNH",
            "layout": "heading + đoạn ngắn hoặc bullet, không có câu meta về gán nhãn",
            "entity_budget": {
                "labeled": [1, 3],
                "intentional_o": [2, 4],
                "total_medical_mentions_max": 8,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "THUỐC"],
        })
    elif case_id.startswith("discourse.drug_example"):
        common.update({
            "document_format": "medication_counseling_leaflet",
            "required_heading": "LƯU Ý KHI DÙNG THUỐC",
            "layout": "heading + bullet tư vấn tự nhiên; thuốc ví dụ nằm trong kiến thức chung",
            "entity_budget": {
                "labeled": [0, 2],
                "intentional_o": [2, 4],
                "total_medical_mentions_max": 7,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["THUỐC", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("occurrence."):
        common.update({
            "document_format": "sectioned_clinical_progress_note",
            "required_heading": "DIỄN BIẾN BỆNH",
            "layout": "heading + hai mốc thời gian, có thể dùng dòng ngắn",
            "length_chars": [180, 420],
            "entity_budget": {
                "labeled": [4, 8],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 10,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"],
        })
    elif case_id.startswith("type.test_to_finding"):
        common.update({
            "document_format": "specialty_examination_report",
            "required_heading": "KHÁM CHUYÊN KHOA",
            "layout": "heading + chỉ định/quan sát/kết luận trên các dòng riêng",
            "length_chars": [170, 420],
            "entity_budget": {
                "labeled": [4, 8],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 10,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": [
                "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "TRIỆU_CHỨNG"
            ],
        })
    elif case_id.startswith("type.test_to_diagnosis"):
        common.update({
            "document_format": "diagnostic_imaging_report",
            "required_heading": "CHẨN ĐOÁN HÌNH ẢNH",
            "layout": "heading + kỹ thuật + mô tả ngắn + kết luận, không dùng bảng",
            "length_chars": [160, 400],
            "entity_budget": {
                "labeled": [3, 7],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 9,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": [
                "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN"
            ],
        })
    elif case_id.startswith("discourse.qa_answer_repeat"):
        common.update({
            "document_format": "patient_question_answer",
            "required_heading": "Hỏi đáp cùng bác sĩ",
            "layout": "một lượt hỏi của người bệnh rồi một lượt trả lời của bác sĩ",
            "length_chars": [220, 520],
            "recommended_type_mix": ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("type.result_standalone"):
        common.update({
            "document_format": "laboratory_result_form",
            "required_heading": "Kết quả xét nghiệm",
            "layout": "câu đọc kết quả cho người bệnh, không nhất thiết nêu tên xét nghiệm",
            "length_chars": [160, 420],
            "recommended_type_mix": ["KẾT_QUẢ_XÉT_NGHIỆM", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("type.qualitative_result"):
        common.update({
            "document_format": "laboratory_result_form",
            "required_heading": "KẾT QUẢ XÉT NGHIỆM",
            "layout": "heading + các dòng tên xét nghiệm: kết quả; không dùng bảng",
            "length_chars": [50, 300],
            "entity_budget": {
                "labeled": [4, 10],
                "intentional_o": [0, 1],
                "total_medical_mentions_max": 11,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"],
        })
    elif case_id.startswith("span.diagnosis_qualifier"):
        common.update({
            "document_format": "discharge_summary_excerpt",
            "required_heading": "CHẨN ĐOÁN RA VIỆN",
            "layout": "heading + chẩn đoán + diễn biến liên quan trên dòng/đoạn riêng",
            "length_chars": [180, 450],
            "entity_budget": {
                "labeled": [4, 9],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 11,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"],
        })
    elif case_id.startswith("span.parenthetical_repeat"):
        common.update({
            "document_format": "clinical_asr_transcript",
            "required_heading": "BẢN GHI LỜI KỂ",
            "layout": (
                "heading + lời kể ASR có một lặp từ tự nhiên; không dùng ngoặc lặp "
                "trong văn bản bệnh án trơn tru"
            ),
            "length_chars": [160, 400],
            "entity_budget": {
                "labeled": [2, 5],
                "intentional_o": [1, 2],
                "total_medical_mentions_max": 7,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("span.split"):
        common.update({
            "document_format": "symptom_list_note",
            "required_heading": "TRIỆU CHỨNG HIỆN TẠI",
            "layout": "heading + bullet/dòng ngắn có các concept độc lập",
            "length_chars": [140, 350],
            "entity_budget": {
                "labeled": [4, 8],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 10,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("assertion.negation"):
        common.update({
            "document_format": "triage_examination_note",
            "required_heading": "KHÁM HIỆN TẠI",
            "layout": "heading + dòng ghi nhận âm tính/dương tính có phạm vi rõ",
            "length_chars": [120, 320],
            "entity_budget": {
                "labeled": [3, 7],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 9,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["TRIỆU_CHỨNG", "CHẨN_ĐOÁN"],
        })
    elif case_id.startswith("assertion.family"):
        common.update({
            "document_format": "family_history_section",
            "required_heading": "TIỀN SỬ GIA ĐÌNH",
            "layout": "heading + người thân cụ thể + một dòng về chính bệnh nhân",
            "length_chars": [150, 380],
            "entity_budget": {
                "labeled": [3, 7],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 9,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"],
        })
    elif case_id.startswith("assertion.historical"):
        common.update({
            "document_format": "past_medical_history_section",
            "required_heading": "TIỀN SỬ BẢN THÂN",
            "layout": "heading + mốc thời gian/đợt bệnh cũ + tình trạng hiện tại",
            "length_chars": [180, 450],
            "entity_budget": {
                "labeled": [4, 8],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 10,
                "primary_hard_pattern": 1,
            },
            "recommended_type_mix": ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC"],
        })
    else:
        common.update({
            "document_format": "sectioned_clinical_note",
            "required_heading": "GHI NHẬN LÂM SÀNG",
            "layout": "heading + trường-giá trị hoặc đoạn ngắn",
            "recommended_type_mix": list(TYPES),
        })
    if design == "full_document_v6":
        base_heading = common.pop("required_heading", None) or "Ghi nhận lâm sàng"
        document_headers = [
            base_heading.rstrip(":"),
            "Diễn biến và ghi nhận",
            "Kế hoạch theo dõi",
        ]
        common.update({
            "design": design,
            "scope": "toàn văn bản nhiều mục",
            "length_chars": [900, 1500],
            "header_mode": "explicit_title",
            "heading_text": document_headers[0],
            "section_path_policy": f"[{document_headers[0]!r}]",
            "document_headers": document_headers,
            "section_count": [3, 5],
            "layout": (
                common["layout"]
                + "; viết một văn bản hoàn chỉnh 3-5 mục, mỗi mục có vai trò riêng, "
                "cùng một chủ thể và timeline nhất quán"
            ),
            "density_policy": (
                "Chỉ dùng entity trong entity_contract trên toàn văn bản. "
                "Dùng câu nối phi-y-khoa để liên kết các mục; không thêm entity y khoa."
            ),
            "closed_entity_allowlist": True,
        })
    elif design in {"long_section_v4", "allowlist_section_v5", "occurrence_gate_v6"}:
        header_layout = {
            "discourse.education_selective.001": (
                "explicit_title", "Thông tin dành cho người bệnh"
            ),
            "discourse.drug_example.001": ("inline", "Lưu ý khi dùng thuốc:"),
            "occurrence.independent_assertion.001": ("none", None),
            "type.test_to_finding.001": ("explicit_title", "Khám chuyên khoa mắt"),
            "type.test_to_diagnosis.001": ("inline", "Chẩn đoán hình ảnh:"),
            "type.qualitative_result.001": ("explicit_title", "Phiếu kết quả xét nghiệm"),
            "span.diagnosis_qualifier.001": ("explicit_title", "Chẩn đoán ra viện"),
            "span.parenthetical_repeat.001": ("none", None),
            "span.split_independent_concepts.001": ("inline", "Triệu chứng hiện tại:"),
            "assertion.negation_scope.001": ("none", None),
            "assertion.family_owner.001": ("inline", "Tiền sử gia đình:"),
            "assertion.historical_disease.001": ("explicit_title", "Tiền sử bản thân"),
        }
        base_mode, base_heading = header_layout.get(
            case_id,
            ("explicit_title", common.get("required_heading") or "Ghi nhận lâm sàng"),
        )
        base_heading = _natural_heading(base_heading)
        # Mỗi case có cả biến thể không header, inline và title để model không
        # overfit vào một layout cố định. Năm surface form lặp lại sau mỗi 5 variant.
        layout_variant = variant_index % 5
        if layout_variant in {1, 4}:
            mode, heading = "none", None
        elif layout_variant == 2:
            mode = "inline"
            heading = (base_heading or "Ghi nhận lâm sàng").rstrip(":") + ":"
        elif layout_variant == 3:
            mode = "explicit_title"
            heading = (base_heading or "Ghi nhận lâm sàng").rstrip(":")
        else:
            mode, heading = base_mode, base_heading
        common.pop("required_heading", None)
        common.update({
            "design": design,
            "scope": "một mục/section hoàn chỉnh, không phải toàn tài liệu",
            "length_chars": [500, 950],
            "header_mode": mode,
            "heading_text": heading,
            "section_path_policy": (
                "[]" if mode == "none" else f"[{heading!r}]"
            ),
            "layout": (
                common["layout"]
                + "; mở rộng thành một mục tự nhiên, nhất quán chủ thể và timeline"
            ),
            "surface_form": (
                "đoạn văn tự nhiên"
                if layout_variant == 0
                else "các dòng trường-giá trị ngắn"
                if layout_variant == 1
                else "mục inline xen câu ngắn"
                if layout_variant == 2
                else "tiêu đề và hai đoạn ngắn"
                if layout_variant == 3
                else "bullet hoặc dòng ghi chép cô đọng"
            ),
            "entity_budget": {
                "labeled": [5, 10],
                "intentional_o": [0, 3],
                "total_medical_mentions_max": 13,
                "primary_hard_pattern": 1,
            },
            "density_policy": (
                "Chỉ một hard pattern. Entity phụ phải phát sinh tự nhiên; "
                "không liệt kê thêm bệnh/thuốc/xét nghiệm để đạt quota."
            ),
        })
        if case_id.startswith("discourse.education"):
            common["entity_budget"] = {
                "labeled": [3, 6],
                "intentional_o": [2, 4],
                "total_medical_mentions_max": 10,
                "primary_hard_pattern": 1,
            }
        elif case_id.startswith("discourse.drug_example"):
            common["entity_budget"] = {
                "labeled": [1, 4],
                "intentional_o": [2, 4],
                "total_medical_mentions_max": 9,
                "primary_hard_pattern": 1,
            }
        elif case_id.startswith("type.qualitative_result"):
            common["entity_budget"] = {
                "labeled": [8, 14],
                "intentional_o": [0, 2],
                "total_medical_mentions_max": 16,
                "primary_hard_pattern": 1,
            }
        if design in {"allowlist_section_v5", "occurrence_gate_v6"}:
            common.update({
                "design": design,
                "length_chars": (
                    [220, 400]
                    if case_id.startswith("discourse.drug_example")
                    else [250, 450]
                    if case_id.startswith("discourse.education")
                    else [500, 1200]
                ),
                "density_policy": (
                    "Chỉ dùng entity trong entity_contract. Viết một section dài gồm 2-4 đoạn "
                    "hoặc các dòng cùng một mục; không lặp lại entity ở câu kết."
                ),
                "closed_entity_allowlist": True,
            })
            if not case_id.startswith(("discourse.drug_example", "discourse.education")):
                common["entity_count_range"] = [8, 15]
        if design == "occurrence_gate_v6":
            # Độ dài: cửa sổ inference 180 từ ≈ 868 ký tự (median trên bộ test). Draft phải
            # lấp trọn một cửa sổ và tràn sang cửa sổ sau, nếu không model không bao giờ
            # thấy phần overlap. LLM viết bằng ~0,82 ngân sách nên đặt mean 1.225 để ra ~1.000.
            # Số entity phải đi kèm độ dài để giữ mật độ gold 13,3/1k:
            # 13,3 × 1.000/1.000 ≈ 13 entity, trừ ~0,8 do repeat_labeled tự thêm.
            if not case_id.startswith(("discourse.drug_example", "discourse.education")):
                common["entity_count_range"] = [10, 16]
                common["length_chars"] = [900, 1550]
            common.update({
                "occurrence_gate": {
                    "enabled": True,
                    "marked_text_required": True,
                    "note": (
                        "entity_plan liệt kê TỪNG occurrence; occurrence repeat_omitted phải "
                        "có mặt trong văn bản nhưng KHÔNG được bọc marker"
                    ),
                },
                "assertion_policy": {
                    "allow_historical_drug": False,
                    "cue_required": True,
                    "no_cue_without_assertion": True,
                },
                "density_policy": (
                    "Chỉ dùng entity trong entity_contract. Viết đủ dài và tự nhiên: phần văn "
                    "xuôi phi-entity phải chiếm phần lớn độ dài; không dồn entity liên tiếp."
                ),
            })
    elif design != "hard_stress_v3":
        raise ValueError(f"generation design không hợp lệ: {design}")
    else:
        common["design"] = design
        common["header_mode"] = "explicit_upper"
        common["heading_text"] = common.get("required_heading")
    return common


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Checkpoint JSONL an toàn: dừng giữa lúc ghi không làm hỏng kết quả trước đó."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_jsonl(temporary, rows)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def _read_evidence(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("examples", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evidence/few-shot phải là list khác rỗng")
    if len(rows) > 10:
        raise ValueError("Micro-pilot chỉ nhận tối đa 10 evidence/few-shot đã tuyển chọn")
    if len(json.dumps(rows, ensure_ascii=False)) > 20_000:
        raise ValueError("Evidence/few-shot vượt 20.000 ký tự; phải tuyển chọn lại")
    return rows


def _approval(casebook: Path, approval_path: Path) -> dict[str, Any]:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("stage") != "casebook_approved_for_micro_pilot":
        raise RuntimeError("Approval không phải approval của casebook")
    if approval.get("casebook_sha256") != sha256_file(casebook):
        raise RuntimeError("Casebook checksum không khớp approval")
    return approval


def prepare(
    casebook: Path,
    approval_path: Path,
    out_dir: Path,
    track: str,
    variants_per_case: int,
    case_ids: set[str] | None,
    evidence_path: Path | None,
    design: str = "hard_stress_v3",
    entity_seed: int = DEFAULT_ENTITY_SEED,
    general_coverage_count: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    if variants_per_case < 1 or variants_per_case > MAX_VARIANTS_PER_CASE:
        raise ValueError(f"variants-per-case phải trong [1,{MAX_VARIANTS_PER_CASE}]")
    approval = _approval(casebook, approval_path)
    cases = load_cases(casebook)
    errors = validate_cases(cases)
    if errors:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False, indent=2))
    approved = set(approval["approved_case_ids"])
    selected = [
        case for case in cases
        if case["case_id"] in approved and (not case_ids or case["case_id"] in case_ids)
    ]
    if case_ids:
        missing = case_ids - {case["case_id"] for case in selected}
        if missing:
            raise ValueError(f"Case không tồn tại hoặc chưa được duyệt: {sorted(missing)}")
    selected = [case for case in selected if case["status"] != "open"]
    if general_coverage_count < 0:
        raise ValueError("general-coverage-count không được âm")
    coverage_cases = [_coverage_case(index) for index in range(general_coverage_count)]
    total = len(selected) * variants_per_case + len(coverage_cases)
    if not selected:
        raise ValueError("Không có case đủ điều kiện")
    if total > MAX_PILOT_DRAFTS:
        raise ValueError(
            f"Pilot có {total} draft, vượt trần {MAX_PILOT_DRAFTS}; giảm case hoặc variant"
        )
    if track == "A" and evidence_path is not None:
        raise ValueError("Track A cấm exact evidence/few-shot Part 3")
    if track in {"B", "C"} and evidence_path is None:
        raise ValueError(f"Track {track} phải nhận --track-evidence tuyển chọn và có provenance")

    evidence = _read_evidence(evidence_path) if evidence_path is not None else None
    closed_contract_design = design in {
        "allowlist_section_v5", "full_document_v6", "occurrence_gate_v6",
    }
    catalog = EntityCatalog(
        track,
        entity_seed,
        max_tier=("A" if closed_contract_design else "B"),
    )
    requests = []
    for case_index, case in enumerate(selected):
        for variant in range(variants_per_case):
            # Rải layout ngay cả khi mỗi case chỉ có một variant; nếu luôn truyền 0
            # thì gần như toàn bộ casebook sẽ mang heading dòng riêng.
            brief = build_generation_brief(case, case_index + variant, design)
            brief["variant_index"] = variant
            entity_contract = (
                build_entity_contract(case, brief, catalog, entity_seed)
                if closed_contract_design
                else None
            )
            requests.append({
                "request_id": f"{track}:{case['case_id']}:v{variant + 1}",
                "track": track,
                "variant_index": variant,
                "case": materialize_case(case),
                "generation_brief": brief,
                **({"entity_contract": entity_contract} if entity_contract else {}),
            })
    for coverage_index, case in enumerate(coverage_cases):
        # Broad coverage cũng luân phiên title / không heading / inline, thay vì
        # mọi mẫu ngẫu nhiên luôn có cùng một kiểu đề mục.
        brief = build_generation_brief(case, coverage_index, design)
        entity_contract = (
            build_entity_contract(case, brief, catalog, entity_seed)
            if closed_contract_design
            else None
        )
        requests.append({
            "request_id": f"{track}:{case['case_id']}:v1",
            "track": track,
            "variant_index": 0,
            "case": materialize_case(case),
            "generation_brief": brief,
            **({"entity_contract": entity_contract} if entity_contract else {}),
        })
    # Requests là hàm thuần của (casebook, approval, tham số, seed) nên resume chỉ cần
    # so khớp nguyên văn: trùng thì giữ nguyên draft đã sinh, lệch thì dừng để không
    # trộn hai batch khác nhau vào cùng thư mục.
    requests_path = out_dir / "requests.jsonl"
    reused = False
    if out_dir.exists():
        if not resume:
            raise FileExistsError(
                f"{out_dir} đã tồn tại. Dùng lệnh pipeline (long-section-review / "
                "full-document-review) để resume, hoặc chọn thư mục khác."
            )
        if not requests_path.exists():
            raise RuntimeError(f"{out_dir} tồn tại nhưng thiếu requests.jsonl; không resume được")
        if _read_jsonl(requests_path) != requests:
            raise RuntimeError(
                "requests.jsonl hiện có khác với requests vừa dựng (casebook/seed/tham số đã "
                "thay đổi). Đổi --pilot-out sang thư mục mới thay vì resume."
            )
        reused = True
    out_dir.mkdir(parents=True, exist_ok=True)
    if evidence is not None:
        (out_dir / "track_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not reused:
        _write_jsonl(requests_path, requests)
    manifest = {
        "schema_version": 1,
        "stage": (
            "review_batch_prepared"
            if total > 60
            else "micro_pilot_prepared"
        ),
        "track": track,
        "casebook_path": str(casebook),
        "casebook_sha256": sha256_file(casebook),
        "casebook_approval_path": str(approval_path),
        "casebook_approval_sha256": sha256_file(approval_path),
        "request_count": len(requests),
        "case_count": len(selected),
        "general_coverage_count": len(coverage_cases),
        "variants_per_case": variants_per_case,
        "generation_design": design,
        "entity_seed": entity_seed if closed_contract_design else None,
        "entity_contract_policy": (
            None if not closed_contract_design
            else "occurrence_gate_v1" if design == "occurrence_gate_v6"
            else "closed_allowlist_v1"
        ),
        "clean_manifest_path": (
            str(CLEAN_MANIFEST) if closed_contract_design else None
        ),
        "clean_manifest_sha256": (
            sha256_file(CLEAN_MANIFEST)
            if closed_contract_design
            else None
        ),
        "exact_part3_evidence_used": evidence_path is not None,
        "track_evidence_source": str(evidence_path) if evidence_path else None,
        "track_evidence_sha256": sha256_file(evidence_path) if evidence_path else None,
        "llm_called": False,
        "resumed_existing_requests": reused,
        "next_stage_runs_automatically": False,
        "does_not_authorize": ["dataset_export", "training"],
    }
    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def full_document_review(
    casebook: Path,
    approval_path: Path,
    pilot_dir: Path,
    app_dir: Path,
    track: str,
    variants_per_case: int,
    general_coverage_count: int,
    entity_seed: int,
    timeout: float,
    attempts: int,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Một lệnh chỉ tạo draft văn bản hoàn chỉnh và export sang app để review."""
    manifest = prepare(
        casebook, approval_path, pilot_dir, track, variants_per_case, None, None,
        "full_document_v6", entity_seed, general_coverage_count, resume=True,
    )
    generation = generate(pilot_dir, timeout, attempts, concurrency)
    audit_report = audit(pilot_dir)
    if not audit_report["valid"]:
        raise RuntimeError("Audit draft thất bại; không export sang giao diện review")
    app_manifest = export_app(pilot_dir, app_dir, replace_unreviewed=True)
    return {
        "stage": "full_document_review_ready",
        "prepared": manifest["request_count"],
        "generated_valid": generation["mechanically_valid"],
        "rejected_or_failed": generation["rejected_or_failed"],
        "audit_valid": audit_report["valid"],
        "app_dir": str(app_dir),
        "app_files": app_manifest["files"],
        "is_training_dataset": False,
    }


def section_review(
    casebook: Path,
    approval_path: Path,
    pilot_dir: Path,
    app_dir: Path,
    variants_per_case: int,
    general_coverage_count: int,
    entity_seed: int,
    timeout: float,
    attempts: int,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Một lệnh tạo các section dài, audit và export sang annotation.app để review."""
    manifest = prepare(
        casebook, approval_path, pilot_dir, "A", variants_per_case, None, None,
        "allowlist_section_v5", entity_seed, general_coverage_count, resume=True,
    )
    generation = generate(pilot_dir, timeout, attempts, concurrency)
    audit_report = audit(pilot_dir)
    if not audit_report["valid"]:
        raise RuntimeError("Audit draft thất bại; không export sang giao diện review")
    app_manifest = export_app(pilot_dir, app_dir, replace_unreviewed=True)
    return {
        "stage": "long_section_review_ready",
        "prepared": manifest["request_count"],
        "generated_valid": generation["mechanically_valid"],
        "rejected_or_failed": generation["rejected_or_failed"],
        "audit_valid": audit_report["valid"],
        "app_dir": str(app_dir),
        "app_files": app_manifest["files"],
        "is_training_dataset": False,
    }


def _system_prompt_for(request: dict[str, Any]) -> str:
    """Design occurrence_gate_v6 dùng prompt marker; các design cũ giữ prompt cũ."""
    contract = request.get("entity_contract") or {}
    if contract.get("policy") == "occurrence_gate_v1":
        return SYSTEM_PROMPT_GATE
    return SYSTEM_PROMPT


def _request_key(request: dict[str, Any], evidence: Any, model: str) -> str:
    payload = {
        "request": request, "evidence": evidence, "model": model,
        "prompt": _system_prompt_for(request),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _gate_positions(
    text: str,
    marks: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> tuple[dict[str, list[int]], list[str]]:
    """Ghép marker với plan và định vị các occurrence CỐ Ý KHÔNG GÁN.

    Occurrence dương tính lấy vị trí từ marker (không đếm thủ công). Occurrence âm tính là
    occurrence cùng surface, word-bounded, KHÔNG nằm trong bất kỳ marker nào.
    """
    errors: list[str] = []
    positions: dict[str, list[int]] = {}
    marks_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mark in marks:
        marks_by_seed[mark["seed_id"]].append(mark)

    labeled_seeds = {item["seed_id"] for item in plan if item["should_label"]}
    for seed_id, found in marks_by_seed.items():
        if seed_id not in labeled_seeds:
            errors.append(f"marker id {seed_id!r} không phải occurrence được gán trong plan")
        if len(found) > 1:
            errors.append(f"marker id {seed_id!r} xuất hiện {len(found)} lần, phải đúng 1")
    for item in plan:
        if item["should_label"] and item["seed_id"] not in marks_by_seed:
            errors.append(f"thiếu marker cho occurrence được gán {item['seed_id']!r}")

    marked_spans = [tuple(mark["position"]) for mark in marks]
    for mark in marks:
        item = next((row for row in plan if row["seed_id"] == mark["seed_id"]), None)
        if item is None:
            continue
        if mark["text"] != item["text"]:
            errors.append(
                f"marker {mark['seed_id']!r} bọc {mark['text']!r} khác plan {item['text']!r}"
            )
            continue
        start, end = mark["position"]
        if not is_word_bounded(text, start, end):
            errors.append(f"marker {mark['seed_id']!r} cắt giữa token")
        positions[mark["seed_id"]] = [start, end]

    used: set[tuple[int, int]] = set()
    for item in plan:
        if item["should_label"]:
            continue
        if item.get("occurrence_role") == "covered_by_longer":
            # Ngược với các occurrence không gán khác: cái này PHẢI nằm trong marker của
            # entity dài, và không được xuất hiện thêm ở đâu khác.
            host_item = next(
                (row for row in plan if row["seed_id"] == item.get("covered_by")), None
            )
            # Entity dài có thể được nhắc lại (repeat_labeled) nên mọi lần xuất hiện của nó
            # đều là chỗ hợp lệ để chứa surface ngắn.
            hosts = (
                _all_exact_occurrences(text, host_item["text"]) if host_item else []
            )
            occurrences = _all_exact_occurrences(text, item["text"])
            inside = [
                (start, end) for start, end in occurrences
                if any(host[0] <= start and end <= host[1] for host in hosts)
            ]
            outside = [
                (start, end) for start, end in occurrences
                if not any(host[0] <= start and end <= host[1] for host in hosts)
            ]
            if not inside:
                errors.append(
                    f"occurrence {item['seed_id']!r} phải nằm trong entity dài "
                    f"{item.get('covered_by')!r} nhưng không tìm thấy"
                )
            else:
                positions[item["seed_id"]] = [inside[0][0], inside[0][1]]
            if outside:
                errors.append(
                    f"occurrence {item['seed_id']!r} ({item['text']!r}) chỉ được xuất hiện "
                    "bên trong entity dài, không được nhắc riêng ở chỗ khác"
                )
            continue
        candidates = [
            (start, end) for start, end in _all_exact_occurrences(text, item["text"])
            if not any(
                mark_start <= start and end <= mark_end
                for mark_start, mark_end in marked_spans
            )
            and (start, end) not in used
        ]
        if not candidates:
            role = item.get("occurrence_role", "intentional_o")
            errors.append(
                f"occurrence không gán {item['seed_id']!r} ({role}) thiếu trong văn bản: "
                f"{item['text']!r} phải xuất hiện thêm một lần KHÔNG bọc marker"
            )
            continue
        chosen = candidates[0]
        used.add(chosen)
        positions[item["seed_id"]] = [chosen[0], chosen[1]]
    return positions, errors


def _assertion_cue_errors(
    text: str,
    surface: str,
    position: list[int],
    assertions: list[str],
    skip_conflict: set[str] | None = None,
) -> list[str]:
    """Nhãn assertion và ngữ cảnh phải cùng một câu chuyện, theo cả hai chiều."""
    support = _cues_support(text, position)
    cues = _cues_before(text, position[0])
    problems = []
    for name in assertions:
        if name not in support:
            problems.append(
                f"assertion {name} không có cue ngay trước occurrence {surface!r}"
            )
    skip = set(skip_conflict or set())
    if "isFamily" in assertions:
        # "bố từng bị X" trước hết là chuyện của người thân; thêm isHistorical hay không là
        # quyết định nghiệp vụ tinh tế, không phải mâu thuẫn cơ học.
        skip.add("isHistorical")
    for name in cues - set(assertions) - skip:
        # isFamily dễ dương tính giả (ví dụ 'bà' trong 'bàng quang' đã chặn bằng ranh giới từ,
        # nhưng 'ông'/'bà' vẫn có thể là đại từ xưng hô) nên chỉ cảnh báo hai loại chắc chắn.
        if name in {"isNegated", "isHistorical"}:
            problems.append(
                f"occurrence {surface!r} đứng sau cue {name} nhưng plan để assertions rỗng"
            )
    return problems


def _repair_assertions(
    text: str,
    plan_by_seed: dict[str, dict[str, Any]],
    mentions: list[dict[str, Any]],
    positions: dict[str, list[int]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Cho nhãn CHẠY THEO VĂN BẢN khi hai bên lệch nhau, thay vì loại cả draft.

    Nếu văn bản viết "không ghi nhận X" thì `isNegated` là nhãn ĐÚNG theo CANONICAL §4.1,
    dù plan để rỗng — sửa nhãn cho khớp văn bản vừa rẻ hơn vừa đúng hơn là bỏ draft.
    Chiều ngược lại cũng vậy: plan ghi isHistorical mà câu không có mốc quá khứ nào thì
    nhãn đúng là rỗng. Mọi lần sửa đều được ghi lại để theo dõi.
    """
    repairs: dict[str, list[str]] = {}
    notes: list[str] = []
    for mention in mentions:
        seed_id = mention.get("seed_id")
        planned = plan_by_seed.get(seed_id)
        position = positions.get(seed_id)
        if not planned or not position or not planned.get("should_label"):
            continue
        typ = planned.get("type")
        if typ not in ASSERTION_TYPES:
            continue
        current = list(planned.get("assertions") or [])
        support = _cues_support(text, position)
        conflict = _cues_before(text, position[0])
        if typ == "THUỐC":
            conflict = conflict - {"isHistorical"}
        if "isFamily" in current:
            conflict = conflict - {"isHistorical"}
        updated = [name for name in current if name in support]
        for name in sorted(conflict - set(updated)):
            updated.append(name)
        if updated != current:
            repairs[seed_id] = updated
            notes.append(
                f"assertion_repair:{seed_id}:{current or '[]'}→{updated or '[]'}"
            )
    return repairs, notes


def _generated_errors(request: dict[str, Any], data: dict[str, Any]) -> tuple[list[str], dict]:
    errors: list[str] = []
    quality_flags: list[str] = []
    contract = request.get("entity_contract") or {}
    gate_mode = contract.get("policy") == "occurrence_gate_v1"
    marks: list[dict[str, Any]] = []
    if gate_mode:
        marked_text = data.get("marked_text")
        if not isinstance(marked_text, str) or not marked_text:
            errors.append("thiếu marked_text cho design occurrence_gate_v6")
            data = {**data, "text": data.get("text") or ""}
        else:
            raw, marks, marker_errors = _parse_marked_text(marked_text)
            errors.extend(marker_errors)
            data = {**data, "text": raw}
    text = data.get("text")
    if data.get("source_case_id") != request["case"]["case_id"]:
        errors.append("source_case_id không khớp")
    if not isinstance(text, str) or not (20 <= len(text) <= 1500):
        errors.append("text phải dài 20-1500 ký tự")
        text = text if isinstance(text, str) else ""
    brief = request.get("generation_brief")
    document_format = data.get("document_format")
    section_path = data.get("section_path")
    if brief:
        low, high = brief["length_chars"]
        if not low <= len(text) <= high:
            quality_flags.append(f"length {len(text)} outside_budget [{low},{high}]")
        # Bố cục: bộ test có 31 dòng/tài liệu và 50% dòng ngắn; mẻ 5.000 chỉ có 6 dòng và 7%.
        # Dưới ngưỡng tối thiểu thì báo LỖI để vòng sửa chữa viết lại, lệch nhẹ thì chỉ gắn cờ.
        lines = [line for line in text.split("\n") if line.strip()]
        short = sum(1 for line in lines if len(line.strip()) <= 40)
        short_ratio = short / max(len(lines), 1)
        # Yêu cầu theo ĐỘ DÀI: bộ test có ~1 dòng mỗi 60 ký tự. Chỉ áp cho draft đủ dài để nói
        # được về bố cục; trích đoạn ngắn không bị đòi phải có nhiều dòng.
        required_lines = min(MIN_LINES_PER_RECORD, max(3, len(text) // 90))
        if len(text) >= LAYOUT_CHECK_MIN_CHARS and (
            len(lines) < required_lines or short_ratio < MIN_SHORT_LINE_RATIO
        ):
            errors.append(
                f"bố cục quá giống văn xuôi: {len(lines)} dòng "
                f"(tối thiểu {required_lines}), {short_ratio:.0%} dòng ngắn "
                f"(tối thiểu {MIN_SHORT_LINE_RATIO:.0%}) — bệnh án thật là danh sách và bảng"
            )
        elif len(text) >= LAYOUT_CHECK_MIN_CHARS and (len(lines) < 24 or short_ratio < 0.45):
            quality_flags.append(f"layout lines={len(lines)} short={short_ratio:.0%}")
        lowered = text.casefold()
        for pattern in META_TEXT_PATTERNS:
            if pattern in lowered:
                errors.append(f"meta_text:{pattern}")
        if document_format != brief["document_format"]:
            errors.append(
                f"document_format phải là {brief['document_format']}, nhận {document_format}"
            )
        if not isinstance(section_path, list):
            errors.append("section_path phải là list")
            section_path = []
        header_mode = brief.get("header_mode", "explicit_upper")
        heading = brief.get("heading_text") or brief.get("required_heading")
        if header_mode == "none":
            if section_path:
                errors.append("header_mode=none yêu cầu section_path=[]")
            first_nonempty = next(
                (line.strip() for line in text.splitlines() if line.strip()), ""
            )
            if (
                len(first_nonempty) >= 4
                and first_nonempty == first_nonempty.upper()
                and any(char.isalpha() for char in first_nonempty)
            ):
                errors.append("header_mode=none nhưng dòng đầu giống header in hoa")
        else:
            if not heading or heading not in text:
                errors.append(f"thiếu heading trong text: {heading}")
            if not heading or heading not in section_path:
                errors.append(f"thiếu heading trong section_path: {heading}")
        if re.search(r"(?m)^\s*\|.*\|.*\|\s*$", text) or re.search(
            r"(?m)^\s*\|?\s*:?-{3,}", text
        ):
            errors.append("table_not_allowed")
    mentions = data.get("mentions")
    if not isinstance(mentions, list) or not mentions:
        errors.append("mentions phải là list khác rỗng")
        mentions = []
    entity_contract = request.get("entity_contract")
    expected_by_seed: dict[str, dict[str, Any]] = {}
    if entity_contract:
        expected_by_seed = {
            item["seed_id"]: item for item in entity_contract["entity_plan"]
        }
        actual_seed_ids = [mention.get("seed_id") for mention in mentions]
        duplicates = sorted(
            seed_id for seed_id, count in Counter(actual_seed_ids).items()
            if seed_id is not None and count > 1
        )
        missing = sorted(set(expected_by_seed) - set(actual_seed_ids))
        unexpected = sorted(
            seed_id for seed_id in set(actual_seed_ids) - set(expected_by_seed)
            if seed_id is not None
        )
        if any(seed_id is None for seed_id in actual_seed_ids):
            errors.append("entity_contract: mention thiếu seed_id")
        if duplicates:
            errors.append(f"entity_contract: seed_id trùng {duplicates}")
        if missing:
            errors.append(f"entity_contract: thiếu seed_id {missing}")
        if unexpected:
            errors.append(f"entity_contract: seed_id ngoài kế hoạch {unexpected}")
    normalized = []
    seen: set[tuple[str, int]] = set()
    seen_seeds: set[str] = set()
    gate_positions: dict[str, list[int]] = {}
    assertion_repairs: dict[str, list[str]] = {}
    if gate_mode and entity_contract:
        gate_positions, gate_errors = _gate_positions(
            text, marks, entity_contract["entity_plan"]
        )
        errors.extend(gate_errors)
        assertion_repairs, repair_notes = _repair_assertions(
            text, expected_by_seed, mentions, gate_positions
        )
        quality_flags.extend(repair_notes)
    labeled_positions = []
    for index, mention in enumerate(mentions):
        prefix = f"mention[{index}]"
        surface = mention.get("text")
        occurrence_index = mention.get("occurrence_index")
        seed_id = mention.get("seed_id")
        expected = expected_by_seed.get(seed_id) if entity_contract else None
        if entity_contract and expected:
            compared = (
                ("text", "should_label", "type")
                if gate_mode
                else ("text", "occurrence_index", "should_label", "type")
            )
            for field in compared:
                if mention.get(field) != expected.get(field):
                    errors.append(
                        f"{prefix}: {field}={mention.get(field)!r} "
                        f"khác entity_plan={expected.get(field)!r}"
                    )
            accepted_assertions = assertion_repairs.get(
                seed_id, list(expected["assertions"])
            )
            if list(mention.get("assertions") or []) != accepted_assertions and (
                seed_id not in assertion_repairs
            ):
                errors.append(
                    f"{prefix}: assertions khác entity_plan "
                    f"{expected['assertions']}"
                )
        if not isinstance(surface, str) or not surface:
            errors.append(f"{prefix}: text rỗng")
            continue
        if gate_mode:
            if seed_id in seen_seeds:
                errors.append(f"{prefix}: seed_id lặp trong mentions")
            seen_seeds.add(seed_id)
            position = gate_positions.get(seed_id)
            if position is None:
                # lỗi đã được _gate_positions ghi nhận
                continue
        else:
            if not isinstance(occurrence_index, int):
                errors.append(f"{prefix}: occurrence_index không phải int")
                continue
            key = (surface, occurrence_index)
            if key in seen:
                errors.append(f"{prefix}: trùng surface/occurrence")
            seen.add(key)
            try:
                position = occurrence_position(text, surface, occurrence_index)
            except ValueError as exc:
                errors.append(f"{prefix}: {exc}")
                continue
        provided_position = mention.get("position")
        if provided_position is not None and provided_position != position:
            errors.append(
                f"{prefix}: position lưu {provided_position} lệch position tính lại {position}"
            )
        should_label = mention.get("should_label")
        typ = mention.get("type")
        assertions = mention.get("assertions")
        if gate_mode and seed_id in assertion_repairs:
            assertions = list(assertion_repairs[seed_id])
            mention = {**mention, "assertions": assertions}
        if not isinstance(should_label, bool):
            errors.append(f"{prefix}: should_label không phải bool")
        if not isinstance(assertions, list):
            errors.append(f"{prefix}: assertions không phải list")
            assertions = []
        if should_label:
            if typ not in TYPES:
                errors.append(f"{prefix}: type không hợp lệ {typ}")
            if set(assertions) - set(ASSERTIONS):
                errors.append(f"{prefix}: assertion không hợp lệ")
            if assertions and typ not in ASSERTION_TYPES:
                errors.append(f"{prefix}: type {typ} không được có assertion")
            if surface.strip().casefold() in {
                "thuốc", "xét nghiệm", "triệu chứng", "chẩn đoán", "bệnh nhân", "bác sĩ"
            }:
                errors.append(f"{prefix}: danh từ meta đứng riêng không được gán")
            if (
                typ == "THUỐC"
                and "isHistorical" in assertions
                and brief
                and not brief.get("assertion_policy", {}).get("allow_historical_drug", False)
            ):
                errors.append(f"{prefix}: isHistorical cho THUỐC chưa được phép trong core pilot")
            if gate_mode and typ in ASSERTION_TYPES:
                for problem in _assertion_cue_errors(
                    text, surface, position, assertions,
                    # isHistorical cho THUỐC đang bị chặn theo nghiệp vụ (còn OPEN), nên
                    # thuốc nằm trong câu tiền sử mà để rỗng là CỐ Ý, không phải mâu thuẫn.
                    skip_conflict={"isHistorical"} if typ == "THUỐC" else None,
                ):
                    errors.append(f"{prefix}: {problem}")
            labeled_positions.append(position)
        elif typ is not None or assertions:
            errors.append(f"{prefix}: intentional-O phải có type=null/assertions=[]")
        context_evidence = mention.get("context_evidence")
        if gate_mode and isinstance(context_evidence, str):
            # LLM hay trích bằng chứng từ bản CÓ marker; bỏ marker rồi mới đối chiếu RAW.
            context_evidence = _MARKER_RE.sub(r"\2", context_evidence)
        if not context_evidence:
            errors.append(f"{prefix}: thiếu context_evidence")
        elif not _context_evidence_supports_occurrence(
            text, surface, position, context_evidence
        ):
            if gate_mode:
                # Marker đã chỉ đích danh occurrence được gán nên evidence không còn là
                # cơ chế định vị; lệch evidence chỉ đáng cảnh báo, không đáng loại draft.
                quality_flags.append(
                    f"{prefix}: context_evidence không neo đúng occurrence"
                )
            else:
                errors.append(
                    f"{prefix}: context_evidence không neo đúng occurrence đã align"
                )
        if not mention.get("rationale"):
            errors.append(f"{prefix}: thiếu rationale")
        item = dict(mention)
        item["position"] = position
        normalized.append(item)
    labeled_positions.sort()
    for left, right in zip(labeled_positions, labeled_positions[1:]):
        if left[1] > right[0]:
            errors.append(f"labeled span overlap: {left} và {right}")
    if entity_contract:
        by_surface: dict[str, int] = {}
        for item in entity_contract["entity_plan"]:
            if gate_mode:
                if item.get("occurrence_role") == "covered_by_longer":
                    # Đã được tính trong lần xuất hiện của entity dài; không đếm riêng.
                    continue
                # Mỗi dòng plan là MỘT occurrence (gán hoặc không gán). Tổng số lần surface
                # xuất hiện trong văn bản phải bằng đúng số dòng plan của surface đó.
                by_surface[item["text"]] = by_surface.get(item["text"], 0) + 1
            else:
                by_surface[item["text"]] = max(
                    by_surface.get(item["text"], 0),
                    item["occurrence_index"] + 1,
                )
        for surface, expected_count in sorted(by_surface.items()):
            actual_count = len(_all_exact_occurrences(text, surface))
            if actual_count != expected_count:
                errors.append(
                    f"entity_contract: {surface!r} xuất hiện {actual_count}, "
                    f"yêu cầu đúng {expected_count}"
                )
        unplanned = _unplanned_medical_mentions(text, request)
        hard_unplanned = [
            item for item in unplanned
            if item.get("guard") == "measurement_pattern"
        ]
        lexicon_warnings = [
            item for item in unplanned
            if item.get("guard") == "tier_A_lexicon"
        ]
        if hard_unplanned:
            errors.append(
                "entity_contract: phát hiện nội dung y khoa ngoài allowlist: "
                + json.dumps(hard_unplanned, ensure_ascii=False)
            )
        if lexicon_warnings:
            # Từ điển sạch vẫn chứa nhiều surface phụ thuộc ngữ cảnh như
            # "bệnh", "bình thường", "nghỉ ngơi". Dùng chúng làm hard gate
            # khiến draft đúng bị loại hàng loạt. Giữ lại để người review chú ý,
            # còn contract/offset/type/assertion và số đo ngoài plan mới là gate.
            quality_flags.append(
                "possible_unplanned_lexicon_mentions: "
                + json.dumps(lexicon_warnings, ensure_ascii=False)
            )
    if brief:
        labeled_count = sum(bool(item.get("should_label")) for item in normalized)
        intentional_o_count = len(normalized) - labeled_count
        budget = brief["entity_budget"]
        if not budget["labeled"][0] <= labeled_count <= budget["labeled"][1]:
            quality_flags.append(
                f"labeled_count {labeled_count} ngoài ngân sách {budget['labeled']}"
            )
        if not (
            budget["intentional_o"][0]
            <= intentional_o_count
            <= budget["intentional_o"][1]
        ):
            quality_flags.append(
                f"intentional_o_count {intentional_o_count} ngoài ngân sách "
                f"{budget['intentional_o']}"
            )
        if len(normalized) > budget["total_medical_mentions_max"]:
            quality_flags.append(
                f"total_mentions {len(normalized)} > "
                f"{budget['total_medical_mentions_max']}"
            )
    output = {
        "draft_id": request["request_id"],
        **({"marked_text": data.get("marked_text")} if gate_mode else {}),
        "source_case_id": request["case"]["case_id"],
        "track": request["track"],
        "document_format": document_format,
        "section_path": section_path if isinstance(section_path, list) else [],
        "text": text,
        "mentions": normalized,
        "reference_case": request["case"],
        "generation_brief": brief,
        "entity_contract": entity_contract,
        "quality_flags": quality_flags,
        "status": (
            "mechanically_rejected"
            if errors
            else "mechanically_valid_with_quality_flags"
            if quality_flags
            else "mechanically_valid"
        ),
    }
    return errors, output


def generate(
    pilot_dir: Path,
    timeout: float,
    attempts: int,
    concurrency: int = 1,
    correction_rounds: int = 2,
) -> dict[str, Any]:
    if concurrency < 1 or concurrency > MAX_CONCURRENCY:
        raise ValueError(f"concurrency phải trong [1,{MAX_CONCURRENCY}]")
    manifest_path = pilot_dir / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"micro_pilot_prepared", "review_batch_prepared"}:
        raise RuntimeError("Pilot chưa ở stage prepared")
    requests = _read_jsonl(pilot_dir / "requests.jsonl")
    if len(requests) > MAX_PILOT_DRAFTS:
        raise RuntimeError("Vượt hard-limit generation batch")
    evidence = None
    evidence_file = pilot_dir / "track_evidence.json"
    if manifest["track"] == "A" and evidence_file.exists():
        raise RuntimeError("Firewall: Track A có track_evidence")
    if evidence_file.exists():
        evidence = json.loads(evidence_file.read_text(encoding="utf-8"))
        source = Path(manifest["track_evidence_source"])
        if sha256_file(source) != manifest["track_evidence_sha256"]:
            raise RuntimeError("Track evidence nguồn đã thay đổi sau prepare")
    client, model = azure_client(timeout)
    print(
        f"[config] model={model} reasoning_effort=none timeout={timeout:.0f}s "
        f"attempts={attempts} concurrency={concurrency} "
        f"correction_rounds={correction_rounds}",
        flush=True,
    )
    print("[preflight] Azure no-reasoning connection: ", end="", flush=True)
    preflight_started = time.monotonic()
    try:
        preflight_client = client.with_options(timeout=min(timeout, 30), max_retries=0)
        preflight_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": 'Trả đúng JSON {"ok":true}.',
            }],
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
    except Exception as exc:
        print(
            f"FAIL ({time.monotonic() - preflight_started:.1f}s) "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise RuntimeError(
            "Preflight Azure thất bại; chưa gọi bất kỳ request sinh draft nào."
        ) from exc
    print(f"PASS ({time.monotonic() - preflight_started:.1f}s)", flush=True)
    cache_dir = pilot_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)
    drafts_path = pilot_dir / "drafts.jsonl"
    failures_path = pilot_dir / "generation_failures.jsonl"
    journal_path = pilot_dir / "generation_checkpoint.jsonl"
    report_path = pilot_dir / "generation_report.json"
    request_ids = {request["request_id"] for request in requests}
    existing_drafts = _read_jsonl(drafts_path) if drafts_path.exists() else []
    existing_failures = _read_jsonl(failures_path) if failures_path.exists() else []
    drafts_by_id = {
        draft["draft_id"]: draft for draft in existing_drafts
        if draft.get("draft_id") in request_ids
    }
    failures_by_id = {
        failure["request_id"]: failure for failure in existing_failures
        if failure.get("request_id") in request_ids
    }
    if journal_path.exists():
        for event in _read_jsonl(journal_path):
            request_id = event.get("request_id")
            if request_id not in request_ids:
                continue
            if event.get("kind") == "draft" and isinstance(event.get("draft"), dict):
                drafts_by_id[request_id] = event["draft"]
                failures_by_id.pop(request_id, None)
            elif event.get("kind") == "failure" and isinstance(event.get("failure"), dict):
                failures_by_id[request_id] = event["failure"]

    state_lock = threading.Lock()
    journal_lock = threading.Lock()

    def append_journal(kind: str, request_id: str, value: dict[str, Any]) -> None:
        event = {"kind": kind, "request_id": request_id, kind: value}
        with journal_lock, journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def checkpoint(stage: str, compact: bool = False) -> dict[str, Any]:
        with state_lock:
            drafts = [drafts_by_id[item["request_id"]] for item in requests
                      if item["request_id"] in drafts_by_id]
            failures = [failures_by_id[item["request_id"]] for item in requests
                        if item["request_id"] in failures_by_id]
        if compact:
            _write_jsonl_atomic(drafts_path, drafts)
            _write_jsonl_atomic(failures_path, failures)
        report = {
            "stage": stage,
            "model": model,
            "requested": len(requests),
            "mechanically_valid": len(drafts),
            "with_quality_flags": sum(bool(draft.get("quality_flags")) for draft in drafts),
            "rejected_or_failed": len(failures),
            "remaining": len(requests) - len(drafts) - len(failures),
            "resume_supported": True,
            "concurrency": concurrency,
            "human_review_started": False,
            "dataset_records_created": 0,
            "next_stage_runs_automatically": False,
        }
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        return report

    already_valid = len(drafts_by_id)
    if already_valid:
        print(f"[resume] giữ {already_valid} draft PASS đã checkpoint", flush=True)
    checkpoint("llm_drafts_in_progress", compact=True)
    total = len(requests)

    def run_request(index: int, request: dict[str, Any]) -> None:
        request_id = request["request_id"]
        key = _request_key(request, evidence, model)
        cache = cache_dir / f"{key}.json"
        started = time.monotonic()
        notes: list[str] = []
        try:
            if cache.exists():
                data = json.loads(cache.read_text(encoding="utf-8"))
                source = "cache"
            else:
                payload = {
                    "reference_case": request["case"],
                    "variant_index": request["variant_index"],
                    "generation_brief": request.get("generation_brief"),
                    "entity_contract": request.get("entity_contract"),
                    "track_evidence": evidence,
                }
                last_error = None
                for _ in range(attempts):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": _system_prompt_for(request)},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                            ],
                            response_format={"type": "json_object"},
                            reasoning_effort="none",
                        )
                        data = json.loads(response.choices[0].message.content)
                        cache.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        source = "azure"
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")
            errors, draft = _generated_errors(request, data)
            # Sửa hợp đồng có mục tiêu. Đây không phải retry mạng và không dùng reasoning.
            # Hai vòng: vòng 1 sửa lỗi chính, vòng 2 dọn phần còn sót — rẻ hơn nhiều so với
            # vứt cả draft đã tốn một lần gọi. Cache chỉ giữ bản mới nhất.
            for round_index in range(1, correction_rounds + 1):
                if not errors:
                    break
                notes.append(f"correct{round_index}")
                correction_payload = {
                    "reference_case": request["case"],
                    "variant_index": request["variant_index"],
                    "generation_brief": request.get("generation_brief"),
                    "entity_contract": request.get("entity_contract"),
                    "track_evidence": evidence,
                    "validator_feedback": errors,
                    "previous_output": data,
                    "instruction": (
                        "Bản trước KHÔNG hợp lệ. Sửa đúng những lỗi trong validator_feedback "
                        "và trả lại TOÀN BỘ output theo đúng schema. Giữ nguyên phần đã đúng. "
                        "Nhắc lại các ràng buộc hay bị vi phạm: mỗi surface chỉ được xuất hiện "
                        "đúng số lần plan đã cấp (không nhiều hơn); occurrence "
                        "covered_by_longer chỉ tồn tại BÊN TRONG entity dài và không được "
                        "nhắc riêng; mỗi occurrence được gán phải có đúng một marker."
                    ),
                }
                last_error = None
                for _ in range(max(1, attempts)):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": _system_prompt_for(request)},
                                {
                                    "role": "user",
                                    "content": json.dumps(correction_payload, ensure_ascii=False),
                                },
                            ],
                            response_format={"type": "json_object"},
                            reasoning_effort="none",
                        )
                        data = json.loads(response.choices[0].message.content)
                        cache.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        source = "azure_corrected"
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    errors.append(
                        f"correction_call:{type(last_error).__name__}: {last_error}"
                    )
                    break
                errors, draft = _generated_errors(request, data)
            draft["generator"] = {
                "backend": source,
                "model": model,
                "prompt_sha256": hashlib.sha256(
                    _system_prompt_for(request).encode()
                ).hexdigest(),
                "request_key": key,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if errors:
                failure = {"request_id": request_id, "errors": errors, "draft": draft}
                with state_lock:
                    failures_by_id[request_id] = failure
                append_journal("failure", request_id, failure)
            else:
                with state_lock:
                    drafts_by_id[request_id] = draft
                    failures_by_id.pop(request_id, None)
                append_journal("draft", request_id, draft)
            label = (
                "REJECT"
                if errors
                else "PASS+FLAGS"
                if draft.get("quality_flags")
                else "PASS"
            )
        except Exception as exc:
            failure = {
                "request_id": request_id,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            with state_lock:
                failures_by_id[request_id] = failure
            append_journal("failure", request_id, failure)
            label = "ERROR"
        suffix = f" [{', '.join(notes)}]" if notes else ""
        print(
            f"[{index:04d}/{total:04d}] {request_id}: {label}{suffix} "
            f"({time.monotonic() - started:.1f}s)",
            flush=True,
        )

    pending = [
        (index, request) for index, request in enumerate(requests, 1)
        if request["request_id"] not in drafts_by_id
    ]
    print(
        f"[plan] {len(pending)}/{total} request cần gọi, concurrency={concurrency}",
        flush=True,
    )
    done = 0
    try:
        if concurrency == 1:
            for index, request in pending:
                run_request(index, request)
                done += 1
                checkpoint("llm_drafts_in_progress", compact=done % 25 == 0)
        else:
            executor = ThreadPoolExecutor(max_workers=concurrency)
            try:
                futures = [
                    executor.submit(run_request, index, request)
                    for index, request in pending
                ]
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    checkpoint("llm_drafts_in_progress", compact=done % 25 == 0)
            finally:
                # Ctrl-C: huỷ phần chưa chạy, không chờ hết 3.000 request.
                executor.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        report = checkpoint("llm_drafts_interrupted", compact=True)
        print(
            f"[interrupt] đã checkpoint {report['mechanically_valid']} draft PASS; "
            "chạy lại đúng lệnh cũ để tiếp tục.",
            flush=True,
        )
        raise
    return checkpoint("llm_drafts_generated", compact=True)


def audit(pilot_dir: Path) -> dict[str, Any]:
    requests = {row["request_id"]: row for row in _read_jsonl(pilot_dir / "requests.jsonl")}
    drafts = _read_jsonl(pilot_dir / "drafts.jsonl")
    errors = []
    for draft in drafts:
        request = requests.get(draft.get("draft_id"))
        if request is None:
            errors.append({"draft_id": draft.get("draft_id"), "errors": ["request không tồn tại"]})
            continue
        local, normalized = _generated_errors(request, draft)
        if local:
            errors.append({"draft_id": draft.get("draft_id"), "errors": local})
        elif normalized["text"] != draft["text"]:
            errors.append({"draft_id": draft.get("draft_id"), "errors": ["normalized text lệch"]})
    total_chars = sum(len(draft["text"]) for draft in drafts)
    labeled = [
        mention
        for draft in drafts
        for mention in draft.get("mentions") or []
        if mention.get("should_label")
    ]
    intentional_o = [
        mention
        for draft in drafts
        for mention in draft.get("mentions") or []
        if not mention.get("should_label")
    ]
    formats = Counter(
        draft.get("document_format") for draft in drafts if draft.get("document_format")
    )
    quality_flags = Counter(
        flag
        for draft in drafts
        for flag in draft.get("quality_flags") or []
    )
    report = {
        "stage": "mechanical_audit",
        "valid": not errors,
        "draft_count": len(drafts),
        "errors": errors,
        "distribution": {
            "characters": total_chars,
            "labeled_mentions": len(labeled),
            "intentional_o_mentions": len(intentional_o),
            "labeled_per_1000_chars": round(
                1000 * len(labeled) / max(1, total_chars), 3
            ),
            "types": dict(Counter(mention.get("type") for mention in labeled)),
            "document_formats": dict(formats),
            "quality_flags": dict(quality_flags),
        },
        "semantic_quality_assessed": False,
        "dataset_records_created": 0,
    }
    (pilot_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def export_training_drafts(
    pilot_dir: Path,
    out_dir: Path,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Xuất các draft cơ học hợp lệ thành dữ liệu thử nghiệm model v2, có manifest rõ provenance."""
    if not (0 <= validation_ratio < 1 and 0 <= test_ratio < 1 and validation_ratio + test_ratio < 1):
        raise ValueError("validation/test ratio không hợp lệ")
    audit_report = audit(pilot_dir)
    if not audit_report["valid"]:
        raise RuntimeError("Audit không đạt; từ chối export training drafts")
    manifest = json.loads((pilot_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    drafts = _read_jsonl(pilot_dir / "drafts.jsonl")
    if not drafts:
        raise RuntimeError("Chưa có draft hợp lệ để export")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} đã tồn tại; dùng --overwrite nếu muốn ghi lại")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    splits = {"train": [], "validation": [], "test": []}
    for draft in drafts:
        score = int(hashlib.sha256(draft["draft_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "test" if score < test_ratio else "validation" if score < test_ratio + validation_ratio else "train"
        splits[split].append({
            "id": draft["draft_id"],
            "text": draft["text"],
            "entities": [
                _app_label(mention) for mention in draft["mentions"]
                if mention.get("should_label")
            ],
            "track": draft["track"],
            "genre": draft["reference_case"].get("genre", "unknown"),
            "source": "synthetic_track_a",
            "record_kind": "segment",
            "source_case_id": draft["source_case_id"],
        })
    for split, rows in splits.items():
        _write_jsonl_atomic(out_dir / f"{split}.jsonl", rows)
    export_manifest = {
        "stage": "synthetic_training_trial_export",
        "source_pilot": str(pilot_dir),
        "source_manifest_sha256": sha256_file(pilot_dir / "pilot_manifest.json"),
        "track": manifest["track"],
        "track_a_part3_exact_evidence_used": False,
        "records": {name: len(rows) for name, rows in splits.items()},
        "mechanically_valid_source_drafts": len(drafts),
        "excluded_rejected_or_failed": len(_read_jsonl(pilot_dir / "generation_failures.jsonl")),
        "review_status": "synthetic_training_trial_user_authorized",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return export_manifest


def export_review(pilot_dir: Path) -> dict[str, Any]:
    audit_path = pilot_dir / "audit_report.json"
    if not audit_path.exists():
        raise RuntimeError("Phải chạy subcommand audit riêng trước")
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit_report.get("valid"):
        raise RuntimeError("Mechanical audit chưa PASS")
    drafts = _read_jsonl(pilot_dir / "drafts.jsonl")
    rows = []
    for draft in drafts:
        item = dict(draft)
        for mention in item["mentions"]:
            mention["review"] = {
                "label_decision_ok": None,
                "span_ok": None if mention["should_label"] else "not_applicable",
                "type_ok": None if mention["should_label"] else "not_applicable",
                "assertion_ok": None if mention["should_label"] else "not_applicable",
                "context_evidence_ok": None,
                "corrected_should_label": None,
                "corrected_span": None,
                "corrected_type": None,
                "corrected_assertions": None,
                "note": "",
            }
        item["review"] = {
            "context_natural": None,
            "context_complete_for_decisions": None,
            "all_medical_mentions_accounted_for": None,
            "missing_or_extra_mentions": [],
            "verdict": None,
            "note": "",
        }
        rows.append(item)
    review_path = pilot_dir / "manual_review.jsonl"
    if review_path.exists():
        raise FileExistsError(f"{review_path} đã tồn tại; không ghi đè review")
    _write_jsonl(review_path, rows)
    report = {
        "stage": "human_review_packet_created",
        "review_path": str(review_path),
        "draft_count": len(rows),
        "allowed_verdicts": ["accept", "needs_edit", "reject"],
        "next_stage_runs_automatically": False,
        "dataset_records_created": 0,
    }
    (pilot_dir / "review_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def check_review(pilot_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(pilot_dir / "manual_review.jsonl")
    incomplete, accepted, needs_edit, rejected = [], [], [], []
    field_errors = Counter()
    for row in rows:
        local = []
        semantic_false = []
        review = row.get("review") or {}
        verdict = review.get("verdict")
        if verdict not in {"accept", "needs_edit", "reject"}:
            local.append("review.verdict")
        for field in (
            "context_natural",
            "context_complete_for_decisions",
            "all_medical_mentions_accounted_for",
        ):
            if not isinstance(review.get(field), bool):
                local.append(f"review.{field}")
        for mention in row.get("mentions") or []:
            mention_review = mention.get("review") or {}
            required = ["label_decision_ok", "context_evidence_ok"]
            if mention.get("should_label"):
                required += ["span_ok", "type_ok", "assertion_ok"]
            for field in required:
                if not isinstance(mention_review.get(field), bool):
                    local.append(f"{mention.get('text')}.{field}")
                elif mention_review[field] is False:
                    field_errors[field] += 1
                    semantic_false.append(f"{mention.get('text')}.{field}")
        if local:
            incomplete.append({"draft_id": row.get("draft_id"), "missing": local})
        elif verdict == "accept":
            if any(
                row["review"].get(field) is not True
                for field in (
                    "context_natural",
                    "context_complete_for_decisions",
                    "all_medical_mentions_accounted_for",
                )
            ) or semantic_false:
                incomplete.append({
                    "draft_id": row.get("draft_id"),
                    "missing": [
                        "verdict accept nhưng còn field context/occurrence sai",
                        *semantic_false,
                    ],
                })
            else:
                accepted.append(row["draft_id"])
        elif verdict == "needs_edit":
            needs_edit.append(row["draft_id"])
        else:
            rejected.append(row["draft_id"])
    report = {
        "stage": "human_review_checked",
        "complete": not incomplete,
        "total": len(rows),
        "accepted": len(accepted),
        "needs_edit": len(needs_edit),
        "rejected": len(rejected),
        "accepted_ids": accepted,
        "field_error_counts": dict(field_errors),
        "incomplete": incomplete,
        "authorizes_next_batch": False,
        "dataset_records_created": 0,
    }
    (pilot_dir / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _app_label(mention: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": mention["text"],
        "position": list(mention["position"]),
        "type": mention["type"],
        "assertions": list(mention.get("assertions") or []),
        "candidates": [],
    }


def _validate_app_labels(raw: str, labels: list[dict[str, Any]]) -> list[str]:
    errors = []
    seen = set()
    spans = []
    for index, entity in enumerate(labels):
        position = entity.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(value, int) for value in position)
        ):
            errors.append(f"[{index}] position không hợp lệ")
            continue
        start, end = position
        if not (0 <= start < end <= len(raw)):
            errors.append(f"[{index}] position ngoài RAW")
        elif raw[start:end] != entity.get("text"):
            errors.append(f"[{index}] raw[position] != text")
        elif not is_word_bounded(raw, start, end):
            errors.append(f"[{index}] span cắt giữa token chữ/số")
        if entity.get("type") not in TYPES:
            errors.append(f"[{index}] type không hợp lệ")
        assertions = entity.get("assertions")
        if not isinstance(assertions, list) or set(assertions) - set(ASSERTIONS):
            errors.append(f"[{index}] assertion không hợp lệ")
        if assertions and entity.get("type") not in ASSERTION_TYPES:
            errors.append(f"[{index}] type xét nghiệm không được có assertion")
        if not isinstance(entity.get("candidates"), list):
            errors.append(f"[{index}] candidates không phải list")
        key = (tuple(position), entity.get("type"))
        if key in seen:
            errors.append(f"[{index}] trùng position/type")
        seen.add(key)
        if (
            isinstance(position, list)
            and len(position) == 2
            and all(isinstance(value, int) for value in position)
        ):
            spans.append((position[0], position[1], index))
    spans.sort()
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            errors.append(
                f"[{left[2]}]/[{right[2]}] span overlap/lồng; dataset BIO không chấp nhận"
            )
    return errors


def export_app(
    pilot_dir: Path,
    app_dir: Path,
    replace_unreviewed: bool = False,
) -> dict[str, Any]:
    """Xuất draft đúng contract annotation.app; metadata nghiệp vụ để sidecar."""
    audit_path = pilot_dir / "audit_report.json"
    if not audit_path.exists():
        raise RuntimeError("Phải chạy audit trước khi export sang app")
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit_report.get("valid"):
        raise RuntimeError("Mechanical audit chưa PASS")
    drafts = _read_jsonl(pilot_dir / "drafts.jsonl")
    if not drafts:
        raise RuntimeError("Không có draft hợp lệ để review")
    if app_dir.exists() and replace_unreviewed:
        # Resume batch: ghi lại export cho đủ draft mới, nhưng KHÔNG được xoá công review
        # của người dùng đã lưu trong thư mục app.
        reviewed_path = app_dir / "_reviewed.json"
        reviewed = (
            json.loads(reviewed_path.read_text(encoding="utf-8"))
            if reviewed_path.exists() else []
        )
        if reviewed:
            raise RuntimeError(
                f"{app_dir} đã có {len(reviewed)} file được review; đổi --app-out sang thư mục "
                "mới thay vì ghi đè."
            )
        shutil.rmtree(app_dir)
    app_dir.mkdir(parents=True, exist_ok=False)
    notes_dir = app_dir / "notes"
    labels_dir = app_dir / "labels"
    notes_dir.mkdir()
    labels_dir.mkdir()
    metadata = []
    for index, draft in enumerate(drafts, 1):
        file_id = f"{index:03d}"
        raw = draft["text"]
        labels = [
            _app_label(mention)
            for mention in draft["mentions"]
            if mention["should_label"]
        ]
        labels.sort(key=lambda row: (row["position"][0], row["position"][1], row["type"]))
        errors = _validate_app_labels(raw, labels)
        if errors:
            raise RuntimeError(f"{draft['draft_id']}: {errors}")
        (notes_dir / f"{file_id}.txt").write_text(raw, encoding="utf-8")
        (labels_dir / f"{file_id}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        metadata.append({
            "file_id": file_id,
            "draft_id": draft["draft_id"],
            "track": draft["track"],
            "source_case_id": draft["source_case_id"],
            "document_format": draft.get("document_format"),
            "section_path": draft.get("section_path") or [],
            "generation_brief": draft.get("generation_brief"),
            "entity_contract": draft.get("entity_contract"),
            "case_status": draft["reference_case"]["status"],
            "case_dimension": draft["reference_case"]["dimension"],
            "case_risk": draft["reference_case"]["risk"],
            "rules": draft["reference_case"]["rules"],
            "all_proposed_mentions": draft["mentions"],
            "initial_label_count": len(labels),
            "note_sha256": sha256_file(notes_dir / f"{file_id}.txt"),
            "initial_labels_sha256": sha256_file(labels_dir / f"{file_id}.json"),
        })
    _write_jsonl(app_dir / "metadata.jsonl", metadata)
    (app_dir / "_reviewed.json").write_text("[]", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "stage": "annotation_app_review",
        "source_pilot": str(pilot_dir),
        "source_drafts_sha256": sha256_file(pilot_dir / "drafts.jsonl"),
        "format": "train_v1_notes_labels",
        "files": len(metadata),
        "initial_entities": sum(row["initial_label_count"] for row in metadata),
        "candidates_policy": "empty_ner_pilot",
        "intentional_o_policy": (
            "not highlighted; reviewer sees RAW and may add missing spans; "
            "proposals retained in metadata.jsonl"
        ),
        "reviewed_files": 0,
        "is_training_dataset": False,
        "next_stage_runs_automatically": False,
    }
    (app_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (app_dir / "README.md").write_text(
        "# Review draft bằng annotation.app\n\n"
        "Chạy từ repository root:\n\n"
        f"```bash\npython -m annotation.app --data_dir {app_dir} --port 5000\n```\n\n"
        "Mở http://127.0.0.1:5000. Sửa/thêm/xóa span rồi bấm **Lưu (đã review)**.\n"
        "Các file ở đây vẫn là draft; `_reviewed.json` chỉ ghi file đã được con người xem.\n",
        encoding="utf-8",
    )
    return manifest


def collect_app_review(app_dir: Path) -> dict[str, Any]:
    metadata_rows = _read_jsonl(app_dir / "metadata.jsonl")
    metadata = {row["file_id"]: row for row in metadata_rows}
    reviewed_path = app_dir / "_reviewed.json"
    reviewed = set(json.loads(reviewed_path.read_text(encoding="utf-8")))
    unknown = sorted(reviewed - set(metadata))
    records, errors, changes = [], [], []
    aggregate_changes = Counter()
    for file_id in sorted(reviewed & set(metadata)):
        raw = (app_dir / "notes" / f"{file_id}.txt").read_text(encoding="utf-8")
        labels = json.loads(
            (app_dir / "labels" / f"{file_id}.json").read_text(encoding="utf-8")
        )
        local = _validate_app_labels(raw, labels)
        if local:
            errors.append({"file_id": file_id, "errors": local})
            continue
        proposed = metadata[file_id]["all_proposed_mentions"]
        initial = {
            tuple(mention["position"]): _app_label(mention)
            for mention in proposed if mention["should_label"]
        }
        intentional_o = {
            tuple(mention["position"]): mention
            for mention in proposed if not mention["should_label"]
        }
        final = {tuple(entity["position"]): entity for entity in labels}
        file_changes = []
        for position in sorted(initial.keys() - final.keys()):
            file_changes.append({
                "kind": "deleted",
                "position": list(position),
                "before": initial[position],
            })
        for position in sorted(final.keys() - initial.keys()):
            kind = "intentional_o_promoted" if position in intentional_o else "added"
            file_changes.append({
                "kind": kind,
                "position": list(position),
                "after": final[position],
                "proposal": intentional_o.get(position),
            })
        for position in sorted(initial.keys() & final.keys()):
            before, after = initial[position], final[position]
            if before["type"] != after["type"]:
                file_changes.append({
                    "kind": "retyped",
                    "position": list(position),
                    "before": before,
                    "after": after,
                })
            if sorted(before["assertions"]) != sorted(after["assertions"]):
                file_changes.append({
                    "kind": "assertion_changed",
                    "position": list(position),
                    "before": before["assertions"],
                    "after": after["assertions"],
                })
        aggregate_changes.update(change["kind"] for change in file_changes)
        if file_changes:
            changes.append({
                "file_id": file_id,
                "draft_id": metadata[file_id]["draft_id"],
                "changes": file_changes,
            })
        records.append({
            "id": metadata[file_id]["draft_id"],
            "text": raw,
            "entities": labels,
            "track": metadata[file_id]["track"],
            "source_case_id": metadata[file_id]["source_case_id"],
            "review": {
                "method": "annotation.app",
                "file_id": file_id,
                "human_reviewed": True,
            },
            "status": "human_reviewed_draft_not_training_data",
        })
    _write_jsonl(app_dir / "reviewed_drafts.jsonl", records)
    report = {
        "stage": "annotation_app_review_collected",
        "total_files": len(metadata),
        "reviewed_files": len(reviewed & set(metadata)),
        "valid_reviewed_files": len(records),
        "unreviewed_files": len(set(metadata) - reviewed),
        "unknown_reviewed_ids": unknown,
        "errors": errors,
        "changed_files": len(changes),
        "change_counts": dict(aggregate_changes),
        "changes": changes,
        "complete": len(reviewed) == len(metadata) and not unknown and not errors,
        "is_training_dataset": False,
        "authorizes_batch_or_training": False,
    }
    (app_dir / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Micro-pilot context-first có human gate")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--casebook", default=str(DEFAULT_CASEBOOK))
    prepare_cmd.add_argument("--casebook-approval", required=True)
    prepare_cmd.add_argument("--out", required=True)
    prepare_cmd.add_argument("--track", choices=["A", "B", "C"], default="A")
    prepare_cmd.add_argument("--variants-per-case", type=int, default=2)
    prepare_cmd.add_argument("--case-id", action="append", default=[])
    prepare_cmd.add_argument("--track-evidence")
    prepare_cmd.add_argument(
        "--design",
        choices=[
            "hard_stress_v3", "long_section_v4", "allowlist_section_v5",
            "full_document_v6", "occurrence_gate_v6",
        ],
        default="hard_stress_v3",
    )
    prepare_cmd.add_argument("--entity-seed", type=int, default=DEFAULT_ENTITY_SEED)
    prepare_cmd.add_argument(
        "--gate-focus", action="store_true",
        help="Mẻ bổ sung: nhiều occurrence gate mỗi bản ghi, để cân lại mẻ đã sinh bị lệch",
    )
    prepare_cmd.add_argument(
        "--omitted-first-rate", type=float, default=0.0,
        help="Ép tỷ lệ occurrence BỊ BỎ viết trước occurrence được gán (mặc định 0.55)",
    )
    prepare_cmd.add_argument(
        "--resume", action="store_true",
        help="Dùng lại thư mục prepare cũ nếu requests dựng ra trùng khít (để resume generate)",
    )
    prepare_cmd.add_argument(
        "--general-coverage-count", type=int, default=0,
        help="Số mẫu broad coverage không dùng casebook/Part 3; chỉ dùng entity contract",
    )

    generate_cmd = sub.add_parser("generate")
    generate_cmd.add_argument("--pilot", required=True)
    generate_cmd.add_argument("--request-timeout", type=float, default=90)
    generate_cmd.add_argument("--attempts", type=int, default=1)
    generate_cmd.add_argument(
        "--correction-rounds", type=int, default=2,
        help="Số vòng sửa hợp đồng có mục tiêu trước khi bỏ draft (0 = tắt)",
    )
    generate_cmd.add_argument(
        "--concurrency", type=int, default=1,
        help=f"Số request Azure song song (1..{MAX_CONCURRENCY}); checkpoint vẫn an toàn",
    )
    generate_cmd.add_argument(
        "--execute-llm",
        action="store_true",
        help="Xác nhận rõ đây là command duy nhất trong module có gọi Azure",
    )

    document_cmd = sub.add_parser(
        "full-document-review",
        help="Chuẩn bị, sinh, audit và export draft VĂN BẢN hoàn chỉnh sang annotation.app",
    )
    document_cmd.add_argument("--casebook", default=str(DEFAULT_CASEBOOK))
    document_cmd.add_argument("--casebook-approval", required=True)
    document_cmd.add_argument("--pilot-out", required=True)
    document_cmd.add_argument("--app-out", required=True)
    document_cmd.add_argument("--track", choices=["A", "B", "C"], default="A")
    document_cmd.add_argument("--variants-per-case", type=int, default=1)
    document_cmd.add_argument("--general-coverage-count", type=int, default=10)
    document_cmd.add_argument("--entity-seed", type=int, default=DEFAULT_ENTITY_SEED)
    document_cmd.add_argument("--request-timeout", type=float, default=90)
    document_cmd.add_argument("--attempts", type=int, default=1)
    document_cmd.add_argument("--concurrency", type=int, default=1)
    document_cmd.add_argument("--execute-llm", action="store_true")

    section_cmd = sub.add_parser(
        "long-section-review",
        help="Chuẩn bị, sinh, audit và export section dài sang annotation.app",
    )
    section_cmd.add_argument("--casebook", default=str(DEFAULT_CASEBOOK))
    section_cmd.add_argument("--casebook-approval", required=True)
    section_cmd.add_argument("--pilot-out", required=True)
    section_cmd.add_argument("--app-out", required=True)
    section_cmd.add_argument("--variants-per-case", type=int, default=1)
    section_cmd.add_argument("--general-coverage-count", type=int, default=10)
    section_cmd.add_argument("--entity-seed", type=int, default=DEFAULT_ENTITY_SEED)
    section_cmd.add_argument("--request-timeout", type=float, default=90)
    section_cmd.add_argument("--attempts", type=int, default=1)
    section_cmd.add_argument("--concurrency", type=int, default=1)
    section_cmd.add_argument("--execute-llm", action="store_true")

    for command in ("audit", "export-review", "check-review"):
        item = sub.add_parser(command)
        item.add_argument("--pilot", required=True)

    export_app_cmd = sub.add_parser("export-app")
    export_app_cmd.add_argument("--pilot", required=True)
    export_app_cmd.add_argument("--out", required=True)

    collect_app_cmd = sub.add_parser("collect-app-review")
    collect_app_cmd.add_argument("--data-dir", required=True)

    export_train_cmd = sub.add_parser(
        "export-training",
        help="Xuất draft cơ học hợp lệ thành train/validation/test JSONL cho training thử nghiệm",
    )
    export_train_cmd.add_argument("--pilot", required=True)
    export_train_cmd.add_argument("--out", required=True)
    export_train_cmd.add_argument("--validation-ratio", type=float, default=0.05)
    export_train_cmd.add_argument("--test-ratio", type=float, default=0.05)
    export_train_cmd.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if getattr(args, "gate_focus", False):
        globals()["GATE_FOCUS"] = True
    if getattr(args, "omitted_first_rate", 0.0):
        globals()["OMITTED_FIRST_RATE_OVERRIDE"] = args.omitted_first_rate
    if args.command == "prepare":
        result = prepare(
            Path(args.casebook).resolve(),
            Path(args.casebook_approval).resolve(),
            Path(args.out).resolve(),
            args.track,
            args.variants_per_case,
            set(args.case_id) or None,
            Path(args.track_evidence).resolve() if args.track_evidence else None,
            args.design,
            args.entity_seed,
            args.general_coverage_count,
            args.resume,
        )
    elif args.command == "generate":
        if not args.execute_llm:
            raise RuntimeError("Thiếu --execute-llm; không gọi Azure ngầm")
        result = generate(
            Path(args.pilot).resolve(), args.request_timeout, args.attempts,
            args.concurrency, args.correction_rounds,
        )
    elif args.command == "full-document-review":
        if not args.execute_llm:
            raise RuntimeError("Thiếu --execute-llm; không gọi Azure ngầm")
        if args.track != "A":
            raise RuntimeError("Full-document review hiện chỉ mở Track A để tránh few-shot Part 3")
        result = full_document_review(
            Path(args.casebook).resolve(),
            Path(args.casebook_approval).resolve(),
            Path(args.pilot_out).resolve(),
            Path(args.app_out).resolve(),
            args.track,
            args.variants_per_case,
            args.general_coverage_count,
            args.entity_seed,
            args.request_timeout,
            args.attempts,
            args.concurrency,
        )
    elif args.command == "long-section-review":
        if not args.execute_llm:
            raise RuntimeError("Thiếu --execute-llm; không gọi Azure ngầm")
        result = section_review(
            Path(args.casebook).resolve(),
            Path(args.casebook_approval).resolve(),
            Path(args.pilot_out).resolve(),
            Path(args.app_out).resolve(),
            args.variants_per_case,
            args.general_coverage_count,
            args.entity_seed,
            args.request_timeout,
            args.attempts,
            args.concurrency,
        )
    elif args.command == "audit":
        result = audit(Path(args.pilot).resolve())
    elif args.command == "export-review":
        result = export_review(Path(args.pilot).resolve())
    elif args.command == "check-review":
        result = check_review(Path(args.pilot).resolve())
    elif args.command == "export-app":
        result = export_app(Path(args.pilot).resolve(), Path(args.out).resolve())
    elif args.command == "export-training":
        result = export_training_drafts(
            Path(args.pilot).resolve(), Path(args.out).resolve(),
            args.validation_ratio, args.test_ratio, args.overwrite,
        )
    else:
        result = collect_app_review(Path(args.data_dir).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
