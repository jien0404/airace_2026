"""Prompt variants cho benchmark nghiệp vụ Part 3.

Không đưa V66ab, RULES_S hay note lịch sử vào prompt. V66ab chỉ được dùng để
tạo ZIP reference và phân tích offline sau khi LLM đã gán độc lập.
"""

from __future__ import annotations

import json
from typing import Any


TYPES = (
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "THUỐC",
)
ASSERTIONS = ("isNegated", "isHistorical", "isFamily")


OUTPUT_CONTRACT = """\
OUTPUT CONTRACT

Chỉ trả về đúng một JSON object, không markdown và không giải thích bên ngoài JSON:

{
  "entities": [
    {
      "line_id": 1,
      "text": "chuỗi nguyên văn trong dòng",
      "occurrence": 1,
      "type": "TRIỆU_CHỨNG",
      "assertions": [],
      "candidates": []
    }
  ],
  "self_check": {
    "scanned_all_lines": true,
    "reviewed_missing_mentions": true,
    "reviewed_overlap": true
  }
}

Quy ước output:
- `line_id` là số trong tiền tố <L0001>, <L0002>... của văn bản được cung cấp.
- `text` phải là substring nguyên văn của đúng dòng, không sửa Unicode/chính tả/ASR.
- Nếu cùng `text` lặp nhiều lần trên một dòng, `occurrence` là thứ tự 1-based của lần đó.
- Liệt kê từng occurrence hợp lệ; không gộp các lần nhắc.
- `type` chỉ được thuộc năm type chuẩn.
- `assertions` chỉ dùng ba giá trị chuẩn. Với TÊN_XÉT_NGHIỆM và
  KẾT_QUẢ_XÉT_NGHIỆM luôn là [].
- `candidates` chỉ chứa chuỗi mã ICD-10 cho CHẨN_ĐOÁN hoặc RXCUI cho THUỐC.
  Các type khác luôn là []. Không chắc thì để [].
- Sắp entity theo line_id rồi theo thứ tự xuất hiện.
- Không tạo hai entity chồng lấn.
"""


OFFICIAL_POLICY = """\
TASK DEFINITION — BASELINE

Gán nhãn các khái niệm y khoa trong note tiếng Việt theo năm type:

- TRIỆU_CHỨNG: dấu hiệu hoặc triệu chứng được nhắc đến.
- CHẨN_ĐOÁN: bệnh, tình trạng hoặc kết luận bệnh.
- TÊN_XÉT_NGHIỆM: tên phép đo, xét nghiệm hoặc thăm dò.
- KẾT_QUẢ_XÉT_NGHIỆM: kết quả của phép đo/xét nghiệm.
- THUỐC: thuốc, hoạt chất, biệt dược hoặc chế phẩm.

Assertions:
- isNegated: occurrence bị phủ định.
- isHistorical: occurrence thuộc tiền sử/quá khứ.
- isFamily: occurrence thuộc người thân.
- Mặc định assertions là [].

Candidates:
- CHẨN_ĐOÁN có thể có mã ICD-10.
- THUỐC có thể có RXCUI.
- Không chắc chắn cao thì để [].

Mỗi occurrence được gán riêng. Trích nguyên văn và chọn boundary/type phù hợp nhất
theo ngữ cảnh. Đọc toàn bộ note trước khi trả lời.
"""


COMPILED_POLICY = """\
EXECUTABLE PART 3 POLICY

Thực hiện sáu lượt nội bộ theo đúng thứ tự dưới đây. Không xuất suy luận trung gian;
chỉ xuất JSON cuối cùng theo OUTPUT CONTRACT.

PASS 1 — LẬP BẢN ĐỒ DISCOURSE

- Nhận diện bệnh án, lời kể, khám, xét nghiệm, tư vấn/giáo dục, danh sách và khối
  thuộc ca khác.
- Nguồn gốc đoạn không tự quyết định việc gán. Không xóa chỉ vì đoạn là giáo dục,
  tư vấn hoặc ca khác. [C1.1]
- Nhưng không gán mọi thuật ngữ y khoa. Thường bỏ cơ chế bệnh sinh, giải thích sinh
  học, checklist biểu hiện điển hình, tác dụng phụ/chống chỉ định/nguy cơ chỉ nêu
  như kiến thức chung, câu giả định "có thể gây/nếu...thì", thuốc chỉ làm ví dụ,
  heading/ô thuộc tính và danh từ meta. [C1.1]
- Trong giáo dục vẫn giữ có chọn lọc tên bệnh chủ đề/subtype, chẩn đoán hoặc finding
  là nội dung chuyên môn chính, tên xét nghiệm, phương pháp điều trị, tên nhóm thuốc
  và kết quả cụ thể. [C1.1]

PASS 2 — QUÉT CANDIDATE MENTION

- Quét từng dòng, riêng cho cả năm type; chưa quyết định chỉ bằng từ điển.
- Gán từng occurrence hợp lệ, kể cả surface lặp. [C1.2]
- Tên nhóm thuốc như corticoid, NSAID, kháng sinh, thuốc chống đông có thể là THUỐC.
  [C1.3]

PASS 3 — TYPE THEO OCCURRENCE

- Type phụ thuộc vai trò trong câu và convention, không chỉ bản chất của surface.
  ICD/RxNorm chỉ chứng minh mã tồn tại, không quyết định type. [C2.1]
- CHẨN_ĐOÁN: bệnh/tình trạng được đặt tên hoặc kết luận patient-specific.
- TRIỆU_CHỨNG: cảm giác, than phiền, dấu hiệu hoặc finding được convention xem là
  symptom.
- TÊN_XÉT_NGHIỆM: phép đo, xét nghiệm, thăm dò hoặc kỹ thuật chẩn đoán.
- KẾT_QUẢ_XÉT_NGHIỆM: trị số, kết luận định tính hoặc narrative kết quả rõ ràng.
- THUỐC: hoạt chất, biệt dược, chế phẩm hoặc nhóm thuốc trong context hợp lệ.
- Sau TÊN_XÉT_NGHIỆM, vế phải có thể là KẾT_QUẢ, CHẨN_ĐOÁN, TRIỆU_CHỨNG hoặc
  một TÊN_XÉT_NGHIỆM khác. Không type theo vị trí mặc định. [C2.2]
- "bình thường", "âm tính", "dương tính" là KẾT_QUẢ khi vai trò kết quả của một
  phép đo/thăm dò rõ. Đứng rời không đủ để gán. [C2.3]
- Vital còn là vùng chưa chốt. Không áp một convention hàng loạt; quyết định thận
  trọng theo occurrence và hạ confidence nếu mơ hồ. [C2.4]

PASS 4 — BOUNDARY

- Trích nguyên văn RAW, không sửa typo, Unicode hoặc ASR. [C3.1]
- TRIỆU_CHỨNG thường lấy concept lõi; không dùng regex cắt modifier toàn cục.
  Modifier có thể là phần định danh trong context khác. [C3.2]
- Giữ ngoặc khi là alias, viết tắt, hoạt chất/formulation hoặc phần định danh.
  Cắt ngoặc khi chỉ lặp concept hay prose dư. [C3.3]
- Split khi có nhiều concept lâm sàng độc lập. Giữ một tên lexicalized hoặc
  coordination nội bộ. Collapse khi toàn vế là một narrative kết quả hình ảnh.
  [C3.4]
- Qualifier định danh bệnh như mạn tính, giai đoạn, không đặc hiệu thường phải giữ.
  [C3.5]

PASS 5 — ASSERTION VÀ CANDIDATE

- Assertion xét riêng từng occurrence; mặc định []. [C4.1]
- isNegated chỉ cho phủ định trực tiếp trong cùng mệnh đề.
- isHistorical chỉ cho bệnh/đợt thật sự thuộc quá khứ; không suy từ heading hoặc
  cue thời gian đơn lẻ.
- isFamily chỉ khi occurrence thuộc người thân.
- TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn assertions [].
- Đặc biệt thận trọng với historical cho thuốc trước nhập viện/đã ngừng/hết thuốc;
  không gán hàng loạt. [C4.2]
- Candidate rỗng là nhãn hợp lệ. Chỉ CHẨN_ĐOÁN và THUỐC có candidate. [C5.1]
- Không gán mã chỉ vì nhớ/tra thấy một mã. Ưu tiên exact context và formulation;
  nếu không chắc cao thì []. [C5.2]
- Thuốc phải xét ingredient, strength, route và dosage form. Ingredient đúng có
  thể tốt hơn một clinical drug sai formulation. [C5.3]
- Các convention xác nhận: tăng bilirubin máu -> []; rụng tóc toàn thể -> L63.1;
  Bệnh amyloidosis tự miễn dịch -> E85.9; bệnh thận mạn, không đặc hiệu Giai đoạn 4
  -> N18.9. [C5.4]

PASS 6 — ADVERSARIAL SELF-CHECK

- Quét lại từ dòng cuối lên đầu để tìm đoạn bị bỏ trắng hoặc occurrence bị thiếu.
- Tìm false positive do cơ chế/checklist/giả định/meta.
- Kiểm tra lại mọi cặp CHẨN_ĐOÁN vs TRIỆU_CHỨNG và TÊN_XN vs KẾT_QUẢ.
- Kiểm tra boundary, ngoặc, split/collapse, assertion theo occurrence.
- Bỏ candidate không chắc; không ép mật độ hoặc quota type.
- Bảo đảm không duplicate/overlap và output đúng schema.

Ví dụ tổng hợp không thuộc tập benchmark:
- "Không ho, tiền sử hen phế quản": ho = TRIỆU_CHỨNG + isNegated; hen phế quản
  = CHẨN_ĐOÁN + isHistorical.
- "Siêu âm ổ bụng: bình thường": siêu âm ổ bụng = TÊN_XÉT_NGHIỆM; bình thường
  = KẾT_QUẢ_XÉT_NGHIỆM.
- "Vastarel có thể gây run tay": trong câu kiến thức chung về tác dụng phụ, không
  tự động gán run tay như tình trạng bệnh nhân.
"""


CRITIC_POLICY = """\
INDEPENDENT VERIFIER

Bạn là lượt kiểm định thứ hai. Bản draft được tạo từ EXECUTABLE PART 3 POLICY nhưng
có thể thiếu mention, thừa mention, sai type/boundary/assertion/candidate.

Hãy đọc lại toàn bộ RAW độc lập rồi đối chiếu draft theo thứ tự:
1. Dòng/đoạn nào bị bỏ sót hoàn toàn?
2. Mention nào chỉ là cơ chế, checklist, giả định, meta hoặc thuốc làm ví dụ?
3. Type có đúng vai trò occurrence không?
4. Boundary, ngoặc và split/collapse có đúng không?
5. Assertion có bị lan theo surface hoặc suy từ heading không?
6. Candidate có bị ép khi chưa chắc hoặc sai formulation không?
7. Có duplicate/overlap hoặc occurrence lặp bị thiếu không?

Không được mặc định tin draft. Trả về DANH SÁCH ĐẦY ĐỦ đã sửa, không trả diff.
Giữ candidate rỗng nếu không chắc chắn cao. Không được tham chiếu hidden labels hay V66ab.
"""


def numbered_note(raw: str) -> str:
    """Thêm line id mà không thay đổi nội dung dùng để align."""
    lines = raw.splitlines()
    if raw.endswith(("\n", "\r")):
        # splitlines bỏ dòng rỗng cuối; line id đó không cần cho entity.
        pass
    return "\n".join(f"<L{i:04d}> {line}" for i, line in enumerate(lines, 1))


def _draft_json(draft: list[dict[str, Any]]) -> str:
    fields = (
        "line_id",
        "text",
        "occurrence",
        "type",
        "assertions",
        "candidates",
    )
    clean = [{key: entity.get(key) for key in fields} for entity in draft]
    return json.dumps({"entities": clean}, ensure_ascii=False, indent=2)


def build_messages(
    variant: str,
    raw: str,
    canonical_text: str,
    draft: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    system = (
        "Bạn là hệ thống gán nhãn độc lập cho benchmark Clinical Information "
        "Extraction tiếng Việt. Không được giả định có hidden labels. "
        "Thực hiện cẩn thận và chỉ trả JSON hợp lệ.\n\n" + OUTPUT_CONTRACT
    )
    note = numbered_note(raw)

    if variant == "official_only":
        policy = OFFICIAL_POLICY
        task = "Gán nhãn note sau chỉ bằng TASK DEFINITION."
    elif variant == "canonical_dump":
        policy = (
            OFFICIAL_POLICY
            + "\n\nCANONICAL RULEBOOK — dùng nguyên văn; XÁC NHẬN là default, "
            "SUY RA cần thận trọng, MỞ không được biến thành luật hàng loạt:\n\n"
            + canonical_text
        )
        task = "Gán nhãn note sau bằng TASK DEFINITION và CANONICAL RULEBOOK."
    elif variant == "compiled":
        # COMPILED_POLICY đã chứa đầy đủ type/assertion/candidate; không lặp lại
        # OFFICIAL_POLICY để prompt gọn và mỗi instruction chỉ xuất hiện một lần.
        policy = COMPILED_POLICY
        task = "Thực hiện đủ sáu pass nội bộ rồi gán nhãn note sau."
    elif variant == "compiled_verified":
        if draft is None:
            raise ValueError("compiled_verified cần draft từ variant compiled")
        policy = COMPILED_POLICY + "\n\n" + CRITIC_POLICY
        task = (
            "Kiểm định độc lập note và draft dưới đây, sau đó trả toàn bộ danh sách "
            "entity đã sửa.\n\nDRAFT:\n" + _draft_json(draft)
        )
    else:
        raise ValueError(f"Variant không hỗ trợ: {variant}")

    user = f"{policy}\n\n{task}\n\nRAW NOTE CÓ ĐÁNH SỐ DÒNG:\n{note}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def templates(canonical_text: str) -> dict[str, str]:
    """Nội dung policy không chứa RAW, dùng lưu provenance."""
    return {
        "00_output_contract.txt": OUTPUT_CONTRACT,
        "01_official_only.txt": OFFICIAL_POLICY,
        "02_canonical_dump.txt": OFFICIAL_POLICY + "\n\n" + canonical_text,
        "03_compiled.txt": COMPILED_POLICY,
        "04_compiled_verified.txt": (
            COMPILED_POLICY + "\n\n" + CRITIC_POLICY
        ),
    }
