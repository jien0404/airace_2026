# Đối chiếu bộ nghiệp vụ với dữ liệu đo được — 31/07/2026

Đối chiếu `CANONICAL.md`, `casebook/core/core_v1.jsonl`, `DATASET_CONTRACT.md`, `OPEN_QUESTIONS.md`
với những gì đo được trên Part 3, gt2 và mẻ synthetic 5.000.

Kết luận: **không có mâu thuẫn logic nào giữa các luật**, nhưng có **một lỗ hổng gốc** giải thích
phần lớn điểm yếu của synthetic, cộng bốn chỗ thiếu và hai con số không tái lập được.

---

## G1 ⭐ Casebook chỉ dạy VĂN XUÔI — gốc rễ của cả A1 lẫn A2

**31/32 case có `text` chỉ một dòng**, case dài nhất hai dòng. Casebook là thứ LLM nhìn để biết
"tài liệu y tế trông thế nào", nên generator sao chép trung thành: mẻ 5.000 ra 6 dòng/bản ghi
trong khi Part 3 có 31 dòng, và 7% dòng ngắn so với 50%.

Hệ quả đo được, phân loại assertion của Part 3 theo DÒNG chứa entity:

| | mục gạch đầu dòng | văn xuôi | dòng `Tên trường: giá trị` |
|---|---:|---:|---:|
| `isHistorical` (279 ca) | **78%** | 13% | 9% |
| `isNegated` (162 ca) | **58%** | 32% | 10% |
| `isFamily` (9 ca) | 67% | 22% | 11% |

Hình thức chiếm đa số tuyệt đối ngoài thực tế là **mục gạch đầu dòng**, mà casebook không có lấy
một case nào. Đây là lý do model gán assertion cho 9,8% entity so với gold 16,6%.

Ví dụ thật từ Part 3: `- Tái phát buồn nôn và tiêu chảy cách đây vài năm` ·
`- Vợ có các triệu chứng tương tự 3 tuần trước` · `Bệnh lý mãn tính: hội chứng ruột kích thích`.
Trong khi case `assertion.historical_disease.001` của ta là
`Năm 2020 bệnh nhân từng mắc viêm phổi và đã điều trị khỏi.` — đúng 13% thiểu số.

**Đã xử lý một phần:** `ASSERTION_CUE_STYLES` trong `pilot.py` nay dùng đúng tỷ lệ đo được theo
từng loại assertion, và validator ép bố cục nhiều dòng. **Còn thiếu:** casebook phải có case mẫu,
vì đó mới là thứ LLM bắt chước.

---

## G2 §4.1 "Không suy từ heading" đang đẩy generator tránh đúng hình thức phổ biến nhất

Luật hiện tại: *"Không suy từ heading hoặc một cue thời gian duy nhất"* và *"Một section 'Tiền sử'
vẫn có thể chứa triệu chứng hiện tại — đọc quan hệ thời gian của từng câu, không gán isHistorical
theo heading."*

Luật này **đúng** và phải giữ. Nhưng viết như hiện tại nó dễ bị đọc thành "cấu trúc không phải
bằng chứng", trong khi 78% `isHistorical` thật nằm trong mục liệt kê dưới heading tiền sử.

Cần bổ sung một câu hoà giải, đại ý: heading **một mình** không đủ; nhưng **mục liệt kê dưới
heading tiền sử KÈM chú thời gian/trạng thái trong chính mục đó** (`- Tăng huyết áp - đã điều trị
ổn`, `- Tái phát ... cách đây vài năm`) là hình thức bằng chứng hợp lệ và phổ biến nhất.

---

## G3 §6 là một phân phối GỘP, che mất hai lớp tài liệu

Bảng §6 ghi mật độ 13,30/1k, TRIỆU_CHỨNG 39,21%, assertion 16,56% cho cả 100 file. Nhưng Part 3
gồm hai lớp rất khác nhau:

| | 27 file không trùng Part 1 | 73 file còn lại |
|---|---:|---:|
| Mật độ | **8,7/1k** | 15,1/1k |
| CHẨN_ĐOÁN | **37%** | 23% |
| TRIỆU_CHỨNG | **23%** | 43% |
| Assertion | **7,7%** | 18,6% |

Dùng con số gộp làm "cổng phân phối" khiến dataset dồn về một điểm trung bình không tồn tại trên
bất kỳ lớp nào. Đúng trên lớp thưa, precision của cả ba model sụp còn 43–47%.

**Đã xử lý:** `DENSITY_BUCKETS` nay mang profile type + tỷ lệ assertion riêng cho từng bucket.
**Còn thiếu:** §6 nên ghi bảng theo lớp tài liệu, không chỉ số gộp.

---

## G4 Hai con số trong §1.2 không tái lập được

| Con số trong CANONICAL | Đo lại được | Ghi chú |
|---|---|---|
| Nhóm mixed gold Part 2 = **20,0%** | **33,1%** | khác nhau ở cách đếm (ranh giới từ, xử lý occurrence bị phủ) |
| Occurrence lặp cùng câu ≤100 ký tự chỉ **16,7%** được lấy | **70,3%** | chênh quá lớn để là sai số |
| Nhóm surface lặp = 2.566 | 1.861 | |
| Occurrence bị phủ = 2.939 | 1.716 | |

Không kết luận bên nào sai — vấn đề là **CANONICAL không ghi cách đếm**, nên con số không kiểm
lại được. Cần một trong hai: ghi rõ script/định nghĩa đã dùng, hoặc thay bằng số của
`occurrence_report` (có mã nguồn kiểm được). Trong lúc chờ, **không mã hoá 16,7% vào generator**.

---

## G5 Không case nào dạy "cùng surface, khác type theo ngữ cảnh"

§2.1 nói type phụ thuộc vai trò của cụm trong câu, và §2.3 nói `bình thường` là
`KẾT_QUẢ_XÉT_NGHIỆM` **khi** nó là kết quả rõ của một phép đo. Nhưng **0/32 case minh hoạ cùng
một surface mang hai type khác nhau**.

Hệ quả: synthetic có 2.103 surface và **0 surface đa type**, trong khi Part 2 có 165 surface đổi
type theo ngữ cảnh. `bình thường` được gán `KẾT_QUẢ_XÉT_NGHIỆM` 133/133 lần. Dataset đang dạy
`surface → type`, đúng thứ §2.1 cấm.

---

## G6 Không thể loại nào ứng với lớp tài liệu THƯA, nặng chẩn đoán

`genre` trong casebook chỉ có: `free_text_clinical_note`, `structured_clinical_note`,
`lab_or_imaging_report`, `medical_education`, `patient_narrative_qa`,
`medication_reconciliation`. Thiếu hẳn dạng **tóm tắt ra viện / danh sách chẩn đoán** — tài liệu
dài nhưng ít nhãn, gần như toàn `CHẨN_ĐOÁN`, rất ít assertion. Đó chính là lớp 27 file ở G3.

---

## G7 `DATASET_CONTRACT.md` không có yêu cầu nào về bố cục

Hợp đồng dataset quy định occurrence gate, tỷ lệ mixed, vị trí occurrence, hard-negative — nhưng
không có một dòng nào về **hình thức văn bản**. Cần thêm mục: số dòng tối thiểu theo độ dài, tỷ lệ
dòng ngắn, và bắt buộc có các hình thức mục liệt kê / dòng trường / bảng xét nghiệm.

---

## Đề xuất mở rộng casebook — 6 case mới, chờ duyệt

| case_id | genre | dạy điều gì | vá lỗ nào |
|---|---|---|---|
| `layout.history_bullet_list.001` | structured_clinical_note | mục Tiền sử dạng gạch đầu dòng, `isHistorical` suy từ chú trong chính mục | G1, G2 |
| `layout.lab_table.001` | lab_or_imaging_report | bảng xét nghiệm mỗi chỉ số một dòng, cặp TÊN/KQ | G1 |
| `layout.field_colon_line.001` | structured_clinical_note | dòng `Tên trường: giá trị` | G1 |
| `layout.sparse_discharge_summary.001` | discharge_summary *(genre mới)* | tài liệu dài, ít nhãn, nặng CHẨN_ĐOÁN, gần như không assertion | G3, G6 |
| `type.same_surface_context_type.001` | lab_or_imaging_report | `bình thường` là KQ_XN ở chỗ này, `TRIỆU_CHỨNG` ở chỗ khác | G5 |
| `assertion.negation_bullet_list.001` | structured_clinical_note | phủ định trong mục điểm lại cơ quan dạng danh sách | G1 |

Casebook là cổng cần người duyệt, nên tôi **chưa tự thêm**. Nói một tiếng là tôi soạn đầy đủ
`text` + `entities` + `rationale` cho sáu case này rồi chạy `casebook validate` để bạn duyệt.

---

## Không phải vấn đề — đã kiểm

- Không có mâu thuẫn logic nào **giữa các luật** trong CANONICAL; §2.3 đã hoà giải đúng hai mệnh
  đề cũ, §3.4 đã hoà giải split/collapse, §1.1 đã hoà giải thể thức phát ngôn vs xuất xứ.
- `OPEN_QUESTIONS.md` O1–O7 vẫn đúng trạng thái, không mục nào bị generator âm thầm biến thành
  default. O7 (13 span vắt dòng) đang được `build_ner_corpus` xử lý đúng hướng: loại khỏi train
  chứ không sửa artifact.
- §5 (candidate ICD/RxNorm) ngoài phạm vi đội, không ảnh hưởng.
