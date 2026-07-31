# Sổ theo dõi — mẻ sinh synthetic tiếp theo

Cập nhật lần cuối: 31/07/2026 (sau smoke Track B `smoke_v4`).
Nền hiện tại: `datasets/pilots/train_v3` (3.220 draft). Mốc điểm đang giữ: **33,8337**
(`ner_v3_fixed_windows` + XLM-R).

⚠ **Backbone**: XLM-R hơn PhoBERT **+0,66** khi khoá dữ liệu. Dùng XLM-R từ giờ.

## Bằng chứng đang có

| Nhánh train | Nộp bài | WER | J_assertion | J_cand | F1 cục bộ | P / R | số nhãn | assertion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nhãn đã rà + XLM-R | **33,8337** | 58,9783 | 45,6849 | 19,5543 | 79,37 | 76,6 / 82,3 | 2.911 | 13,8% |
| nhãn đã rà + PhoBERT | 33,1754 | 60,0582 | 44,9916 | 19,2384 | 79,10 | 74,0 / 85,0 | 3.112 | 14,1% |
| synthetic v3 + XLM-R | 32,5916 | 60,8521 | 43,5648 | 19,4446 | 78,47 | 75,1 / 82,2 | 2.965 | 13,6% |
| synthetic + gt2 + part1 | 31,9128 | 61,9995 | **43,109** | **18,9498** | **76,86** | 69,8 / 85,6 | 3.326 | 13,7% |
| gt2 + part1 (thật) | 28,4558 | 65,4598 | 37,4717 | 17,1306 | 73,18 | 64,3 / 85,0 | 3.583 | 20,0% |
| synthetic một mình | 26,2326 | 70,4132 | 34,5815 | 17,4552 | 69,48 | 64,3 / 75,6 | 3.190 | 9,8% |
| *gold part3* | — | — | — | — | — | — | *2.711* | *16,6%* |

**Trộn thắng cả hai nguồn đứng riêng** (+3,46 so với dữ liệu thật). Synthetic không thừa — nó bù
được thứ dữ liệu thật không có.

### ⚠ Đo lại trên nhóm Part 3 KHÔNG trùng Part 1 (27/100 tài liệu, 496 entity)

Part 1 là bộ test vòng trước, nhãn tay của ta, và **40,5% shingle 12 từ của Part 3 nằm nguyên văn
trong Part 1** (73/100 tài liệu trùng >10%). Mọi nhánh có Part 1 đều được đọc lại đề bài.
`python -m training.clean_test_split`:

| Nhánh | F1 toàn bộ | F1 nhóm sạch | chênh | P sạch | R sạch |
|---|---:|---:|---:|---:|---:|
| trộn cả ba | 76,86 | **55,83** | −21,03 | 47,1 | 68,5 |
| gt2 + part1 | 73,18 | **53,50** | −19,68 | 43,2 | 70,2 |
| synthetic một mình | 69,48 | **52,99** | −16,49 | 44,7 | 65,1 |

**Khoảng cách giữa dữ liệu thật và synthetic sụp từ 3,70 xuống 0,51** — tức nằm trong sai số của
496 entity. Ưu thế của dữ liệu thật gần như hoàn toàn là **nhớ lại văn bản**, không phải dạy tốt
hơn. Trộn vẫn dẫn nhưng chỉ còn +2,84.

Kèm cảnh báo: 27 tài liệu sạch có phân bố khác hẳn phần còn lại (xem A0), nên mức tụt tuyệt đối
là hỗn hợp của "mất phần thuộc lòng" và "tài liệu khác loại". Chỉ so sánh GIỮA các nhánh trên cùng
27 tài liệu này mới hợp lệ.

### A0. Không có tài liệu THƯA — `chưa vá` ⭐ mới, phát hiện từ nhóm sạch

| | Part 3 nhóm sạch (27) | Part 3 nhóm trùng (73) | synthetic |
|---|---:|---:|---:|
| Mật độ entity | **8,7/1k** | 15,1/1k | 11,8/1k |
| CHẨN_ĐOÁN | **37%** | 23% | 25% |
| TRIỆU_CHỨNG | **23%** | 43% | 36% |
| assertion | **7,7%** | 18,6% | 19,5% |

Tồn tại cả một lớp tài liệu **thưa, nặng CHẨN_ĐOÁN, ít assertion** mà synthetic hầu như không sinh.
Trên đúng lớp này precision của cả ba model sụp còn 43–47% — model quen mật độ ~13/1k nên gán tràn
lan trên văn bản thưa. Đây là nguồn gán thừa lớn nhất đo được.

- Việc cần làm: `DENSITY_BUCKETS` phải gắn **profile type theo từng bucket**, không dùng chung một
  `SAMPLING_TYPE_WEIGHTS`. Bucket thưa: mật độ 7–10/1k, CHẨN_ĐOÁN ~37%, TRIỆU_CHỨNG ~23%,
  assertion ~8%. Nâng tỷ trọng bucket thưa lên ít nhất 30%.

### Synthetic đang LÀM TỐT ba việc — mẻ sau không được phá

1. **Phân bố type đúng nhất.** `TRIỆU_CHỨNG` dự đoán: synthonly 41% (gold 39%), realonly **49%**,
   trộn 47%. Dữ liệu thật kéo lệch vì gt2 gán 67% TRIỆU_CHỨNG; synthetic là nguồn duy nhất giữ
   được tỷ lệ đúng.
2. **Giảm gán thừa.** Số nhãn so với gold 2.711: synthonly **+18%**, trộn +23%, realonly **+32%**.
   Mẫu âm cố ý (bản ghi rỗng, đoạn giáo dục) đang có tác dụng thật.
3. **J_candidates cao hơn dữ liệu thật** (17,4552 vs 17,1306) dù F1 thấp hơn.

### Và đang HỎNG ở hai việc

1. **Assertion**: gán cho 9,8% entity, gold 16,6% — thiếu gần một nửa. Xem A2.
2. **Phủ từ vựng**: recall 75,6% so với 85,0% của dữ liệu thật. Xem A3.

Ngoài ra **cả ba model đều gán thừa 18–32%**, precision 64–70% mà recall 76–86%, trong khi đã đo
được rằng xoá nhãn sai lãi gấp ~13 lần thêm nhãn đúng. Xem mục F.

Trạng thái: `đã vá` (chờ sinh lại) · `chưa vá` · `chờ quyết` · `đã đóng` (đo xong, không phải vấn đề).

---

## A. Chặn — phải xong trước khi sinh mẻ mới

### A1. Văn bản synthetic không giống bệnh án thật về BỐ CỤC — `đã vá`

| | synthetic | part3 (đích) |
|---|---:|---:|
| Số dòng / bản ghi | **6** | **31** |
| Dòng ngắn ≤40 ký tự | **7%** | **50%** |
| Xuống dòng / 1k ký tự | 4,0 | 13,1 |
| Độ dài trung vị | 1.062 | 1.838 |

Bệnh án thật là **danh sách và bảng**; synthetic đang viết **văn xuôi**. Đây là chênh lệch lớn nhất
đo được giữa hai bên, và nó kéo theo A2 bên dưới. Nghi là nguyên nhân chính của cả việc gán thừa
lẫn `KẾT_QUẢ_XÉT_NGHIỆM` yếu nhất (F1 58,4) — kết quả xét nghiệm thật nằm trong bảng, không nằm
trong câu.

- **Đã vá bằng few-shot dòng thật** (`dataset_factory/style_skeleton.py`): 3.366 dòng trích từ
  296 hồ sơ part1/gt2/part3, đã tước hết nội dung y khoa. Mô tả bằng lời không đủ — phải cho LLM
  *nhìn* một dòng thật.

  Đo lại, chuẩn hoá theo độ dài (số dòng tuyệt đối KHÔNG so sánh được giữa hai độ dài khác nhau):

  | | ký tự/dòng | % dòng ngắn |
  |---|---:|---:|
  | Part 3 thật | 79 | 60% |
  | **smoke_v4** | **60** | **50%** |
  | mẻ v3 | 43 | 58% |

  smoke_v4 gần hồ sơ thật hơn v3.

### A2. Cue assertion sai HÌNH THỨC (không phải sai số lượng) — `đã vá`, chờ đo trên mẻ đầy đủ

Số lượng assertion **không có vấn đề** — synthetic `isHistorical` 10,02% vs gold 10,29%,
`isNegated` 8,12% vs 5,98%. Vấn đề là cue tập trung vào quá ít khuôn:

| | 10 bigram cue phổ biến nhất phủ bao nhiêu ca |
|---|---:|
| synthetic `isNegated` | **76%** |
| part3 `isNegated` | 50% |
| synthetic `isHistorical` | **66%** |
| part3 `isHistorical` | **33%** |

Và quan trọng hơn con số: **cue phổ biến nhất của part3 không phải từ ngữ**. Top bigram
`isHistorical` của part3 là `tính -`, `áp -`, `phì -`, `hiệu -` — tức tiền sử được suy từ **mục
gạch đầu dòng** (`Tăng huyết áp - đã điều trị`), trong khi synthetic chỉ dạy cue văn xuôi
`từng có`, `đã từng`, `tiền sử`. Model chưa bao giờ thấy tiền sử đánh dấu bằng cấu trúc danh sách
→ gặp văn bản thật là im lặng, gán assertion cho vỏn vẹn **9,8%** entity so với gold 16,6%.

- **Đã vá cùng A1**: dòng mẫu mang sẵn hình thức cue. Quan sát trực tiếp trên smoke_v4:

  ```
  - Nói ngọng - không ghi nhận lúc khám này.     -> Nói ngọng  [isNegated]
  Tiền sử trước đây: Sưng khớp từng xuất hiện…   -> Sưng khớp  [isHistorical]
  ```

  Cue mọc từ **cấu trúc** mục liệt kê và dòng trường-giá trị, đúng thứ cần. Nhưng tỷ lệ assertion
  smoke_v4 là 12,2% vs đích 16,6% — 56 bản ghi thì sai số ±2,6% nên **chưa phân biệt được với
  v3 (14,5%)**. Phải đo lại trên mẻ đầy đủ.
- Ghi chú: `assertion_f1` validation 0,988 vs test 0,804 chính là dấu hiệu của việc này — model
  học thuộc bộ cue ta mã hoá chứ không học được thực tế.

### A3. Lặp surface quá nhiều — `đã vá`, kèm đánh đổi đã biết

| | synthetic | part3 |
|---|---:|---:|
| Số ca / số surface | 36.293 / **2.103** | 2.711 / 1.179 |
| Lặp trung bình mỗi surface | **17,3×** | 2,3× |
| Surface chỉ xuất hiện **một lần** | **0%** | **56%** |

Thực tế là phân bố đuôi dài; synthetic không có đuôi nào cả. Hệ quả đo được: **33,5% entity gold
part3 (907/2.711) có surface chưa từng xuất hiện trong synthetic** — trần recall cứng.
Nặng nhất: `CHẨN_ĐOÁN` 338, `TRIỆU_CHỨNG` 232, `KẾT_QUẢ_XÉT_NGHIỆM` 205.

**Đã hạ mức độ:** phủ từ vựng của synthetic thực ra ngang bộ nhãn tay tốt nhất —
part1 phủ 69,0% surface gold Part 3, **synthetic 66,5%**, part2 44,5%; chỉ 6,9% gold là do riêng
part1 phủ. Khoảng cách recall 75,6% vs 85,0% KHÔNG phải do từ vựng mà do part1 trùng 40,5% văn bản
Part 3. Vẫn nên sửa vì 0% surface singleton là bất thường, nhưng đây không còn là mục chặn.

**Phân rã lại trần recall** (2.711 entity gold) — quan trọng hơn mọi con số ở trên:

| | ca | % |
|---|---:|---:|
| đã sinh được | 1.790 | 66,0 |
| ⛔ provenance **CHỈ part3** → firewall chặn | 494 | **18,2** |
| có trong kho, chưa từng bốc → **reuse cap** | 323 | **11,9** |
| không có ở đâu | 85 | 3,1 |
| cùng surface khác type (§2.1) | 19 | 0,7 |

**18,2% là firewall đang làm đúng việc, không phải lỗi.** `vàng da`, `Bệnh Kawasaki`, `Nitralmyl`
đều có `sources: ["nhãn tay:part3"]`. Lấy chúng là Track C, `contaminated=true`. Không làm.

Nới `max_tier` A→B: **0 surface gold**, chỉ thêm 78k `TRIỆU_CHỨNG` chất lượng thấp. Đã bỏ.

- **Đã vá**: `--surface-reuse-cap 6` trong `EntityCatalog`. smoke_v4: singleton 0,1% → **75,1%**,
  lặp 18,8× → **1,3×**.
- **Đánh đổi đã biết, đã chấp nhận**: cap kéo theo đuôi nhiễu của từ điển. ~3% entity là
  `TRIỆU_CHỨNG` một từ trần (`rát`, `cứng`, `hôi`, `đầy`) — có thật trong tier A nhưng là nhiễu
  gán nhãn của gt2 (gold đứng riêng 0-2 lần, so với `sốt` 204). Chọn giữ vì 34% trần recall lớn
  hơn nhiều so với 3% mảnh vụn, và mảnh vụn thì `label_audit` xoá được sau còn từ vựng thiếu thì
  không cứu được.

### A4. Vị trí occurrence được gán mang thông tin — `đã vá`, chờ sinh lại

Nhóm mixed: **99,3%** gán occurrence ĐẦU TIÊN (gold gt2 27,6%; với nhóm 2 occurrence gold còn
nghiêng về lần SAU 58%). 86,2% cặp lệch nằm trọn trong một cửa sổ 180 từ nên model nhìn thấy được.
Đây đúng luật cũ đã bị probe bác bỏ (−0,0762).

- Vá: `order_hint` + `REPEAT_OMITTED_FIRST_RATE = 0.55` + luật thứ tự trong `SYSTEM_PROMPT_GATE`.
- Đo: `occurrence_report` → `mixed_first_only_pct`, mục tiêu **45–55%**.
- Mẻ bù dùng `--gate-focus --omitted-first-rate 0.9`.

### A5. Mật độ nhóm gate tụt sau cách ly — `chưa vá`

`train_v2_5000_kept` chỉ còn `mixed_pct = 8,5%` (mục tiêu train 35–40%). Cần ~900–1.000 bản ghi
chế độ `--gate-focus` để bù.

---

### A6. Dấu hiệu sinh tồn tự chốt một convention còn MỞ — `đã vá` ⭐ phát hiện ở smoke_v4

12 entity trên 56 bản ghi (21%), **lần nào cũng** gán `TÊN_XÉT_NGHIỆM`. `CANONICAL` §2.4 và
`OPEN_QUESTIONS` O1 ghi rõ đây là câu hỏi còn MỞ; `DATASET_CONTRACT` §6 cấm *"sinh vital theo một
convention duy nhất khi câu hỏi còn MỞ"*; `GENERATOR_AUDIT` đã hạ `--vital-rate` về 0 vì đúng lý
do này. Dataset đang âm thầm dạy một luật chưa ai đo.

Vào qua **hai** đường, và đường thứ hai là chỗ dễ sót:

1. từ điển sạch — chặn được bằng tag `lab_panel_VITALS`;
2. **47/497 cặp `xn_pairs` quan hệ `ket_qua`** (`huyết áp` → trị số) — cặp KHÔNG mang tag đó nên
   phải chặn theo tên (`VITAL_NAMES` trong `sources.py`).

- Đã vá: `EntityCatalog(allow_vitals=False)` mặc định, chặn cả hai đường. Còn 0/793 mention.
  Bật `allow_vitals=True` khi O1 đóng.

### A7. Từ vựng hướng dẫn rò vào nội dung — `đã vá` ⭐ phát hiện ở smoke_v4

14/56 bản ghi (25%) viết `"vào mục"`, `"nhập mục"`, `"lý do vào mục"`. LLM nhặt chữ `mục` từ chính
`generation_brief` (`"scope": "một mục/section hoàn chỉnh"`) rồi dùng như tên nơi chốn. Văn bản
thành ngọng, phá đúng mục tiêu "giống hồ sơ thật" của cả mẻ.

- Đã vá: đổi cách diễn đạt brief, và thêm luật cấm mượn từ vựng hướng dẫn vào
  `SYSTEM_PROMPT_GATE`.

---

## B. Chất lượng nhãn

### B1. Phủ định kép — `đã vá ở generator`, `đã vá ở corpus`

42 entity mang `isNegated` mà surface đã tự phủ định: `không không nhấc chân phải khỏi mặt giường`,
`phủ nhận không có năng lượng`, `không có Không dung nạp thức ăn`.

- `SELF_NEGATED_SURFACE` trong `pilot.py` chặn ở cả plan lẫn validator. Mẻ v3 còn **1 ca** trên
  39.777 entity.

### B1b. Span nuốt từ phủ định mà assertion để trống — `đã vá ở corpus`, `chưa vá ở nguồn`

Khác B1: đây không phải phủ định kép mà là **thiếu một nửa**. Span là `không sốt`, `Không ho`,
`không đau đầu` — assertion rỗng. Đúng quy ước phải là span `sốt` + `isNegated`.

Quy ước đo trên cả bốn nguồn: **2.093 ca để từ phủ định NGOÀI span, 2 ca ngược lại**. Không có
chỗ cho tranh cãi.

Nhưng KHÔNG cắt bừa được: `không thể tự đứng dậy`, `không nhấc chân phải khỏi mặt giường`,
`chưa phát hiện bất thường` là gold part3 nguyên văn — ở đó phủ định CHÍNH LÀ phát hiện. Tiêu chí
tách hai lớp: chỉ cắt khi **phần lõi tự nó đã là entity đứng riêng trong gold thật ≥5 lần**.

- Đã vá: `strip_leading_negator()` trong `training/build_ner_corpus.py`, tập
  `NEGATABLE_CORES` = {chướng, ho, ngủ, nôn, sốt, đau, đau đầu}. Sửa 112 span khi build
  `ner_v4` (105 synthetic + 7 gt2). Đo lại bằng
  `python -m training.negation_span_audit --data datasets/ner_v4/track_a`.
- **Còn ở nguồn:** `dictionary/clean/TRIỆU_CHỨNG.json` có 646/80.091 surface mở đầu bằng từ phủ
  định, `entity_sources` có 519/105.257. Đa số hợp lệ (thể loại bất-lực-làm-gì), nhưng lớp
  `Không buồn nôn`, `Không ban đỏ…` thì nên tách ở chính từ điển để mẻ sau không đẻ lại. Chưa
  làm — build đang gánh hộ.
- Loại trừ có chủ ý: `tự chủ đại tiện` đủ 5 lần đỡ nhưng gold part3 gán
  `tiểu tiện không tự chủ` NGUYÊN CỤM; cắt nó là lật ngược nghĩa.

### B2. Mỗi surface chỉ có đúng một type — `chưa vá`

Synthetic: 2.103 surface, **0 surface đa type**. Part 2: 7.022 surface, **165 surface đổi type theo
ngữ cảnh**. Ví dụ `bình thường` luôn là `KẾT_QUẢ_XÉT_NGHIỆM` (133 lần). Đang dạy `surface → type`.

- Vá: thêm lớp case "cùng surface, khác type theo ngữ cảnh", lấy chính 165 surface đa type của
  part2 làm nguồn.

---

## C. Phân bố — chỉnh tham số

| Chỉ số | Mẻ 5.000 | Part 3 (đích) | Việc |
|---|---:|---:|---|
| Mật độ entity | 11,8/1k | 13,2/1k | nâng nhẹ |
| TÊN_XÉT_NGHIỆM | 19% | 15% | hạ |
| TRIỆU_CHỨNG | 36% | 39% | nâng nhẹ |
| CHẨN_ĐOÁN / KẾT_QUẢ / THUỐC | 25 / 9 / 10 | 26 / 10 / 10 | ổn |
| `near_same_sentence_taken_pct` | 88,3% | 70,3% (gt2) | hạ |

---

## D. Đã đóng — đo xong, KHÔNG phải vấn đề

- **Rò rỉ Part 3 không ảnh hưởng điểm.** Bỏ 35 surface nhiễm khỏi phép chấm: F1 69,48 → 69,44.
  Vẫn nên dọn `dictionary/sources.json` cho sạch (part3 đang ở `priority: 30`, kéo theo 1.135 mục
  có provenance part3, 318 mục chỉ nhờ part3), nhưng đây là việc vệ sinh, không phải việc điểm số.
- **Số lượng assertion đúng rồi** — xem A2, vấn đề nằm ở hình thức cue.
- **Hình thái `KẾT_QUẢ_XÉT_NGHIỆM` đúng rồi**: 62% surface có chữ số (part3 61%), dài trung vị 11
  (part3 10). F1 thấp là do bố cục (A1), không do cách viết giá trị.
- **Độ dài entity đúng rồi**: trung vị 3 từ cả hai bên; ≥5 từ 24,5% vs 23,2%; ≥8 từ 4,4% vs 5,9%.
- Tỷ lệ draft đạt 96%; bản ghi rỗng 4,8% là mẫu âm cố ý; marker `<E id>` cho 0 lỗi offset trên
  36.293 entity.

---

## E. Chưa xác minh — đừng mã hoá vào generator

- Spec nói chỉ **16,7%** occurrence lặp cùng câu trong 100 ký tự được gán; đo trên gt2 ra **70,3%**.
  Chưa hoà giải được, **không** đưa vào generator.
- Artifact Part 3 đang dùng là `v66ab` (50,4346). Bản "best 50.6598" mà audit đối chiếu không có
  trong `business_rules/artifacts/current/`.

---

## F. Không thuộc sinh data, nhưng nên thử trước

Cả hai model **gán thừa 18–23%**, precision 64–70% trong khi recall 76–86%. Nâng
`--type-threshold` và **hạ** `--assertion-threshold` lúc suy luận là đòn bẩy **không cần train lại**.
Nên quét ngưỡng trước khi tốn một mẻ sinh mới.
