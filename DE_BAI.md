# Đề bài: Clinical Information Extraction & Concept Normalization (Tiếng Việt)

> Bài toán trích xuất khái niệm y tế + chuẩn hóa mã từ note lâm sàng tiếng Việt.
> Thuộc "họ" bài toán kinh điển: i2b2 2010 / n2c2 (concepts, assertions, relations).

---

## 1. Thể thức & Quy định

- **Vòng 1**: nộp `output.zip`, giải nén ra thư mục `output/` chứa `1.json … 100.json` (nhãn tương ứng `1.txt … 100.txt`).
- **Top ~15 đội** phải nộp source code để BTC dựng lại và chấm trên **private test** (chống hard-code output).
  - Source gồm: toàn bộ code (data processing, training, inference), data nhóm sử dụng, model weights, 1 file README cài đặt.
  - Nếu BTC không cài được → liên lạc hỗ trợ trong thời gian nhất định; không kịp → **bị loại**.
- **Ràng buộc tài nguyên**: giải pháp LLM/agent chỉ được **self-host model ≤ 9B params**, **không dùng API ngoài**.
  - Lưu ý: ràng buộc này áp cho pipeline lúc **re-run inference**. Việc dùng model lớn offline để *sinh/annotate data huấn luyện* **cần xác nhận với BTC** — nếu họ coi đó là một phần "lời giải" thì có thể bị loại.

---

## 2. Input / Output

### Input
Mỗi bản ghi là một đoạn text y khoa tự do (không cấu trúc), như bác sĩ gõ tay: lặp lại, sai chính tả, trộn Anh/Việt, nhiều section.

**Ví dụ input (VD vòng 1):**
```
'Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po
daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily 4.
guaifenesin ml po q6h:prn điều trị ho 5. nystatin oral suspension 5 ml po qid:prn
điều trị đau nhức 6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7.
pravastatin 40 mg po daily 8. docusate sodium 100 mg po bid điều trị táo bón 9.
senna 8.6 mg po bid:prn điều trị táo bón 10. clonazepam 0.5 mg po qam:prn điều
trị lo âu 11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ'
```

### Output — mỗi khái niệm là 1 object với các trường:

| Trường | Ý nghĩa |
|---|---|
| `text` | Cụm chữ bóc ra, **y nguyên** như input |
| `position` | `[start, end]` vị trí ký tự. **KHÔNG chấm điểm** — chỉ cần đúng format |
| `type` | 1 trong 5 loại (xem dưới) |
| `assertions` | List ngữ cảnh (rỗng nếu bình thường). Chỉ cho CHẨN_ĐOÁN / THUỐC / TRIỆU_CHỨNG |
| `candidates` | List mã chuẩn. Chỉ cho CHẨN_ĐOÁN (ICD-10) và THUỐC (RxNorm) |

### 5 loại `type`
- **TRIỆU_CHỨNG** — dấu hiệu bệnh nhân than phiền ("tức ngực", "ho").
- **CHẨN_ĐOÁN** — kết luận bệnh của bác sĩ ("trào ngược dạ dày – thực quản").
- **TÊN_XÉT_NGHIỆM** — tên phép xét nghiệm ("WBC", "NEUT%").
- **KẾT_QUẢ_XÉT_NGHIỆM** — con số + đơn vị ("14,43").
- **THUỐC** — tên thuốc bệnh nhân dùng ("Chlorpheniramine 0.4 MG/ML").

> ⚠️ Tên xét nghiệm và kết quả là **HAI khái niệm riêng**. "WBC" = TÊN_XÉT_NGHIỆM; "14,43" = KẾT_QUẢ_XÉT_NGHIỆM. Phải tách.

### 3 giá trị `assertions`
- **isNegated** — bị phủ định ("không ho").
- **isFamily** — của người nhà ("bố bệnh nhân cũng đau bụng").
- **isHistorical** — thuộc tiền sử ("có tiền sử hen suyễn", "tiền sử sử dụng...").
- Bình thường (đang có, hiện tại) → `[]`.

### `candidates`
- Bệnh → **ICD-10** (VN dùng chính thức, có bảng tiếng Việt).
- Thuốc → **RxNorm** (chuẩn Mỹ — khó map trực tiếp thị trường VN).
- Là **list** vì one-to-many: "trào ngược dạ dày – thực quản" → `K21.0` và `K21.9`.
- 3 loại còn lại (triệu chứng, tên XN, kết quả) → luôn `[]`.

---

## 3. Metric đánh giá (QUAN TRỌNG NHẤT)

```
final_score = 0.3 · text_score + 0.3 · assertions_score + 0.4 · candidates_score
```

Với mỗi sample `i`, mỗi candidate `k`; `X` là trường tương ứng:

```
text_score       = Σ_i (1 − WER(i)) / len(test)

assertions_score = Σ_i J_assertions(i) / len(test)

candidates_score = Σ_i [ J_candidates(i) · Σ_k (len(GT(k))+1) ]
                   ─────────────────────────────────────────────
                        Σ_i Σ_k (len(GT(k))+1)
```

**Jaccard J_X(i):**
- `= 1` nếu GT rỗng **và** prediction rỗng.
- `= 0` nếu GT rỗng **và** prediction ≠ rỗng.
- `= |GT ∩ pred| / |GT ∪ pred|` các trường hợp còn lại.

### Hệ quả then chốt (phản trực giác)

1. **Sai type = phạt kép.** Không có type_score riêng. Đoán đúng `text` nhưng sai `type` → concept bị tính **2 lần** (1 thừa + 1 thiếu), **cả 2 đều 0 điểm trên cả 3 metric**. → Sai type **tệ hơn bỏ sót**. **Type accuracy là ưu tiên số 1.**
2. **position không chấm điểm** — đừng tối ưu offset, chỉ cần format hợp lệ.
3. **Rỗng là baseline mạnh.** GT rỗng + pred rỗng = 1 điểm; over-predict = 0. → **Default rỗng** cho assertions & candidates trừ khi có bằng chứng rõ.
4. **candidates (0.4) có trọng số `len(GT)+1`** → chẩn đoán/thuốc nhiều mã chi phối điểm. Phải hiệu chỉnh số mã trả về (thiếu cap Jaccard, thừa phình union).
5. **TÊN_XÉT_NGHIỆM & KẾT_QUẢ_XÉT_NGHIỆM** không có candidates/assertions → chỉ ăn điểm `text` (phần "dễ" của 0.3).

---

## 4. Giải pháp tổng thể

**Tư tưởng cốt lõi:** đây là *bài toán dữ liệu*, không phải bài toán model. Không có training data, runtime ≤9B, metric thưởng *grounding* (mã từ DB thật) chứ không thưởng *generation*.

### Kiến trúc: pipeline có grounding, KHÔNG phải 1 LLM sinh JSON
Lý do:
- `text_score` (WER) cần span khớp chữ gốc → encoder token-classification cho span+offset miễn phí; LLM sinh tự do trôi chữ.
- `candidates` cần mã có thật → model 9B sẽ bịa RXCUI/ICD → phải retrieve từ DB.
- `type` sai phạt kép → cần khối gán type có kiểm soát.

### Engine sinh dữ liệu (offline) — quyết định ~80% thắng thua
- **Giả thuyết nguồn**: test giống **MIMIC** (medications-on-admission EN + RxNorm) + **n2c2** (assertion, ICD), narrative được Việt hóa.
- Lấy corpora có gold code: MIMIC-III/IV, n2c2/i2b2, + ViMedNER, danh mục Cục QLD, ICD-10 tiếng Việt (phần thuần Việt).
- Dùng LLM lớn offline: Việt hóa narrative (giữ tên thuốc EN + mã gốc), bơm nhiễu đúng style (typo, header, lặp mention, trộn Anh/Việt), auto-annotate, **validate mọi mã bằng chính DB được cấp**.
- **Bắt buộc**: chốt guideline gán nhãn nhất quán trước rồi nướng vào prompt sinh data. Model chỉ tốt bằng độ nhất quán của nhãn.

### 4 khối runtime (≤9B, self-host)
| Khối | Nội dung | Metric gate |
|---|---|---|
| **A. NER encoder** | PhoBERT/ViHealthBERT + token-classification → text, type, position. Cân nhắc span-based cho nested | **type accuracy** |
| **B. Assertion** | ConText tiếng Việt (rule, cue hữu hạn) + classifier nhỏ cho cue xa. **Default rỗng** | assertions 0.3 |
| **C. Normalization** | Thuốc: parse ingredient+strength+form → RxNorm SCD (tất định). Chẩn đoán: embed (BGE-m3/SapBERT) → retrieve top-k ICD-10 VN. LLM 9B chỉ **rerank trong ứng viên có thật** | candidates 0.4 |
| **D. Lắp ráp** | dedup theo (text chuẩn hóa + type), ép candidates rỗng cho non-drug/dx, validate mã tồn tại, xuất JSON | — |

### Thứ tự dồn công (bám trọng số)
1. **Type accuracy** — gate mọi thứ.
2. **candidates 0.4** — nhánh thuốc trước (tất định, điểm to), rồi retrieval chẩn đoán.
3. **text 0.3** — tự có nếu NER tốt.
4. **assertions 0.3** — rule precision cao, mặc định rỗng.

### 3 cược sống còn
- **Generalize, không memorize** — BTC re-run private test → normalization phải bằng retrieval, không tra-cứng-theo-text.
- **Kỷ luật "rỗng mặc định"** cho cả assertion và candidates.
- **Số lượng mã trả về là hyperparameter** — tune trên dev, không trả hết top-k.

---

## 5. Các điểm mờ cần chốt guideline (từ mẫu note thực tế)

1. **Dedup mention lặp** — 1 triệu chứng xuất hiện nhiều lần ở nhiều section → extract 1 lần hay nhiều lần? (giả định: dedup theo text chuẩn hóa + type). **Cần xác nhận bằng file gold.**
2. **Xung đột assertion khi dedup** — thuốc vừa ở "tiền sử" (isHistorical) vừa ở "bệnh sử hiện tại" → chọn assertion nào?
3. **Xét nghiệm gồm gì** — imaging (x-quang, ecg, holter) vs lab (phân tích nước tiểu). Đề tách riêng "kết quả XN" và "kết quả chẩn đoán hình ảnh" → có thể chỉ lab mới là TÊN_XÉT_NGHIỆM.
4. **Kết quả định tính** — "bình thường", "không ghi nhận bất thường" có phải KẾT_QUẢ_XÉT_NGHIỆM không? (đề định nghĩa là giá trị + đơn vị → có thể không).
5. **Vital signs** (`VS98.3 12987 56 18 99RA`) — có phải xét nghiệm không? (nghiêng: không extract).
6. **Yếu tố tâm lý xã hội** (căng thẳng, mất việc) — TRIỆU_CHỨNG hay bỏ?
7. **Chẩn đoán ngầm** ("doxycycline cho viêm tuyến mồ hôi") — có code ICD không? (nghiêng: có, isHistorical).
8. **Kết quả bình thường** (nhịp xoang) không code; bất thường (ngoại tâm thu nhĩ/thất) → code ICD I49.x.

---

## 6. Việc cần làm tiếp theo (thứ tự)

**Bước 0 — gỡ bất định (làm trước, ít công):**
- Hỏi BTC: dùng LLM lớn offline sinh data huấn luyện có hợp lệ không?
- Xin file example đã gán nhãn (dù 1–2 file) → trả lời trực tiếp các điểm mờ §5.

**Bước 1 — chốt guideline gán nhãn** (1 trang, dứt khoát cho 8 điểm mờ). Input bắt buộc cho engine sinh data.

**Bước 2 — nhánh thuốc → RxNorm SCD** (điểm chắc nhất, gần tất định, không cần training data):
- Tải RxNorm RRF (RXNCONSO), build index local.
- Parser `ingredient + strength + form` → match SCD/SBD.

**Bước 3 — engine sinh data + NER encoder** (sau khi có guideline). Đo **type accuracy** số 1.

**Bước 4 — retrieval chẩn đoán + assertion ConText tiếng Việt.**

---

## 7. Ví dụ output tham chiếu (từ input VD vòng 1)

Toàn bộ thuốc mang `isHistorical` (danh sách "thuốc trước nhập viện"); các "điều trị X" tách thành TRIỆU_CHỨNG riêng.

```json
[
  {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "candidates": ["308135"], "assertions": ["isHistorical"], "position": [58, 83]},
  {"text": "aspirin 81 mg po daily", "type": "THUỐC", "candidates": ["243670"], "assertions": ["isHistorical"], "position": [89, 111]},
  {"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC", "candidates": ["866436"], "assertions": ["isHistorical"], "position": [117, 155]},
  {"text": "guaifenesin ml po q6h:prn", "type": "THUỐC", "candidates": ["392085"], "assertions": ["isHistorical"], "position": [161, 186]},
  {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [196, 198]},
  {"text": "nystatin oral suspension 5 ml po qid:prn", "type": "THUỐC", "candidates": ["7597"], "assertions": ["isHistorical"], "position": [204, 244]},
  {"text": "đau nhức", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [254, 262]},
  {"text": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC", "candidates": ["313782"], "assertions": ["isHistorical"], "position": [268, 303]},
  {"text": "sốt đau", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [313, 320]},
  {"text": "pravastatin 40 mg po daily", "type": "THUỐC", "candidates": ["904475"], "assertions": ["isHistorical"], "position": [326, 352]},
  {"text": "docusate sodium 100 mg po bid", "type": "THUỐC", "candidates": ["1099279"], "assertions": ["isHistorical"], "position": [358, 387]},
  {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [397, 404]},
  {"text": "senna 8.6 mg po bid:prn", "type": "THUỐC", "candidates": ["312935"], "assertions": ["isHistorical"], "position": [410, 433]},
  {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [443, 450]},
  {"text": "clonazepam 0.5 mg po qam:prn", "type": "THUỐC", "candidates": ["197527"], "assertions": ["isHistorical"], "position": [457, 485]},
  {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [495, 500]},
  {"text": "clonazepam 1.5 mg po qhs", "type": "THUỐC", "candidates": ["197528"], "assertions": ["isHistorical"], "position": [507, 531]},
  {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [541, 546]},
  {"text": "mất ngủ", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [547, 554]}
]
```

> ⚠️ Các mã RxNorm/ICD ở đây là phán đoán, **phải verify bằng DB thật** BTC cấp. RXCUI là loại dễ bịa.
