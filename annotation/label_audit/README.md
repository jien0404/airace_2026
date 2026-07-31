# label_audit — rà soát nhãn part1/gt2 theo nghiệp vụ

Mục tiêu: tìm những chỗ nhãn tay của `best_54.92` (part1) và `groundtruth_part2` (gt2) **mâu
thuẫn trực tiếp** với quy ước trong [`business_rules/CANONICAL.md`](../../business_rules/CANONICAL.md).
Chỉ bắt lỗi rõ ràng; nghi ngờ mơ hồ được bỏ qua có chủ ý.

Hai luồng độc lập, chạy song song được (journal và file kết quả riêng):

| Luồng | Sửa gì | Ứng viên |
|---|---|---|
| `--mode fix` (mặc định) | `assertions`, `type`, ranh giới span | chỉ entity bị luật sàng nghi ngờ |
| `--mode delete` | XOÁ entity không đáng gán | **mọi** entity |

Luồng `delete` chỉ đề xuất xoá, **không bao giờ đề xuất thêm**: xoá nhãn sai lãi gấp ~13 lần thêm
nhãn đúng, và nguồn "thiếu nhãn" lớn nhất là occurrence lặp — thêm chúng đúng là luật đã bị probe
bảng xếp hạng bác bỏ (−0,0762).

Nhãn gốc **không bao giờ bị sửa**. Bản đã sửa nằm ở thư mục riêng, tạo bởi lệnh `apply` sau khi
người dùng duyệt từng đề xuất.

## Ba tầng, mỗi tầng chặt hơn tầng trước

1. **Sàng bằng luật** (`screen.py`) — chỉ giữ entity mà một luật cơ học đã đủ để nghi ngờ:

   | Nhóm | Ý nghĩa | Cần LLM? |
   |---|---|---|
   | `assertion_on_forbidden_type` | TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM mang assertion | không — luật tuyệt đối |
   | `unknown_assertion` | assertion ngoài ba loại hợp lệ | không |
   | `offset_mismatch` | `raw[start:end] != text` | không |
   | `assertion_without_cue` | có assertion nhưng câu không có cue nào đỡ | có |
   | `cue_without_assertion` | cue phủ định/tiền sử ngay trước entity mà nhãn để trống | có |
   | `historical_drug` | THUỐC mang isHistorical (nghiệp vụ còn OPEN) | có |
   | `span_swallows_negation` | span mang isNegated mà nuốt luôn từ phủ định (`"Không đau đầu"`) | có |
   | `negation_inside_span` | span bắt đầu bằng từ phủ định nhưng assertions rỗng | có |
   | `type_minority_vs_corpus` | type thiểu số so với đa số ≥85% trên ≥4 occurrence của chính part1+gt2 | có |

   Luồng `delete` KHÔNG sàng bằng luật: đã thử sàng theo từ khoá diễn ngôn (`nếu`, `nguy cơ`,
   `điển hình`) và gần như toàn báo động giả — `gợi ý`, `điển hình` xuất hiện đầy trong câu mô tả
   bệnh nhân thật. Thể thức phát ngôn phải do LLM đọc mà phân biệt.

2. **LLM phân xử** (`run.py adjudicate`) — mỗi nghi ngờ kèm context ±400 ký tự **và heading của
   mục**. Prompt yêu cầu
   trả `keep` khi còn lăn tăn. Được sửa **`assertions`, `type` và ranh giới span**; KHÔNG được
   thêm/bớt entity, và span mới bắt buộc là khúc con liền mạch của span cũ (chặn ở
   `_resolve_change`, không tin LLM tự giới hạn).

   **Vì sao ±400 mà không phải to hơn:** đo trên part1+gt2, p95 khoảng cách tới đầu câu là 170 ký
   tự, tới cuối câu 156. ±400 phủ trọn câu chứa entity 99,5%; nới lên 700 chỉ thêm 0,5 điểm.
   Nhưng heading thì **không** với tới được bằng cửa sổ — chỉ lọt 29,5% ở ±320 và 62% ở ±1000 —
   nên nó được lấy riêng và gửi kèm trường `muc_dang_o`, phủ 76,8% số entity.

   **Ba rào chắn của luồng delete**, rút ra từ hai vòng hiệu chuẩn 300 ca: không xoá vì entity bị
   phủ định (vẫn là entity, chỉ mang `isNegated`), không xoá vì span cắt lệch, không xoá dấu hiệu
   sinh tồn (CANONICAL §2.4 còn OPEN, gold được giữ nguyên).

   **Chữ ký đầu vào** `_input_signature()` băm prompt + `CONTEXT_CHARS` + cờ heading, đóng dấu vào
   từng phán xử; journal chỉ tái dùng verdict có chữ ký khớp. Trước đây đây là một con số đếm tay
   và nó đã hỏng đúng một lần: nới cửa sổ 320→400 nhưng quên tăng số, phán xử sinh với context cũ
   vẫn được dùng lại im lặng.

3. **Người duyệt** (`app.py`) — hai nút: đồng ý sửa / giữ nguyên.

## Chạy

```bash
# 1. xem trước bằng luật, không tốn API
python -m annotation.label_audit.run screen

# 2a. sửa assertion / type / span (resume được: chạy lại là bỏ qua phần đã xong)
python -m annotation.label_audit.run adjudicate \
  --batch-size 8 --concurrency 6 --request-timeout 120 --attempts 3 --execute-llm

# 2b. soát lại quyết định gán trên MỌI entity — chạy song song 2a được
python -m annotation.label_audit.run adjudicate --mode delete \
  --batch-size 10 --concurrency 6 --request-timeout 120 --attempts 3 --execute-llm

# hiệu chuẩn nhanh trước khi chạy full: lấy mẫu ngẫu nhiên
python -m annotation.label_audit.run adjudicate --mode delete --sample 300 \
  --batch-size 10 --concurrency 5 --request-timeout 120 --attempts 2 --execute-llm

# 3. duyệt trên giao diện — A = đồng ý sửa, G = giữ nguyên
python -m annotation.label_audit.app --audit-dir annotation/data/label_audit --port 5001

# 4. xuất bản nhãn đã sửa sang thư mục riêng
python -m annotation.label_audit.run apply
```

Kết quả:

```text
annotation/data/label_audit/
  screened.jsonl      # toàn bộ nghi ngờ từ tầng luật
  verdicts.jsonl        # journal luồng fix (append-only, fsync mỗi batch, dùng để resume)
  verdicts_delete.jsonl # journal luồng delete
  findings.jsonl        # đề xuất sửa đã xác nhận, đầu vào của giao diện
  findings_delete.jsonl # đề xuất xoá đã xác nhận
  audit_report.json
  decisions.json      # quyết định của người dùng
annotation/data/label_fixed/
  part1/{notes,labels}/   # bản sao đã áp dụng quyết định
  gt2/{notes,labels}/
  manifest.json
```

`apply` chỉ ghi những nguồn thực sự có thay đổi. Span mới bắt buộc là **khúc con liền mạch** của
span cũ (chặn ở `_resolve_change`, không tin LLM tự giới hạn), và trước khi ghi còn kiểm lại
`raw[start:end] == text`; lệch thì bỏ phần span và ghi vào manifest thay vì ghi bừa. Đổi type sang
TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM thì assertions tự xoá theo luật tuyệt đối.

## Quy mô đo được (31/07/2026)

18.170 entity trên part1 (2.726) + gt2 (15.444).

| Luồng | Ứng viên | Lượt gọi | Thời gian ở concurrency 6 |
|---|---:|---:|---:|
| fix | 3.716 nghi ngờ, 333 cơ học | ~423 | ~35 phút |
| delete | 18.167 | ~1.817 | ~1,5 giờ |

Sàng luồng fix, theo nhóm:

| Nhóm | Số ca |
|---|---:|
| `cue_without_assertion` | 1.970 |
| `assertion_without_cue` | 1.053 |
| `assertion_on_forbidden_type` | 330 — vi phạm luật tuyệt đối, không cần LLM |
| `negation_inside_span` | 297 |
| `type_minority_vs_corpus` | 30 |
| `historical_drug` | 29 |
| `span_swallows_negation` | 4 |
| `offset_mismatch` | 3 |

Tỷ lệ đề xuất xoá: **13,7% trên mẫu ngẫu nhiên 300**, nhưng **4,5% trên 1.380 ca đầu của lượt
chạy đầy đủ** — chênh lệch chưa lý giải được, chờ chạy hết mới có số thật. Ngoại suy nằm đâu đó
giữa 820 và 2.480.

⚠ Nhóm ngoài `TRIỆU_CHỨNG` là nhóm rủi ro khi duyệt: đã đo được rằng xoá TÊN_XÉT_NGHIỆM lỗ
−0,0096 và xoá CHẨN_ĐOÁN/THUỐC lỗ −0,0071. Bằng chứng đó đo trên bài nộp chứ không phải trên dữ
liệu train nên không áp thẳng được, nhưng đủ để soi kỹ nhóm này.

⚠ Nhãn part3 gán tay ở local chỉ đạt WER ~50, và part1 trùng 40,5% văn bản với part3. Mọi kết
luận về "chất lượng sau khi sửa" phải chờ điểm nộp bài, không đọc từ F1 cục bộ.
