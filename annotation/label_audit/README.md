# label_audit — rà soát nhãn part1/gt2 theo nghiệp vụ

Mục tiêu: tìm những chỗ nhãn tay của `best_54.92` (part1) và `groundtruth_part2` (gt2) **mâu
thuẫn trực tiếp** với quy ước trong [`business_rules/CANONICAL.md`](../../business_rules/CANONICAL.md),
đặc biệt là assertion. Chỉ bắt lỗi rõ ràng; nghi ngờ mơ hồ được bỏ qua có chủ ý.

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

2. **LLM phân xử** (`run.py adjudicate`) — mỗi nghi ngờ kèm context ±320 ký tự. Prompt yêu cầu
   trả `keep` khi còn lăn tăn. Được sửa **`assertions`, `type` và ranh giới span**; KHÔNG được
   thêm/bớt entity, và span mới bắt buộc là khúc con liền mạch của span cũ (chặn ở
   `_resolve_change`, không tin LLM tự giới hạn).

   `PROMPT_VERSION` đóng dấu vào từng phán xử. Đổi ý nghĩa prompt thì tăng số này — verdict của
   phiên bản cũ sẽ bị hỏi lại thay vì tái dùng.

3. **Người duyệt** (`app.py`) — hai nút: đồng ý sửa / giữ nguyên.

## Chạy

```bash
# 1. xem trước bằng luật, không tốn API
python -m annotation.label_audit.run screen

# 2. để LLM phân xử (resume được: chạy lại là bỏ qua phần đã xong)
python -m annotation.label_audit.run adjudicate \
  --batch-size 8 --concurrency 6 --request-timeout 120 --attempts 3 --execute-llm

# 3. duyệt trên giao diện — A = đồng ý sửa, G = giữ nguyên
python -m annotation.label_audit.app --audit-dir annotation/data/label_audit --port 5001

# 4. xuất bản nhãn đã sửa sang thư mục riêng
python -m annotation.label_audit.run apply
```

Kết quả:

```text
annotation/data/label_audit/
  screened.jsonl      # toàn bộ nghi ngờ từ tầng luật
  verdicts.jsonl      # journal phán xử của LLM (append-only, dùng để resume)
  findings.jsonl      # đề xuất sửa đã được xác nhận, đầu vào của giao diện
  audit_report.json
  decisions.json      # quyết định của người dùng
annotation/data/label_fixed/
  part1/{notes,labels}/   # bản sao đã áp dụng quyết định
  gt2/{notes,labels}/
  manifest.json
```

`apply` chỉ ghi những nguồn thực sự có thay đổi, giữ nguyên `position`, `text`, `type` và số
lượng entity — chỉ trường `assertions` được đổi.

## Quy mô đo được (31/07/2026)

- part1: 2.726 entity → 349 nghi ngờ (12,8%)
- gt2: 15.444 entity → 3.036 nghi ngờ (19,7%), trong đó **330 ca KẾT_QUẢ_XÉT_NGHIỆM mang
  assertion** — vi phạm luật tuyệt đối, không cần bàn.
- Tổng 3.385 nghi ngờ, 333 thuộc nhóm cơ học. Còn ~3.050 ca cần LLM ≈ 382 lần gọi
  (batch 8) ≈ 30 phút ở concurrency 6.

Trên mẫu thử 40 ca, LLM xác nhận 26 và giữ nguyên 14 — ví dụ nó **bác** đề xuất bỏ
`isHistorical` khi câu có "trước đây", "trong khoảng vài năm nay", nhưng **chấp nhận** bỏ khi
`isHistorical` chỉ được suy từ heading "Thuốc trước khi nhập viện" (CANONICAL §4.1 cấm suy
assertion từ heading).
