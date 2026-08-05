# Part 3 rule/prompt evaluation

Benchmark này kiểm định hai giả thuyết:

1. Nghiệp vụ trong `business_rules/CANONICAL.md` có phản ánh đúng convention Part 3 không.
2. Cách biên dịch nghiệp vụ thành prompt có giúp LLM thực thi tốt hơn việc chỉ đọc đề bài hoặc
   đọc nguyên văn rulebook không.

Đây không phải luồng tối ưu trực tiếp nhãn V66ab. LLM gán độc lập từ RAW và không được nhận
nhãn V66ab, `RULES_S.md`, điểm leaderboard hay archive lịch sử.

## Thiết kế

Tập cố định gồm 10 file:

```text
3, 5, 11, 22, 24, 31, 32, 36, 65, 92
```

Nhóm này được chọn từ số thao tác nghiệp vụ trong `annotation/evidence/ops_index.jsonl`, exact
diff giữa V66ab và các artifact lịch sử, sau đó bổ sung file 65 để phủ winner V66b. Lý do từng
file nằm trong `benchmark.json` và được ghi lại ở `selection_report.json` sau khi chạy.

Một run tạo năm ZIP:

| ZIP | Ý nghĩa |
|---|---|
| `00_reference_v66ab_10.zip` | Nhãn V66ab trên 10 file — baseline/reference |
| `01_official_only_10.zip` | LLM chỉ đọc định nghĩa chính thức |
| `02_canonical_dump_10.zip` | LLM đọc định nghĩa + nguyên văn `CANONICAL.md` |
| `03_compiled_10.zip` | Nghiệp vụ được biên dịch thành workflow sáu pass |
| `04_compiled_verified_10.zip` | Workflow sáu pass + critic độc lập |

Mọi ZIP có đủ `labels/1.json`…`labels/100.json`. Chỉ 10 file benchmark có nhãn; 90 file còn lại
là `[]`. Do đó trần điểm xấp xỉ 10 và chỉ được so sánh các ZIP trong cùng run.

## Chạy một lệnh

Điều kiện:

- `.env` ở root đã có `CHATGPT_ENDPOINT`, `CHATGPT_API_KEY`, `CHATGPT_API_VERSION`,
  `CHATGPT_DEPLOYMENT`, `CHATGPT_MODEL`;
- package `openai` đã cài — repo hiện dùng Azure OpenAI;
- artifact/rulebook vẫn khớp `business_rules/artifacts/MANIFEST.json`.

Từ root dự án:

```bash
python -m annotation.part3_eval.run
```

Lệnh mặc định gọi Azure OpenAI trực tiếp, giống `annotation/llm_annotate.py`; nó không tự set
proxy. Proxy Viettel chỉ dùng khi tải dataset/model, không dùng cho Azure API.

Chỉ khi hạ tầng thực sự yêu cầu proxy cho API mới truyền rõ:

```bash
python -m annotation.part3_eval.run --proxy-url http://host:port
```

Chạy lại cùng lệnh là an toàn: response được cache theo hash đầy đủ của model + messages +
temperature + reasoning effort. Lệnh chỉ gọi lại request còn thiếu hoặc prompt đã thay đổi.

Run đầy đủ có một preflight ngắn rồi 40 request tuần tự: 10 file × ba lượt labeler và 10 lượt
critic. Cấu hình mặc định dùng `temperature=0`, `reasoning_effort=none`, structured JSON và
timeout cứng 300 giây/request; nếu Azure deployment không hỗ trợ một tham số, client tự fallback
hẹp mà không thay nội dung prompt. Mỗi log file ghi cả deadline và thời gian thực tế.

## Output

Mặc định:

```text
annotation/part3_eval/runs/v1_10files/
├── REPORT.md
├── manifest.json
├── selection_report.json
├── scores_template.csv
├── prompt_templates/
├── cache/
├── audits/
├── labels/
├── diffs/
└── submissions/
```

Chỉ nộp ZIP có `gate PASS` trong `REPORT.md`. Gate xác minh:

- đủ chính xác 100 entry `labels/N.json`;
- 90 file ngoài subset rỗng;
- mọi `raw[start:end] == text`;
- không duplicate/overlap;
- type/assertion/candidate đúng schema;
- candidate LLM trả về phải tồn tại trong ICD-10 hoặc RxNorm local.

LLM chỉ trả `line_id`, surface nguyên văn và occurrence. Code tính offset RAW; không fuzzy-match,
không tự lan một prediction ra mọi occurrence.

## Thứ tự nộp

Nộp cả năm ZIP theo thứ tự tên file. Với mỗi lượt, ghi lại:

- score;
- WER;
- J_assertion;
- J_candidates.

Điền các số vào `scores_template.csv` hoặc gửi nguyên bảng cho người phân tích.

Diễn giải:

- `official_only → canonical_dump`: giá trị thông tin của nghiệp vụ tổng hợp;
- `canonical_dump → compiled`: giá trị của cách tổ chức prompt;
- `compiled → compiled_verified`: giá trị của critic;
- `compiled_verified → reference_v66ab`: khoảng cách còn lại với artifact tốt nhất.

Candidate normalization là stage riêng theo `business_rules/DATASET_CONTRACT.md`. Khi đánh giá
khả năng hiểu nghiệp vụ NER, phải đọc WER và J_assertion bên cạnh tổng điểm/J_candidates.

## Lệnh phụ

Chỉ kiểm tra input, selection, baseline ZIP và prompt templates; không gọi API:

```bash
python -m annotation.part3_eval.run --dry-run
```

Chạy một số variant:

```bash
python -m annotation.part3_eval.run --variants official_only,compiled
```

Override Azure deployment:

```bash
python -m annotation.part3_eval.run --deployment <deployment-name>
```

Chỉ tăng reasoning effort khi cần một probe riêng và hạ tầng không cắt request dài:

```bash
python -m annotation.part3_eval.run --reasoning-effort high
```

Có thể đặt timeout ngắn hơn để phát hiện lỗi nhanh:

```bash
python -m annotation.part3_eval.run --request-timeout 60
```
