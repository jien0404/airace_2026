# Thử nghiệm baseline NER: VietMed-NER (phobert-base-v2)

Chạy model [`leduckhai/VietMed-NER`](https://huggingface.co/leduckhai/VietMed-NER/tree/main/phobert-base-v2-VietMed-NER)
cho **bước NER** của bài toán (xem `DE_BAI.md`), để xem chất lượng bóc span/type
trước khi đầu tư pipeline đầy đủ.

## 1. Cài môi trường (máy có GPU)

```bash
cd test_baseline_models/vietmed_ner
python -m venv .venv && source .venv/bin/activate
# Cài torch hợp với CUDA của máy, ví dụ CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# (tuỳ chọn) tách từ tiếng Việt — cần Java 8+:
# pip install py_vncorenlp
```

## 2. Chạy

```bash
# Không tách từ (đơn giản, không cần Java):
python run_ner.py

# Có tách từ VnCoreNLP (khuyến nghị, cần Java + py_vncorenlp):
python run_ner.py --segment
```

Mặc định đọc `../../input/*.txt`, ghi kết quả vào `./output/`. Chạy CPU cũng được
(chỉ 100 file, ngắn) nhưng GPU nhanh hơn: script tự dò `cuda`.

## 3. Output

Với mỗi `N.txt` sinh 2 file trong `output/`:

- `N.raw.json` — entity với **nhãn gốc VietMed** (36 loại). Dùng file này để đánh
  giá thực chất khả năng bóc span của model.
- `N.json` — đã ánh xạ tạm sang **5 loại của đề** + format object
  (`text, position, type, assertions=[], candidates=[]`). Chỉ để nhìn nhanh.

## 4. Điểm cần lưu ý khi đọc kết quả

Bộ nhãn VietMed **không** khớp 1-1 với 5 loại của đề. Ánh xạ hiện tại (sửa trong
`LABEL_MAP` ở đầu `run_ner.py`):

| VietMed | → Đề |
|---|---|
| `DRUGCHEMICAL` | `THUỐC` |
| `DISEASESYMTOM` | `TRIỆU_CHỨNG` ⚠️ |
| `DIAGNOSTICS` | `TÊN_XÉT_NGHIỆM` |
| `UNITCALIBRATOR` | `KẾT_QUẢ_XÉT_NGHIỆM` |
| còn lại (ORGAN, TREATMENT, SURGERY, ...) | bỏ |

⚠️ **Vấn đề cốt lõi:** VietMed gộp *bệnh* và *triệu chứng* vào chung
`DISEASESYMTOM`, trong khi đề tách `TRIỆU_CHỨNG` vs `CHẨN_ĐOÁN` — và metric
**phạt kép nếu sai type**. Nên riêng model này *không thể* tự phân biệt 2 loại
đó; sẽ cần thêm 1 tầng phân loại (rule/classifier) phía sau. Đây là hạn chế lớn
nhất cần đánh giá.

Các quan sát khác đáng ghi lại khi review output:
- `KẾT_QUẢ_XÉT_NGHIỆM` (số + đơn vị) có được `UNITCALIBRATOR` bắt đúng không?
- Tên thuốc tiếng Anh (`amlodipine 10 mg po daily`) có bị cắt vụn không?
- Model có bắt nhầm nhiều thứ ngoài 5 loại (ORGAN, TREATMENT...) không?

## 5. Chi tiết kỹ thuật

- Subfolder của model trên HF **chỉ có** `config.json` + `pytorch_model.bin`
  (không kèm file tokenizer), nên tokenizer nạp từ base `vinai/phobert-base-v2`
  (đổi bằng `--tokenizer`).
- Suy luận ở mức **word-level** (gán nhãn subtoken đầu của mỗi từ), không dùng
  `offset_mapping` → chạy được cả với slow `PhobertTokenizer`.
- `position` tính trên text gốc để `text` xuất ra khớp input (đề chấm `text` bằng
  WER, không chấm `position`).
- Tự chia chunk ≤ 250 subtoken (model max 256) cho file dài.
- Bước này **chỉ NER**: `assertions` và `candidates` để rỗng.
