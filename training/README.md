# training — Fine-tune NER **multi-task** (type + assertion) song song 2 model

Train **ViHealthBERT-syllable** + **XLM-R base** cho bước NER, **multi-task**:
- head TYPE: 5 loại (TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, THUỐC).
- head ASSERTION: 3 nhãn nhị phân (isNegated/isFamily/isHistorical) — 1 model ra CẢ type
  lẫn assertion → nộp được ngay (candidates để rỗng, làm bước sau).

Code chạy trên **server GPU**; dataset giữ private (zip + upload riêng, KHÔNG commit).

File chính: `model.py` (MultiTaskNER), `train_mtl.py` (train), `predict.py` (inference →
output format thi), `build_ner_dataset.py` (tạo dataset), `train_all.sh` (train song song).

## Vì sao 2 model này
Data của ta trộn Anh-Việt nặng (tên thuốc EN, sig `po bid`), syllable-level, nhiều typo:
- **ViHealthBERT-syllable** (`demdecuong/vihealthbert-base-syllable`): domain y tế VN
  (25M câu health), base 135M. Dùng bản **syllable** nên train ở mức syllable, KHÔNG cần
  VnCoreNLP word-segment.
- **XLM-R base** (`xlm-roberta-base`): subword đa ngữ → nuốt tên thuốc EN + sig tự nhiên,
  robust typo/code-mix, không cần segment.

> Đổi model chỉ cần sửa `--model` (hoặc sửa `train_all.sh`). VietMed-NER phobert-v2 /
> CafeBERT cũng chạy được với script này (CafeBERT cần GPU lớn hơn).

## Dữ liệu — nhãn BIO, config-driven, không phụ thuộc offset_mapping
`build_ner_dataset.py` đổi `train.jsonl` (text + entity char-span) thành cửa sổ BIO
token-level:
- val/test tách Ở MỨC NOTE từ nguồn **LLM** (sát thật hơn); train = LLM còn lại + no-LLM.
- `train_ner.py` align subword THỦ CÔNG (nhãn ở subword đầu, -100 phần còn lại) nên chạy
  được CẢ tokenizer slow (ViHealthBERT/PhoBERT) lẫn fast (XLM-R).

### (Đã build sẵn trên máy local, chỉ cần upload)
```
training/dataset/
  train.jsonl  validation.jsonl  test.jsonl   # {tokens:[...], ner_tags:[...]}
  labels.txt                                   # 11 nhãn BIO
```
Muốn build lại (nếu sinh thêm data):
```bash
python -m training.build_ner_dataset \
    --llm data_gen/generated/train_v1 \
    --extra data_gen/generated/no-llm_train_v1 \
    --out training/dataset --val 0.1 --test 0.1
```

## Quy trình trên server GPU

1. **Commit code** (không có dataset — đã .gitignore) lên GitHub, pull về server.
2. **Upload dataset**: nén `training/dataset` thành `dataset.zip` (đã tạo sẵn ở local),
   tải lên server, giải nén vào `training/dataset/`:
   ```bash
   cd <repo>/training && unzip dataset.zip -d dataset   # -> training/dataset/*.jsonl
   ```
3. **Cài môi trường** (khớp CUDA server; cài torch theo hướng dẫn pytorch.org nếu cần):
   ```bash
   pip install -r training/requirements.txt
   ```
4. **Train 2 model — 2 tiến trình độc lập, 2 GPU** (khuyến nghị để theo dõi log riêng):
   ```bash
   cd <repo>
   # terminal 1 — GPU 0
   bash training/train_vihealthbert.sh          # log: runs/vihealthbert/train.log
   # terminal 2 — GPU 1
   bash training/train_xlmr.sh                   # log: runs/xlmr/train.log
   ```
   Tùy chỉnh qua biến môi trường: `GPU=0 EPOCHS=5 BS=32 bash training/train_vihealthbert.sh`.
   Chạy nền: thêm `nohup ... &` (log vẫn ghi qua `tee`), theo dõi `tail -f runs/*/train.log`.

   Hoặc 1 lệnh chạy cả hai nền (GPU0+GPU1): `bash training/train_all.sh`.

## Chạy 1 model lẻ
```bash
python -m training.train_mtl --data_dir training/dataset \
    --model demdecuong/vihealthbert-base-syllable --out runs/vihealthbert --fp16
python -m training.train_mtl --data_dir training/dataset \
    --model xlm-roberta-base --out runs/xlmr --fp16
```
Tham số: `--epochs 4 --bs 16 --lr 3e-5 --max_len 256 --assert_weight 1.0 --fp16`.

## Sinh output nộp bài
```bash
# chọn model tốt hơn (theo NER_F1/ASSERT_F1), chạy trên 100 file input/
python -m training.predict --model_dir runs/xlmr/best \
    --input_dir input --out_dir output
cd output && zip -r ../output.zip . && cd ..     # -> output.zip để nộp vòng 1
```
`predict.py` xuất mỗi file `output/<id>.json` đúng schema thi (text/position/type/
assertions/candidates). **candidates để rỗng** (baseline mạnh: GT rỗng + pred rỗng = 1 điểm;
bước chuẩn hóa ICD/RxNorm làm sau).

## Output train
```
runs/<model>/best/              model + tokenizer + heads.pt + mtl_meta.json
runs/<model>/test_metrics.json  NER_F1 + ASSERT_F1 trên test
```
Log in `classification_report` chi tiết F1 theo từng loại entity.

## Lưu ý
- Model tải từ HuggingFace Hub -> server cần internet (hoặc pre-download rồi trỏ `--model`
  vào path local). Nếu qua proxy: `export https_proxy=...`.
- ViHealthBERT tokenizer có thể là slow -> script tự fallback `use_fast=False`, vẫn train
  bình thường nhờ align thủ công.
- Model này ra **type + assertion** (multi-task). Còn **candidates** (ICD cho CHẨN_ĐOÁN,
  RxNorm cho THUỐC) là bước chuẩn hóa riêng — RxNorm đã có ở `dictionary/vocabulary_download_*`
  (định dạng OMOP: CONCEPT.csv, DRUG_STRENGTH.csv…), ICD-10 VN ở `dictionary/DM ICD10-*.xlsx`.
  Bước này để rỗng candidates trước để nộp thử nghiệm nhanh.
